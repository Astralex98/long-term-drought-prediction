# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Script to carry out Fourcastnet inference"

import logging

import modulus
import modulus.sym
import numpy as np
import omegaconf
import torch
from modulus.sym.distributed.manager import DistributedManager
from modulus.sym.hydra import to_absolute_path
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.key import Key
from torch.utils.data import DataLoader, Sampler

from src.dataset import ERA5HDF5GridDataset
from src.fourcastnet import FourcastNetArch
from src.metrics import Metrics

logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)


def to_device(tensor_dict, device):
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in tensor_dict.items()
    }


@modulus.sym.main(config_path="conf", config_name="config_FCN")
def run(cfg: ModulusConfig) -> None:
    # load configuration
    chkpt_path = cfg.custom.chkpt_path
    res_path = "/long-term-drought-prediction/pdsi_FourCastNet/examples/fourcastnet/" + chkpt_path.split(".")[0] + "_test_metrics.txt"
    model_path = to_absolute_path(chkpt_path)
    # get device
    device = DistributedManager().device
    # load test data
    test_dataset = ERA5HDF5GridDataset(
        cfg.custom.test_dataset.data_path,  # Test data location e.g. /era5/20var/test
        chans=list(range(cfg.custom.n_channels)),
        tstep=cfg.custom.tstep,
        n_tsteps=1,  # set to one for inference
        thresholds=cfg.custom.thresholds,
        patch_size=cfg.arch.afno.patch_size,
    )

    m = Metrics(
        test_dataset.img_shape,
        num_classes=len(cfg.custom.thresholds) + 1,
        device=device,
    )
    # define input/output keys
    input_keys = [Key(k, size=test_dataset.nchans) for k in test_dataset.invar_keys]
    output_keys = [Key(k, size=test_dataset.nchans) for k in test_dataset.outvar_keys]



    # create model
    model = FourcastNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        img_shape=test_dataset.img_shape,
        patch_size=cfg.arch.afno.patch_size,
        embed_dim=cfg.arch.afno.embed_dim,
        depth=cfg.arch.afno.depth,
        num_blocks=cfg.arch.afno.num_blocks,
        num_classes=len(cfg.custom.thresholds) + 1,
    )
    
    n_class = len(cfg.custom.thresholds) + 1

    # load parameters
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    logging.info(f"Loaded model {model_path}")

    dataloader = DataLoader(
        dataset=test_dataset,
        pin_memory=True,
        num_workers=1,
        worker_init_fn=test_dataset.worker_init_fn,
    )
    # run inference
    with torch.no_grad():
        for tstep, (invar, true_outvar, _) in enumerate(dataloader):
            invar = to_device(invar, device)
            true_outvar = to_device(true_outvar, device)
            pred_outvar_single = model(invar)
            if tstep == 0:
                all_targets = true_outvar[str(output_keys[0])]
                all_preds = pred_outvar_single[str(output_keys[0])]
            else:
                all_targets = torch.cat(
                    (all_targets, true_outvar[str(output_keys[0])]), 0
                )
                all_preds = torch.cat(
                    (all_preds, pred_outvar_single[str(output_keys[0])]), 0
                )

    # remove dim=1, which is equal to C=1 of prediction
    all_targets = torch.squeeze(all_targets, dim=1).int()
    print(torch.sum(all_targets))
    print(all_targets.shape)
    print(
        torch.sum(all_targets)
        / (all_targets.shape[0] * all_targets.shape[1] * all_targets.shape[2])
    )
    # normalizing probs of output
    all_preds = torch.softmax(all_preds, dim=1)
    if n_class == 2:
        all_preds = all_preds[:, 1, :, :]
        print(all_preds.shape)
        rocauc_table, ap_table, f1_table, acc_table, thresholds = m.metrics_celled(
            all_targets, all_preds
        )
    else:
        print(all_preds.shape)
        acc_table, rocauc_table_macro, rocauc_table_weighted, thresholds = m.metrics_celled(
            all_targets, all_preds
        )

    # logging results

    res_file = open(res_path, "w")
    res_file.write(f"test_data_path: {cfg.custom.test_dataset.data_path} \n")
    res_file.write(f"chkpt_path: {cfg.custom.chkpt_path} \n")
    res_file.write(f"test_data_path: {cfg.custom.test_dataset.data_path} \n")
    res_file.write(f"forward: {cfg.custom.tstep} \n")
    res_file.write(f"num_classes: {n_class} \n")
    
    if n_class == 2:
        logging.info(f"test/rocauc_median: {torch.median(rocauc_table)}")
        res_file.write(f"test/rocauc_median: {torch.median(rocauc_table)} \n")
        logging.info(f"test/ap_median: {torch.median(ap_table)}")
        res_file.write(f"test/ap_median: {torch.median(ap_table)} \n")
        logging.info(f"test/f1_median: {torch.median(f1_table)}")
        res_file.write(f"test/f1_median: {torch.median(f1_table)} \n")
        logging.info(f"test/acc_median: {torch.median(acc_table)}")
        res_file.write(f"test/acc_median: {torch.median(acc_table)} \n")
    else:
        logging.info(f"test/acc_median: {torch.median(acc_table)}")
        res_file.write(f"test/acc_median: {torch.median(acc_table)} \n")
        logging.info(f"test/rocauc_macro_median: {torch.median(rocauc_table_macro)}")
        res_file.write(f"test/rocauc_macro_median: {torch.median(rocauc_table_macro)} \n")
        logging.info(f"test/rocauc_weighted_median: {torch.median(rocauc_table_weighted)}")
        res_file.write(f"test/rocauc_weighted_median: {torch.median(rocauc_table_weighted)} \n")

    res_file.close()


if __name__ == "__main__":
    run()
