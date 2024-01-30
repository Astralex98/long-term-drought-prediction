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

import torch
import numpy as np
from typing import Tuple
from torchmetrics.classification import AUROC, AveragePrecision, BinaryF1Score, ROC, Accuracy
from torcheval.metrics.functional import binary_f1_score, binary_accuracy

class Metrics:
    """Class used for computing performance related metrics. Expects predictions /
    targets to be of shape [C, H, W] where H is latitude dimension and W is longitude
    dimension. Metrics are computed for each channel separately.

    Parameters
    ----------
    img_shape : Tuple[int]
        Shape of input image (resolution for fourcastnet)
    clim_mean_path : str, optional
        Path to total climate mean data, needed for ACC. By default "/era5/stats/time_means.npy"
    device : torch.device, optional
        Pytorch device model is on, by default 'cpu'
    """

    def __init__(
        self,
        img_shape: Tuple[int],
        num_classes: int = 2,
        device: torch.device = "cpu",
    ):
        self.img_shape = tuple(img_shape)
        self.device = device
        self.num_classes = num_classes

    def _check_shape(self, *args):
        # checks for shape [C, H, W]
        for x in args:
            assert x.ndim == 3
            assert tuple(x.shape[1:]) == self.img_shape

    def metrics_celled(self, all_targets, all_preds):
        if self.num_classes > 2:
            acc_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            acc = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                top_k=1,
                average="micro",
            ).to(self.device)
            acc_table = torch.tensor(
                [
                    [
                        acc(all_preds[:, :, x, y], all_targets[:, x, y])
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            acc_table = torch.nan_to_num(acc_table, nan=0.0)
            rocauc_table_macro = torch.zeros(self.img_shape[0], self.img_shape[1])
            rocauc_table_weighted = torch.zeros(self.img_shape[0], self.img_shape[1])
            rocauc = AUROC(
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
                thresholds=20,
            ).to(self.device)
            rocauc_table_macro = torch.tensor(
                [
                    [
                        rocauc(all_preds[:, :, x, y], all_targets[:, x, y].long())
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            rocauc = AUROC(
                task="multiclass",
                num_classes=self.num_classes,
                average="weighted",
                thresholds=20,
            ).to(self.device)
            rocauc_table_weighted = torch.tensor(
                [
                    [
                        rocauc(all_preds[:, :, x, y], all_targets[:, x, y].long())
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            rocauc_table_macro = torch.nan_to_num(rocauc_table_macro, nan=0.0)
            rocauc_table_weighted = torch.nan_to_num(rocauc_table_weighted, nan=0.0)
            thresholds = torch.zeros(self.img_shape[0], self.img_shape[1])

            return acc_table, rocauc_table_macro, rocauc_table_weighted, thresholds

        else:
            rocauc_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            rocauc = AUROC(task="binary") #, num_classes=1)
            rocauc_table = torch.tensor(
                [
                    [
                        rocauc(all_preds[:, x, y], all_targets[:, x, y])
                        for x in range(self.img_shape[0])
                    ]
                    for y in range(self.img_shape[1])
                ]
            )
            rocauc_table = torch.nan_to_num(rocauc_table, nan=0.0)

            ap_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            acc_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            f1_table = torch.zeros(self.img_shape[0], self.img_shape[1])
            thresholds = torch.zeros(self.img_shape[0], self.img_shape[1])

            ap = AveragePrecision(task="binary")
            roc = ROC(task="binary")
            for x in range(self.img_shape[0]):
                for y in range(self.img_shape[1]):
                    ap_table[x][y] = ap(all_preds[:, x, y], all_targets[:, x, y])
                    fpr, tpr, thr = roc(all_preds[:, x, y], all_targets[:, x, y])
                    j_stat = tpr - fpr
                    ind = torch.argmax(j_stat).item()
                    thresholds[x][y] = thr[ind].item()
                    #print(thresholds[x][y])
                    #f1 = BinaryF1Score(threshold=thresholds[x][y]).to(
                    #    self.device
                    #)
                    #f1_table[x][y] = f1(all_preds[:, x, y], all_targets[:, x, y])
                    f1_table[x][y] = binary_f1_score(all_preds[:, x, y], all_targets[:, x, y], threshold=thresholds[x][y])
                    acc_table[x][y] = binary_accuracy(all_preds[:, x, y], all_targets[:, x, y], threshold=thresholds[x][y])

            ap_table = torch.nan_to_num(ap_table, nan=0.0)
            f1_table = torch.nan_to_num(f1_table, nan=0.0)
            acc_table = torch.nan_to_num(acc_table, nan=0.0)

            return rocauc_table, ap_table, f1_table, acc_table, thresholds

