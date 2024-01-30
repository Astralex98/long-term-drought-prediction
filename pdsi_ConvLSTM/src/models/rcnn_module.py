import sys
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.autograd import Variable

from src.models.components.conv_block import ConvBlock
from src.utils.metrics import metrics_celled
from src.utils.plotting import make_heatmap


class ScaledTanh(nn.Module):
    def __init__(self, coef: int = 10):
        super().__init__()
        self.c = coef

    def forward(self, x):
        output = torch.mul(torch.tanh(x), self.c)
        return output


class RCNNModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        mode: str = "regression",
        embedding_size: int = 16,
        hidden_state_size: int = 32,
        kernel_size: int = 3,
        groups: int = 1,
        dilation: int = 1,
        n_cells_hor: int = 200,
        n_cells_ver: int = 250,
        history_length: int = 1,
        periods_forward: int = 1,
        batch_size: int = 1,
        num_of_additional_features: int = 0,
        boundaries: List[int] = [-2],
        num_classes: int = 2,
        values_range: int = 10,
        dropout: float = 0.0,
        lr: float = 0.003,
        weight_decay: float = 0.0,
    ):
        super(self.__class__, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        self.history_length = history_length
        self.periods_forward = periods_forward
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.num_of_features = num_of_additional_features + 1
        self.tanh_coef = values_range
        # number of bins for pdsi
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.num_class = num_classes
        self.boundaries = torch.tensor(boundaries).cuda()

        self.emb_size = embedding_size
        self.hid_size = hidden_state_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups

        self.global_avg = None
        self.saved_predictions = None
        self.saved_targets = None

        self.embedding = nn.Sequential(
            ConvBlock(
                self.num_of_features * history_length,
                self.emb_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                dilation=self.dilation,
                groups=self.groups,
            ),
            nn.ReLU(),
            ConvBlock(
                self.emb_size,
                self.emb_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
        )

        self.f_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Sigmoid(),
        )
        self.i_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Sigmoid(),
        )
        self.c_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Tanh(),
        )
        self.o_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Sigmoid(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                self.hid_size,
                self.periods_forward,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            ScaledTanh(self.tanh_coef),
        )

        self.final_classify = nn.Sequential(
            ConvBlock(
                self.hid_size,
                self.num_class,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            ),
            nn.Sigmoid(),
        )

        self.register_buffer(
            "prev_state_h",
            torch.zeros(
                self.batch_size,
                self.hid_size,
                self.n_cells_hor,
                self.n_cells_ver,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "prev_state_c",
            torch.zeros(
                self.batch_size,
                self.hid_size,
                self.n_cells_hor,
                self.n_cells_ver,
                requires_grad=False,
            ),
        )

        self.mode = mode
        # loss
        if self.mode == "regression":
            self.criterion = nn.MSELoss()
            self.loss_name = "MSE"
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.loss_name = "CrossEntropy"

    def forward(self, x: torch.Tensor):
        prev_c = self.prev_state_c
        prev_h = self.prev_state_h
        x = self.dropout(x)
        x_emb = self.embedding(x)
        if x_emb.shape[0] < self.batch_size:
            x_emb = torch.nn.functional.pad(
                x_emb,
                pad=(0, 0, 0, 0, 0, 0, 0, self.batch_size - x_emb.shape[0]),
                value=0,
            )
        x_and_h = torch.cat([prev_h, x_emb], dim=1)

        f_i = self.f_t(x_and_h)
        i_i = self.i_t(x_and_h)
        c_i = self.c_t(x_and_h)
        o_i = self.o_t(x_and_h)

        next_c = prev_c * f_i + i_i * c_i
        next_h = torch.tanh(next_c) * o_i

        assert prev_h.shape == next_h.shape
        assert prev_c.shape == next_c.shape

        if self.mode == "regression":
            prediction = self.final_conv(next_h)
        elif self.mode == "classification":
            prediction = self.final_classify(next_h)
        self.prev_state_c = next_c
        self.prev_state_h = next_h

        return prediction

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        # padding of last batch
        if y.shape[0] < self.batch_size:
            y = torch.nn.functional.pad(
                y, pad=(0, 0, 0, 0, 0, 0, 0, self.batch_size - y.shape[0]), value=0
            )
        # checking last (forward) value of target
        loss = self.criterion(preds, y[:, -1, :, :])
        return loss, preds, y[:, -1, :, :]

    def class_baseline(self, batch: Any):
        x, y = batch
        # return most frequent class along history_dim
        x_binned = torch.bucketize(x, self.boundaries)
        most_freq_values, most_freq_indices = torch.mode(x_binned, dim=1)
        return most_freq_values

    def on_after_backward(self) -> None:
        self.prev_state_c.detach_()
        self.prev_state_h.detach_()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        all_targets = outputs[0]["targets"]
        all_preds = outputs[0]["preds"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)
        all_preds = torch.softmax(all_preds, dim=1)
        all_preds = all_preds[:, 1, :, :]
        rocauc_table, ap_table, f1_table, acc_table, thr = metrics_celled(
            all_targets, all_preds
        )
        # log metrics
        if self.mode == "classification":
            self.log(
                "train/convlstm/f1_median",
                torch.median(f1_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train/convlstm/ap_median",
                torch.median(ap_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train/convlstm/rocauc_median",
                torch.median(rocauc_table),
                on_epoch=True,
                prog_bar=True,
            )

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        all_targets = outputs[0]["targets"]
        all_preds = outputs[0]["preds"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)

        all_preds = torch.softmax(all_preds, dim=1)
        all_preds = all_preds[:, 1, :, :]
        rocauc_table, ap_table, f1_table, acc_table, thr = metrics_celled(
            all_targets, all_preds
        )
        # log metrics
        if self.mode == "classification":
            self.log(
                "val/convlstm/f1_median",
                torch.median(f1_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/convlstm/ap_median",
                torch.median(ap_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/convlstm/rocauc_median",
                torch.median(rocauc_table),
                on_epoch=True,
                prog_bar=True,
            )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        baseline = self.class_baseline(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "baseline": baseline}

    def test_epoch_end(self, outputs: List[Any]):
        all_targets = outputs[0]["targets"]
        all_baselines = outputs[0]["baseline"]
        all_preds = outputs[0]["preds"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)
            all_baselines = torch.cat((all_baselines, outputs[i]["baseline"]), 0)

        # remove padded values
        init_len = all_baselines.shape[0]
        all_preds = all_preds[:init_len, :, :, :]
        all_targets = all_targets[:init_len, :, :]

        all_preds = torch.softmax(all_preds, dim=1)
        # log confusion matrix
        preds_for_cm = torch.argmax(all_preds, dim=1)
        # self.logger.experiment.log_confusion_matrix(torch.flatten(all_targets), torch.flatten(preds_for_cm))
        # probability of first class
        all_preds = all_preds[:, 1, :, :]

        self.saved_predictions = all_preds
        self.saved_targets = all_targets

        # global baseline
        all_global_baselines = self.global_avg.to("cuda:0")
        all_global_baselines = all_global_baselines.unsqueeze(0).repeat(
            len(all_targets), 1, 1
        )
        all_global_baselines = torch.where(all_global_baselines > 0, 100.0, 0.0)
        all_global_baselines = all_global_baselines.double()
        # all zeros baseline - no droughts at all
        all_zeros = torch.zeros(
            all_preds.shape[0],
            all_preds.shape[1],
            all_preds.shape[2],
            dtype=torch.long,
        ).to("cuda:0")
        all_zeros = all_zeros.double()
        rocauc_table, ap_table, f1_table, acc_table, thr = metrics_celled(
            all_targets, all_preds, "test"
        )
        rocauc_path = make_heatmap(rocauc_table, filename="rocauc_spatial.png")
        torch.save(rocauc_table, "rocauc_table.pt")
        if self.logger:
            self.logger.experiment.log_image(rocauc_path)
        (
            rocauc_table_zeros,
            ap_table_zeros,
            f1_table_zeros,
            acc_table_zeros,
            thr,
        ) = metrics_celled(all_targets, all_zeros, "test")
        (
            rocauc_table_global,
            ap_table_global,
            f1_table_global,
            acc_table_global,
            thr,
        ) = metrics_celled(all_targets, all_global_baselines, "test")

        # log metrics
        if self.mode == "classification":
            self.log(
                "test/convlstm/f1_median",
                torch.median(f1_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/convlstm/ap_median",
                torch.median(ap_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/convlstm/rocauc_median",
                torch.median(rocauc_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/convlstm/accuracy_median",
                torch.median(acc_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/global/f1_median",
                torch.median(f1_table_global),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/global/ap_median",
                torch.median(ap_table_global),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/global/rocauc_median",
                torch.median(rocauc_table_global),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/global/accuracy_median",
                torch.median(acc_table_global),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/zeros/f1_median",
                torch.median(f1_table_zeros),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/zeros/ap_median",
                torch.median(ap_table_zeros),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/zeros/rocauc_median",
                torch.median(rocauc_table_zeros),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/zeros/accuracy_median",
                torch.median(acc_table_zeros),
                on_epoch=True,
                prog_bar=True,
            )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
