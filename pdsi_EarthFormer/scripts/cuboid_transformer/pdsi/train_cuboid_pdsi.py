import warnings
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
from torchmetrics.classification import BinaryAccuracy

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything


from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.pdsi.weather_datamodule import WeatherDataModule
import pickle 


_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")

class CuboidPDSIPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidPDSIPLModule, self).__init__()
        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="conv"
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only
        
        
        # Save losses
        
        # Arrays for losses
        self.train_losses = []
        self.val_losses = []
        
        # Save metrics
        self.boundaries_len = len(oc.data.boundaries)
        
        # Binary classification
        if (self.boundaries_len == 1):
            
            # Loss
            self.criterion = torch.nn.BCELoss()

            # Metrics
            self.valid_acc = BinaryAccuracy()
            self.valid_aucroc = torchmetrics.AUROC(task="binary")
            self.valid_f1 = torchmetrics.F1Score(task="binary")
            self.valid_ap = torchmetrics.AveragePrecision(task="binary")
            self.test_acc = BinaryAccuracy()
            self.test_aucroc = torchmetrics.AUROC(task="binary")
            self.test_f1 = torchmetrics.F1Score(task="binary")
            self.test_ap = torchmetrics.AveragePrecision(task="binary")
        
        # Multiclass classification
        elif (self.boundaries_len > 1):
            
            # Loss
            self.criterion = torch.nn.CrossEntropyLoss()
            
            # Accuracy
            self.valid_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.boundaries_len + 1)
            self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.boundaries_len + 1)

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            # oc = apply_omegaconf_overrides(oc, oc_from_file)
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        height = 64
        width = 64
        in_len = 10
        out_len = 10
        data_channels = 1
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'  # The method for initializing the first input of the decoder
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = 2
        cfg.initial_downsample_conv_layers = 2
        cfg.final_upsample_conv_layers = 1
        cfg.checkpoint_level = 2
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @staticmethod
    def get_layout_config():
        oc = OmegaConf.create()
        oc.in_len = 10
        oc.out_len = 10
        oc.layout = "NTHWC"  # The layout of the data, not the model
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 50
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 5
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "PDSI"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        cfg.use_wandb = False
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc

    @staticmethod
    def get_vis_config():
        oc = OmegaConf.create()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [0, ]
        oc.test_example_data_idx_list = [0, ]
        oc.eval_example_only = False
        return oc

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            
            # save
            default_root_dir=self.save_dir,
            
            # ddp
            accelerator="gpu",
            
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_pdsi_datamodule(oc_file_name, micro_batch_size: int = 1):
        dm = WeatherDataModule(oc_file = oc_file_name,
                               train_val_test_split = (0.7, 0.2, 0.1),
                               batch_size=micro_batch_size)
        return dm

    def forward(self, batch):
        
        in_seq, target_seq = batch
        
        # Since transformer predicts target for several periods simultaneously
        # then we get the final element of predicted sequence
        # Pred_seq shape - see cfg.yaml (model.target shape)
        pred_seq = self.torch_nn_module(in_seq)
        
        pred_seq = pred_seq[:, -1, :, :, :].unsqueeze(1)
        
        # Binary classification
        if (self.boundaries_len == 1):
            sigm = torch.nn.Sigmoid()
            
            # Get probabilities from raw logits
            probs = sigm(pred_seq)

            loss = self.criterion(probs, target_seq.to(torch.float32))
        
        # Multiclass classification
        elif (self.boundaries_len > 1):
            
            # Final classification head
            c_in = pred_seq.shape[1]
            c_out = self.boundaries_len + 1
            pred_seq = pred_seq [:, :, :, :, -1]
            target_seq = target_seq [:, -1, :, :, -1]
            
            final_classify = torch.nn.Sequential(torch.nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False,),
                                                 torch.nn.BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=False),
                                                 torch.nn.MaxPool2d(3, stride=1, padding=1, dilation=1),
                                                 ).cuda()
            
            
            probs = final_classify(pred_seq)
            
            loss = self.criterion(probs, target_seq)
            
        return probs, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss
    
    
    def training_epoch_end(self, outputs : list) -> None:    
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        print("[Epoch %d] Train: loss=%.3f"% (self.current_epoch + 1, loss))
        self.train_losses.append(float(loss))
    
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        B, T_out, H, W, C = y.shape
                
        y_hat, valid_loss = self(batch)
        
        self.log('val_loss', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Binary classification
        if (self.boundaries_len == 1):
            step_acc = self.valid_acc(y_hat, y)
            step_aucroc = self.valid_aucroc(y_hat, y)
            step_f1 = self.valid_f1(y_hat, y)
            step_ap = self.valid_ap(y_hat, y)
            
            self.log('valid_acc_step', step_acc, prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_aucroc_step', step_aucroc, prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_f1_step', step_f1, prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_ap_step', step_ap, prog_bar=True, on_step=True, on_epoch=False)
        
        # Multiclass classification
        elif (self.boundaries_len > 1):
            
            y = y[:, -1, :, :, -1]
            acc = self.valid_accuracy(y_hat, y)
            self.log('valid_accuracy_step', acc, prog_bar=True, on_step=True, on_epoch=False)
        
        return {"val_loss": valid_loss}

    def validation_epoch_end(self, outputs):
        
        aver_loss = sum(output['val_loss'] for output in outputs) / len(outputs)
        
        # Binary classification
        if (self.boundaries_len == 1):
            aver_acc = self.valid_acc.compute()
            aver_aucroc = self.valid_aucroc.compute()
            aver_ap = self.valid_ap.compute()
            aver_f1 = self.valid_f1.compute()
            
            self.log('valid_acc_epoch', aver_acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log('valid_aucroc_epoch', aver_aucroc, prog_bar=True, on_step=False, on_epoch=True)
            self.log('valid_ap_epoch', aver_ap, prog_bar=True, on_step=False, on_epoch=True)
            self.log('valid_f1_epoch', aver_f1, prog_bar=True, on_step=False, on_epoch=True)

            self.valid_acc.reset()
            self.valid_aucroc.reset()
            self.valid_ap.reset()
            self.valid_f1.reset()
        
        # Multiclass classification
        elif (self.boundaries_len > 1):
            aver_accuracy = self.valid_accuracy.compute()
            self.log('valid_accuracy_epoch', aver_accuracy, prog_bar=True, on_step=False, on_epoch=True)
            self.valid_accuracy.reset()
        
        
        self.log('valid_loss_epoch', aver_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_losses.append(float(aver_loss))

    def test_step(self, batch, batch_idx):
        x, y = batch
        B, T_out, H, W, C = y.shape
        
        y_hat, _ = self(batch)
        
        # Binary classification
        if (self.boundaries_len == 1):
            step_acc = self.test_acc(y_hat, y)
            step_aucroc = self.test_aucroc(y_hat, y)
            step_ap = self.test_ap(y_hat, y)
            step_f1 = self.test_f1(y_hat, y)
            
            self.log('test_acc_step', step_acc, prog_bar=True, on_step=True, on_epoch=False)
            self.log('test_aucroc_step', step_aucroc, prog_bar=True, on_step=True, on_epoch=False)
            self.log('test_ap_step', step_ap, prog_bar=True, on_step=True, on_epoch=False)
            self.log('test_f1_step', step_f1, prog_bar=True, on_step=True, on_epoch=False)
        
        # Multiclass classification
        elif (self.boundaries_len > 1):
            y = y[:, -1, :, :, -1]
            step_accuracy = self.test_accuracy(y_hat, y)
            self.log('test_accuracy_step', step_accuracy, prog_bar=True, on_step=True, on_epoch=False)
            
        return H, W

    def test_epoch_end(self, outputs):
        H, W = outputs[0]
        
        # Binary classification
        if (self.boundaries_len == 1):
            aver_acc = self.test_acc.compute()
            aver_aucroc = self.test_aucroc.compute()
            aver_ap = self.test_ap.compute()
            aver_f1 = self.test_f1.compute()

            self.log('test_acc_epoch', aver_acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log('test_aucroc_epoch', aver_aucroc, prog_bar=True, on_step=False, on_epoch=True)
            self.log('test_ap_epoch', aver_ap, prog_bar=True, on_step=False, on_epoch=True)
            self.log('test_f1_epoch', aver_f1, prog_bar=True, on_step=False, on_epoch=True)

            self.test_aucroc.reset()
            self.test_ap.reset()
            self.test_f1.reset()
        
        # Multiclass classification
        elif (self.boundaries_len > 1):
            aver_accuracy = self.test_accuracy.compute()
            self.log('test_accuracy_epoch', aver_accuracy, prog_bar=True, on_step=False, on_epoch=True)
            self.test_accuracy.reset()
            

    def save_vis_step_end(
            self,
            batch_idx: int,
            in_seq: torch.Tensor, target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            mode: str = "train"):

        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if batch_idx in example_data_idx_list:
                micro_batch_size = in_seq.shape[self.layout.find("N")]
                data_idx = int(batch_idx * micro_batch_size)
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    layout=self.layout,
                    plot_stride=1,
                    label=self.oc.logging.logging_prefix)
    
    def get_train_losses(self):
        return self.train_losses
    
    def get_val_losses(self):
        return self.val_losses
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_mnist', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on N-body MovingMNIST.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
    seed_everything(seed, workers=True)
    dm = CuboidPDSIPLModule.get_pdsi_datamodule(oc_file_name = args.cfg, micro_batch_size=micro_batch_size)
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidPDSIPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = CuboidPDSIPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    if args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        trainer.test(ckpt_path="best",
                     datamodule=dm)
        
        # Collect losses for train and val sets
        train_losses = pl_module.get_train_losses()
        val_losses = pl_module.get_val_losses()
        losses_dict = {"train_loss" : train_losses, "val_loss" : val_losses}
        
        with open('losses.pkl', 'wb') as f:
            pickle.dump(losses_dict, f)

if __name__ == "__main__":
    main()