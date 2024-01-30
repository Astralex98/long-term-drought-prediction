from earthformer.datasets.pdsi.data_utils import create_celled_data
from earthformer.config import cfg
from omegaconf import OmegaConf

import pathlib
from typing import Optional, Tuple, List

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class Dataset_RNN(Dataset):
    """
    Simple Torch Dataset for many-to-many RNN
        celled_data: source of data,
        start_date: start date index,
        end_date: end date index,
        periods_forward: number of future periods for a target,
        history_length: number of past periods for an input,
        transforms: input data manipulations
    """

    def __init__(
        self,
        celled_data: torch.Tensor,
        celled_features_list: List[torch.Tensor],
        start_date: int,
        end_date: int,
        periods_forward: int,
        history_length: int,
        boundaries: Optional[List[int]],
        mode,
        normalize: bool,
        moments: Optional[List[None]],
        global_avg: Optional[List[None]],
    ):
        self.data = celled_data[start_date:end_date, :, :]
        self.features = [
            feature[start_date:end_date, :, :]
            for feature in celled_features_list
        ]
        self.periods_forward = periods_forward
        self.history_length = history_length
        self.mode = mode
        self.target = self.data
        
        # bins for pdsi
        self.boundaries = boundaries
        if self.mode == "classification":
            
            # 1 is for drought
            self.target = len(boundaries) - torch.bucketize(self.target, self.boundaries)
            if len(global_avg) > 0:
                self.global_avg = global_avg
            else:
                self.global_avg, _ = torch.mode(self.target, dim=0)
                
        # normalization
        if moments:
            self.moments = moments
            if normalize:
                self.data = (self.data - self.moments[0][0]) / (
                    self.moments[0][1] - self.moments[0][0]
                )
                for i in range(1, len(self.moments)):
                    self.features[i - 1] = (
                        self.features[i - 1] - self.moments[i][0]
                    ) / (self.moments[i][1] - self.moments[i][0])
        else:
            self.moments = []
            if normalize:
                self.data = (self.data - torch.min(self.data)) / (
                    torch.max(self.data) - torch.min(self.data)
                )
                self.moments.append((torch.min(self.data), torch.max(self.data)))
                for i in range(len(self.features)):
                    self.features[i] = (
                        self.features[i] - torch.min(self.features[i])
                    ) / (torch.max(self.features[i]) - torch.min(self.features[i]))
                    self.moments.append(
                        (torch.min(self.features[i]), torch.max(self.features[i]))
                    )

    def __len__(self):
        return len(self.data) - self.periods_forward - self.history_length

    def __getitem__(self, idx):
        
        input_tensor = self.data[idx : idx + self.history_length]
        for feature in self.features:
            input_tensor = torch.cat(
                (input_tensor, feature[idx : idx + self.history_length]), dim=0
            )
                    
        # many-to-one
        trg_start_idx = idx + self.history_length + self.periods_forward - 1
        trg_end_idx = idx + self.history_length + self.periods_forward
        target = self.target[trg_start_idx:trg_end_idx]
            
       # many-to-many case
#         target = self.target[
#             idx + self.history_length : idx + self.history_length + self.periods_forward
#         ]

        return (input_tensor, target)


class WeatherDataModule(LightningDataModule):
    """LightningDataModule for Weather dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        oc_file: str = "cfg.yaml",
        mode: str = "classification",
        data_dir: str = cfg.datasets_dir,
        dataset_name: str = "pdsi_MadhaPradesh.csv",
        left_border: int = 0,
        down_border: int = 0,
        right_border: int = 2000,
        up_border: int = 2500,
        time_col: str = "date",
        event_col: str = "value",
        x_col: str = "x",
        y_col: str = "y",
        train_val_test_split: Tuple[float] = (0.8, 0.1, 0.1),
        periods_forward: int = 5,
        history_length: int = 3,
        data_start: int = 0,
        data_len: int = 100,
        feature_to_predict: str = "pdsi",
        num_of_additional_features: int = 0,
        additional_features: Optional[List[str]] = [],
        boundaries: Optional[List[int]] = [-2],
        patch_size: int = 8,
        normalize: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        # Read config file
        oc_from_file = OmegaConf.load(open(oc_file, "r"))
        oc = OmegaConf.create()
        oc = OmegaConf.merge(oc, oc_from_file)
        
        # Get data from config file
        self.mode = oc.data.mode
        self.dataset_name = oc.data.name
        self.feature_to_predict = oc.data.feature_to_predict
        self.boundaries = torch.Tensor(oc.data.boundaries)
        self.history_length = oc.data.history
        self.periods_forward = oc.data.forecast_hor
        
        self.data_dir = oc.data.dir
        self.left_border = left_border
        self.right_border = right_border
        self.down_border = down_border
        self.up_border = up_border
        self.h = 0
        self.w = 0
        self.time_col = time_col
        self.event_col = event_col
        self.x_col = x_col
        self.y_col = y_col

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_val_test_split = train_val_test_split
        self.data_start = data_start
        self.data_len = data_len
        self.num_of_features = num_of_additional_features + 1
        self.additional_features = additional_features
        self.patch_size = patch_size
        self.normalize = normalize

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        celled_data_path = pathlib.Path(self.data_dir, "celled", self.dataset_name)
        if not celled_data_path.is_file():
            celled_data = create_celled_data(
                self.data_dir,
                self.dataset_name,
                self.time_col,
                self.event_col,
                self.x_col,
                self.y_col,
            )
            torch.save(celled_data, celled_data_path)

        data_dir_geo = self.dataset_name.split(self.feature_to_predict)[1]
        for feature in self.additional_features:
            celled_feature_path = pathlib.Path(
                self.data_dir, "celled", feature + data_dir_geo
            )
            if not celled_feature_path.is_file():
                celled_feature = create_celled_data(
                    self.data_dir,
                    feature + data_dir_geo,
                    self.time_col,
                    self.event_col,
                    self.x_col,
                    self.y_col,
                )
                torch.save(celled_feature, celled_feature_path)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            celled_data_path = pathlib.Path(self.data_dir, "celled", self.dataset_name)
            celled_data = torch.load(celled_data_path)
            
            # make borders divisible by patch size
            h = celled_data.shape[1]
            self.h = h - h % self.patch_size
            w = celled_data.shape[2]
            self.w = w - w % self.patch_size
            celled_data = celled_data[:, :self.h, :self.w]
            celled_data = celled_data[
                self.data_start : self.data_start + self.data_len,
                self.down_border : self.up_border,
                self.left_border : self.right_border,
            ]
            
            # Add last channel dimension to accomplish NTHWC 
            celled_data = celled_data[:,:,:,None]
            
            # loading features
            celled_features_list = []
            data_dir_geo = self.dataset_name.split(self.feature_to_predict)[1]
            for feature in self.additional_features:
                celled_feature_path = pathlib.Path(
                    self.data_dir, "celled", feature + data_dir_geo
                )
                celled_feature = torch.load(celled_feature_path)
                
                # make borders divisible by patch size
                h = celled_feature.shape[1]
                h = h - h % self.patch_size
                w = celled_feature.shape[2]
                w = w - w % self.patch_size
                celled_feature = celled_feature[:, :h, :w]
                celled_feature = celled_feature[
                    self.data_start : self.data_start + self.data_len,
                    self.down_border : self.up_border,
                    self.left_border : self.right_border,
                ]
                celled_features_list.append(celled_feature)
                
            # Train dataloader
            train_start = 0
            train_end = int(self.train_val_test_split[0] * celled_data.shape[0])
            self.data_train = Dataset_RNN(
                celled_data,
                celled_features_list,
                train_start,
                train_end,
                self.periods_forward,
                self.history_length,
                self.boundaries,
                self.mode,
                self.normalize,
                [],
                [],
            )
            
            # Validation dataloader
            valid_end = celled_data.shape[0]
            self.data_val = Dataset_RNN(
                celled_data,
                celled_features_list,
                train_end - self.history_length,
                valid_end,
                self.periods_forward,
                self.history_length,
                self.boundaries,
                self.mode,
                self.normalize,
                self.data_train.moments,
                self.data_train.global_avg,
            )
            test_end = celled_data.shape[0]
            
            # Test dataloader
            self.data_test = Dataset_RNN(
                celled_data,
                celled_features_list,
                train_end - self.history_length,
                test_end,
                self.periods_forward,
                self.history_length,
                self.boundaries,
                self.mode,
                self.normalize,
                self.data_train.moments,
                self.data_train.global_avg,
            )
    
    @property
    def num_train_samples(self):
        return len(self.data_train)
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
