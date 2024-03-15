from typing import Any, Tuple, Mapping

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from utils.common import instantiate_from_config


class lamavaeDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        train_config: str,
        val_config: str=None
    ) -> "lamavaeDataModule":
        super().__init__()
        self.train_config = OmegaConf.load(train_config)
        self.val_config = OmegaConf.load(val_config) if val_config else None

    def load_dataset(self, config: Mapping[str, Any]) -> Dataset:
        dataset = instantiate_from_config(config["dataset"])
        return dataset

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.load_dataset(self.train_config)
            if self.val_config:
                self.val_dataset = self.load_dataset(self.val_config)
            else:
                self.val_dataset = None, None
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, **self.train_config["data_loader"]
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            return None
        return DataLoader(
            dataset=self.val_dataset, **self.val_config["data_loader"]
        )
