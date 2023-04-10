import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data import Dataset
import numpy as np


class CogPonderDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset: Dataset,
        train_ratio=.75,
        batch_size=4,
        randomized_split=False,
        num_workers: int = -1
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'], logger=False)

        self.dataset = dataset

        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.randomized_split = randomized_split
        self.num_workers = num_workers

    def prepare_data(self):

        # remove invalid trials
        rts = self.dataset[:][5]
        self.dataset = TensorDataset(*self.dataset[rts > 0])

        # test/train split
        n_trials = torch.unique(self.dataset[:][0]).shape[0]
        train_idx = torch.where(self.dataset[:][0] <= n_trials * self.train_ratio)[0]
        test_idx = torch.where(self.dataset[:][0] > n_trials * self.train_ratio)[0]

        self.train_dataset = Subset(self.dataset, train_idx.tolist())
        self.test_dataset = Subset(self.dataset, test_idx.tolist())

    def _dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.randomized_split,
            num_workers=self.num_workers)

        return dataloader

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            raise ValueError('Dataset is not loaded yet. It will be loaded automatically. '
                             'If you need manual access, call prepare_data() first.')

        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            raise ValueError('Dataset is not loaded yet. It will be loaded automatically. '
                             'If you need manual access, call prepare_data() first.')

        return self._dataloader(self.test_dataset)

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            raise ValueError('Dataset is not loaded yet. It will be loaded automatically. '
                             'If you need manual access, call prepare_data() first.')

        return self._dataloader(self.test_dataset)
