import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional


class CogPonderDataModule(pl.LightningDataModule):

    def __init__(
        self,
        datasets: TensorDataset | Dataset | Dict[str, Dataset],
        train_ratio=.75,
        batch_size=4,
        shuffle=False,
        num_workers: int = -1,
        dataset_name: Optional[str] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['datasets'], logger=False)

        self.dataset_name = dataset_name

        if isinstance(datasets, Dataset):
            self.dataset = datasets
        elif isinstance(datasets, dict) and len(datasets.keys()) == 1:
            self.dataset_name, self.dataset = datasets.popitem()
        elif isinstance(datasets, dict):
            # TODO support multitask
            raise ValueError('Multitask datasets are not supported yet.')

        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self):

        # FIXME: index 5 is response_steps (see mappings in the dataset class)
        response_steps = self.dataset[:][5]

        # remove invalid trials (rt <= 0)
        valid_trials = (response_steps > 0)
        self.dataset = TensorDataset(*self.dataset[valid_trials])

        trial_ids = self.dataset[:][0]
        n_trials = torch.unique(trial_ids).shape[0]

        # test/train split (FIXME: should it pick the first n trials, not trial 1..n)
        train_ids = torch.where(trial_ids <= n_trials * self.train_ratio)[0].tolist()
        test_ids = torch.where(trial_ids > n_trials * self.train_ratio)[0].tolist()

        self.train_dataset = Subset(self.dataset, train_ids)
        self.test_dataset = Subset(self.dataset, test_ids)

    def _dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
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
