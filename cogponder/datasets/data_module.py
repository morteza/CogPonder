import os
from attr import attr
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset


class CogPonderDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset,
        train_ratio=.75,
        batch_size=4,
        randomized_split=False,
        num_workers=os.cpu_count()
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'], logger=False)

        # X, conditions, is_corrects, response_steps = dataset[0]
        self.dataset = TensorDataset(*dataset[0])

        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.randomized_split = randomized_split
        self.num_workers = num_workers

    def prepare_data(self) -> None:

        # only the first subject is used
        # TODO return all the subjects

        train_size = int(len(self.dataset) * self.train_ratio)
        test_size = len(self.dataset) - train_size

        if self.randomized_split:
            self.train_dataset, self.test_dataset = random_split(self.dataset, lengths=(train_size, test_size))
        else:
            self.train_dataset = Subset(self.dataset, torch.arange(0, train_size))
            self.test_dataset = Subset(self.dataset, torch.arange(train_size, len(self.dataset)))

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
