import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset


class NBackDataModule(pl.LightningDataModule):

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

        X, trial_types, is_targets, responses, response_steps = dataset[0]
        self.dataset = TensorDataset(X, trial_types, is_targets, responses, response_steps)

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
        return self._dataloader(self.train_dataset)
    
    def val_dataloader(self):
        return self._dataloader(self.test_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)
