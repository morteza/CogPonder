import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from pytorch_lightning.trainer.supporters import CombinedLoader


class NBackMockDataset(Dataset):
    """Human or mock N-back dataset -- a binary classification task/

    """

    def __init__(
        self,
        n_subjects,
        n_trials,
        n_stimuli,
        n_back=2,
        device='cpu'
    ):

        """Initialize the dataset.

        Args:
            n_subjects (int): Number of subjects.
            n_trials (int): Number of trials in the dataset.
            n_stimuli (int): Number of stimuli in the dataset.
            device (torch.device): Device to use for the dataset.
            n_back (int): Number of items in the N-back sequence. Default: 2.
        """
        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.n_stimuli = n_stimuli
        self.device = device
        self.n_back = n_back

        # Generate the mock data
        mock_data = self._generate_mock_data(n_subjects, n_trials, n_stimuli, n_back)
        self.X, self.responses, self.targets, self.response_times = mock_data

    def __len__(self):
        """Get the number of samples.
        """
        return self.n_subjects

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        return (self.X[idx, :],
                self.responses[idx],
                self.targets[idx],
                self.response_times[idx, :]
        )

    @classmethod
    def _generate_mock_data(cls, n_subjects, n_trials, n_stimuli, n_back=2):
        """[summary]
        # TODO required data columns:
        #   subject_index, trial_index, stimulus_index, accuracy, response_time

        Args:
            n_subjects (int): [description]
            n_trials (int): number of trials per subject including warm-up
            n_stimuli (int): [description]

        Returns:
            X: stimulus features
                shape: (n_subjects, n_trials - n_back, n_back)
            targets: whether a trial was a match or not
                To match tensor datatypes, 0 is False, and 1 is True.
                shape: (n_subjects, n_trials - n_back)
            responses: target labels, either 0 or 1
                shape: (n_subjects, n_trials - n_back)
            response_times: response time of each trial
                shape: (n_subjects, n_trials - n_back)
        """

        # random stimuli
        X = torch.randint(low=1, high=n_stimuli + 1, size=(n_subjects, n_trials))

        # response (either matched or not-matched)
        _min_matched, _max_matched = 0.2, 1.0
        subject_responses = torch.rand(n_subjects)
        subject_responses = subject_responses * (_max_matched - _min_matched)
        subject_responses = subject_responses + _min_matched
        subject_responses = torch.round(subject_responses * n_trials) / n_trials

        n_targets = (subject_responses * n_trials)

        responses = []
        for subj in range(n_subjects):
            n_subj_targets = n_targets[subj].int().item()
            match_trials = torch.randperm(n_trials)[:n_subj_targets]
            match_trials = torch.zeros(n_trials).scatter_(0, match_trials, 1)
            responses.append(match_trials)
        responses = torch.stack(responses)

        X = X.unfold(1, n_back + 1, 1)  # sliding window of size n_back

        targets = []
        for subj_x in X:
            subj_targets = torch.stack(
                [x[-1] == x[-1 - n_back] for x in subj_x.unbind(dim=0)]
            )
            targets.append(subj_targets)
        targets = torch.stack(targets)

        # y = targets
        # ALT: generate correct responses and fill the rest with incorrect ones
        # y = torch.where(y == 1, X, (X + 1) % (n_stimuli + 1))

        # response time
        # TODO move rate (.5) to hyper-parameters
        rt_dist = torch.distributions.exponential.Exponential(.5)
        response_times = rt_dist.sample(responses.shape) + 2.0

        # convert RTs to steps; time resolution is 100ms
        # TODO move time resolution (100ms) to hyper-parameters
        torch.round(response_times * 1000 / 10)

        return X, targets, responses[:, n_back:], response_times[:, n_back:].int()


class NBackMockDataModule(pl.LightningDataModule):

    def __init__(
        self,
        n_subjects=2,
        n_trials=100,
        n_stimuli=6,
        n_back=2,
        train_ratio=.8,
        batch_size=4,
        randomized_split=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.n_stimuli = n_stimuli
        self.n_back = n_back
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.randomized_split = randomized_split
        self.n_workers = os.cpu_count()

    def prepare_data(self) -> None:

        self.dataset = NBackMockDataset(
            n_subjects=self.n_subjects,
            n_trials=self.n_trials,
            n_back=self.n_back,
            n_stimuli=self.n_stimuli)

        # only the first subject is used
        # TODO return all the subjects
        X, is_targets, responses, response_steps = self.dataset[0]
        self.dataset = TensorDataset(X, is_targets, responses, response_steps)

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
            num_workers=self.n_workers)

        return dataloader

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)
    
    def val_dataloader(self):
        return self._dataloader(self.test_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)
