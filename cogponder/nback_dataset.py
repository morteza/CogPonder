import torch
from torch.utils.data import Dataset
import numpy as np


class NBackDataset(Dataset):
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
        self.X, self.y, self.accuracies, self.response_times = mock_data

    def __len__(self):
        """Get the number of samples.
        """
        return self.n_subjects

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """
        return (self.X[idx, :],
                self.y[idx, :],
                self.accuracies[idx, :],
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
            y: target labels
                shape: (n_subjects, n_trials - n_back)
            accuracies: accuracy of each trial
                shape: (n_subjects, n_trials - n_back)
            response_times: response time of each trial
                shape: (n_subjects, n_trials - n_back)
        """

        # random stimuli
        X = torch.randint(low=1, high=n_stimuli + 1, size=(n_subjects, n_trials))

        # response accuracy
        _min_accuracy, _max_accuracy = 0.2, 1.0
        subject_accuracies = torch.rand(n_subjects)
        subject_accuracies = subject_accuracies * (_max_accuracy - _min_accuracy)
        subject_accuracies = subject_accuracies + _min_accuracy
        subject_accuracies = torch.round(subject_accuracies * n_trials) / n_trials

        n_corrects = (subject_accuracies * n_trials)

        accuracies = []

        for subj in range(n_subjects):
            n_subj_corrects = int(n_corrects[subj].item())
            correct_trials = torch.randperm(n_trials)[:n_subj_corrects]
            trial_accuracies = torch.zeros(n_trials).scatter_(0, correct_trials, 1)
            accuracies.append(trial_accuracies)

        accuracies = torch.stack(accuracies)

        # generate correct responses and fill the rest with incorrect ones
        y = torch.where(accuracies == 1, X, (X + 1) % (n_stimuli + 1))

        # response time
        response_times = np.random.exponential(.5, size=accuracies.shape)

        X = X.unfold(1, n_back + 1, 1)  # sliding window of size n_back

        if n_subjects == 1:
            X = X.squeeze()
            y = y.squeeze()
            accuracies = trial_accuracies.squeeze()
            response_times = response_times.squeeze()

        return X, y[:, n_back:], accuracies[:, n_back:], response_times[:, n_back:]


# DEBUG
# dataset = NBackDataset(2, 10, 3)
# dataset[1]
