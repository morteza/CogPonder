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
            responses: target labels, either 0 or 1
                shape: (n_subjects, n_trials - n_back)
            targets: whether a trial was a match or not
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
        response_times = rt_dist.sample(responses.shape) + 1.0

        # convert RTs to steps; time resolution is 100ms
        # TODO move time resolution (100ms) to hyper-parameters
        torch.round(response_times * 1000 / 10)

        return X, targets, responses[:, n_back:], response_times[:, n_back:].int()


# DEBUG
# dataset = NBackDataset(2, 10, 5, n_back=2)
# print(dataset[0])
