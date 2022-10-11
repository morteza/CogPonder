import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class NBackSRODataset(Dataset):
    """Self-regulation ontology adaptive N-back dataset -- a binary classification task.

    """

    def __init__(
        self,
        n_subjects,
        n_back=2,
        device='cpu',
        data_file='data/Self_Regulation_Ontology/adaptive_n_back.csv.gz'
    ):

        """Initialize the dataset.

        Args:
            n_subjects (int): Number of subjects.
            n_back (int): Number of items in the N-back sequence. Default: 2.
            device (torch.device): Device to use for the dataset.
        """
        self.n_subjects = n_subjects
        self.device = device
        self.n_back = n_back
        self.data_file = data_file

        # load and cleanup the data
        self.X, self.trial_types, self.is_targets, self.responses, self.response_times = \
            self.prepare_data(self.data_file)

    def __len__(self):
        """Get the number of samples.
        """
        return self.n_subjects

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        return (self.X[idx, :],
                self.trial_types[idx],
                self.is_targets[idx],
                self.responses[idx],
                self.response_times[idx, :])

    @classmethod
    def prepare_data(cls, data_file, n_back=2):
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

        data = pd.read_csv(data_file, index_col=0)
        data = data.query('worker_id == worker_id.unique()[-1] and exp_stage == "adaptive"')

        data = data.query('load == @n_back').sort_values(['block_num', 'trial_num'])
        stimuli = data.stim.str.upper().astype('category').cat.codes.values

        X = torch.tensor(stimuli).reshape(1, -1)

        # response (either matched or not-matched)
        responses = torch.tensor(data.key_press.astype('category').cat.codes.values)
        responses = responses.reshape(1, -1).float()

        X = X.unfold(1, n_back + 1, 1)  # sliding window of size n_back

        is_targets = []
        for subj_x in X:
            subj_targets = torch.stack(
                [x[-1] == x[-1 - n_back] for x in subj_x.unbind(dim=0)]
            )
            is_targets.append(subj_targets)
        is_targets = torch.stack(is_targets)

        # response time
        # convert RTs to steps; time resolution is 50ms
        # TODO move time resolution (100ms) to hyper-parameters
        response_times = torch.tensor(data.rt.values).reshape(1, -1)
        response_steps = torch.round(response_times / 100).int()

        # TODO trial types
        trial_types = torch.bitwise_or(is_targets * 2, responses[:, n_back:].long()).int() + 1

        return (
            X.float(),
            trial_types,
            is_targets,
            responses[:, n_back:],
            response_steps[:, n_back:]
        )
