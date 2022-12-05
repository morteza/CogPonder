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
        response_step_interval=10,
        non_decision_time='auto',
        data_file='data/Self_Regulation_Ontology/adaptive_n_back.csv.gz'
    ):

        """Initialize the dataset.

        Args:
            n_subjects (int): Number of subjects.
            n_back (int): Number of items in the N-back sequence. Default: 2.
        """
        self.n_subjects = n_subjects
        self.n_back = n_back
        self.response_step_interval = response_step_interval
        self.non_decision_time = non_decision_time
        self.data_file = data_file

        # load and cleanup the data
        self.X, self.trial_types, self.is_targets, self.responses, self.response_times = \
            self.prepare_data(worker_id='s521')

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

    def prepare_data(self, worker_id):
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

        data = pd.read_csv(
            self.data_file,
            index_col=0,
            dtype={
                'worker_id': 'str',
                'stim_color': 'category',
                'stim_word': 'category',
                'condition': 'category'})

        # filter out practice and no-response trials
        data = data.query('worker_id==@worker_id and '
                          'exp_stage == "adaptive" and '
                          'load == @self.n_back').copy()
        data = data.sort_values(['block_num', 'trial_num'])

        data['worker_id'] = data['worker_id'].astype('category')

        stimuli = data.stim.str.upper().astype('category').cat.codes.values

        X = torch.tensor(stimuli).reshape(1, -1)

        # response (either matched or not-matched)
        responses = torch.tensor(data.key_press.astype('category').cat.codes.values)
        responses = responses.reshape(1, -1).float()

        X = X.unfold(1, self.n_back + 1, 1)  # sliding window of size n_back

        is_targets = []
        for subj_x in X:
            subj_targets = torch.stack(
                [x[-1] == x[-1 - self.n_back] for x in subj_x.unbind(dim=0)]
            )
            is_targets.append(subj_targets)
        is_targets = torch.stack(is_targets)

        # TODO trial types
        trial_types = torch.bitwise_or(is_targets * 2, responses[:, self.n_back:].long()).int() + 1

        # response time
        # convert RTs to steps
        response_times = torch.tensor(data['rt'].values)
        response_times = response_times.reshape(1, -1)  # (n_subjects, n_trials)

        # automatically calculate non-decision time (min RT - 1-step)
        if self.non_decision_time == 'auto':
            valid_rts = torch.where(response_times <= 0, torch.inf, response_times)
            min_rt = torch.min(valid_rts, dim=1, keepdim=True)[0]
            self.non_decision_time = min_rt - self.response_step_interval

        # subtract non decision time
        response_times = torch.where(response_times <= 0,
                                     0, response_times - self.non_decision_time)

        # convert to steps
        response_steps = torch.round(response_times / self.response_step_interval).int()

        return (
            X.float(),
            trial_types,
            is_targets,
            responses[:, self.n_back:],
            response_steps[:, self.n_back:]
        )
