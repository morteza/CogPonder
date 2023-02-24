import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union


class NBackSRODataset(Dataset):
    """Self-regulation ontology adaptive N-back dataset -- a binary classification task.

    """

    def __init__(
        self,
        n_subjects,
        n_back=2,
        response_step_interval=10,
        non_decision_time: Union[str, int] = 'auto',
        data_file='data/Self_Regulation_Ontology/adaptive_n_back.csv.gz'
    ):

        """Initialize the dataset.

        Args:
            n_subjects (int): Number of subjects.
            n_back (int): Number of items in the N-back sequence. Defaults to 2.
            non_decision_time (str or int): Non-decision time in milliseconds. Defaults to 'auto'.
        """
        self.n_subjects = n_subjects
        self.n_back = n_back
        self.response_step_interval = response_step_interval
        self.non_decision_time = non_decision_time
        self.data_file = data_file

        # load and cleanup the data
        self.X, self.trial_types, self.is_targets, self.responses, self.response_times = \
            self.prepare_data(self.n_subjects, self.n_back)

    def __len__(self):
        """Get the number of samples.
        """
        return self.n_subjects

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        return (
            self.X[idx, :],
            self.trial_types[idx],
            self.is_targets[idx],
            self.responses[idx],
            self.response_times[idx])

    def prepare_data(self, n_subjects, n_back, **kwargs):
        """[summary]
        # TODO required data columns:
        #   subject_index, trial_index, stimulus_index, accuracy, response_time

        Args:
            n_subjects (int): number of subjects to load. -1 for all subjects.
            n_back (int): number of items in the N-back sequence
            worker_ids (list[str], optional): list of SRO worker ids to load. Defaults to None.

        Returns:
            X: stimulus features
                shape: (n_subjects * n_trials, n_back)
            trial_types: whether a trial was a match or not
                To match tensor datatypes, 0 is False, and 1 is True.
                shape: (n_subjects * n_trials)
            is_corrects: whether a response was correct or not
                shape: (n_subjects * n_trials)
            responses: whether it was responded as a match or not
                shape: (n_subjects * n_trials)
            response_times: response time
                shape: (n_subjects * n_trials)
        """

        data = pd.read_csv(
            self.data_file,
            index_col=0,
            dtype={
                'worker_id': 'str',
                'stim_color': 'category',
                'stim_word': 'category',
                'condition': 'category'})

        # figure out worker_ids to load
        worker_ids = kwargs.get('worker_ids', [])

        if len(worker_ids) == 0:
            worker_ids = data['worker_id'].unique()[:n_subjects]

        # filter out worker_ids and practice trials
        data = data.query('worker_id in @worker_ids and '
                          'exp_stage == "adaptive" and '
                          # 'block_num == 0.0 and '
                          'load == @n_back').copy()

        data = data.sort_values(['worker_id', 'block_num', 'trial_num'])

        data['worker_id'] = data['worker_id'].astype('category')

        stimuli = data['stim'].str.upper().astype('category').cat.codes.values

        X = torch.tensor(stimuli)
        X = X.unfold(0, n_back + 1, 1)  # sliding window of size n_back
        X = torch.cat([torch.zeros(n_back, X.size(1)), X])  # pad  with zeros (burn-in trials)

        # add subject_index as the first column of X
        X_ids = torch.tensor(data['worker_id'].cat.codes.values).reshape(-1, 1)  # (n_trials, 1)

        X = torch.cat([X_ids, X], dim=1)

        is_targets = data['stim'].str.lower() == data['target'].str.lower()

        trial_types = torch.tensor(is_targets.values).reshape(1, -1).int()

        # response (either matched or not-matched)
        responses = torch.tensor(data.key_press.astype('category').cat.codes.values)
        responses = responses.reshape(1, -1)

        # is_corrects
        is_corrects = torch.tensor(data['correct'].values).reshape(1, -1).bool()

        # response time
        # convert RTs to steps
        response_times = torch.tensor(data['rt'].values)
        response_times = response_times.reshape(1, -1)  # (n_subjects, n_trials)

        # automatically calculate non-decision time (min RT - 1-step)
        if self.non_decision_time == 'auto':
            valid_rts = torch.where(response_times <= 0, torch.inf, response_times)
            min_rt = torch.min(valid_rts, dim=1, keepdim=True)[0]
            self.non_decision_time = min_rt - self.response_step_interval

        # subtract non-decision time from RT
        response_times = torch.where(response_times <= 0,
                                     0, response_times - self.non_decision_time)

        # convert RTs to steps
        response_steps = torch.round(response_times / self.response_step_interval).int()

        # discard trials with invalid targets, e.g., burn-in trials
        valid_trials_mask = data['target'].notna().values
        X = X[:, valid_trials_mask, :]
        trial_types = trial_types[:, valid_trials_mask]
        is_corrects = is_corrects[:, valid_trials_mask]
        responses = responses[:, valid_trials_mask]
        response_steps = response_steps[:, valid_trials_mask]

        return (X, trial_types, is_corrects, responses, response_steps)
