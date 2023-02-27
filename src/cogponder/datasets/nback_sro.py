import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Union
import numpy as np


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
        self._data = self.prepare_data(self.n_subjects, self.n_back)

    def __len__(self):
        """Get the number of samples.
        """
        return self.n_subjects

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        _item = []

        for x in self._data:
            _item.append(x[idx])

        return tuple(_item)

    def _remove_non_decision_time(self, rts: pd.Series, response_step_interval=20):
        rts[rts <= 0] = pd.NA
        ndt = rts.min() - response_step_interval
        rts = rts.apply(lambda rt: rt - ndt if rt > 0 else rt)
        return rts

    def prepare_data(self, n_subjects, n_back, **kwargs):
        """[summary]
        # TODO required data columns:
        #   subject_index, trial_index, stimulus_index, accuracy, response_time

        Args:
            n_subjects (int): number of subjects to load. -1 for all subjects.
            n_back (int): number of items in the N-back sequence
            worker_ids (list[str], optional): list of SRO worker ids to load. Defaults to None.

        Returns:
            subject_ids:
                subject index.
            stimulus:
                stimulus features. In N-back, this is the letter presented.
            trial_types:
                Whether current stimulus was a match or not: 0 is non-match, and 1 is a match.
            responses:
                whether current stimulus was detected as a match or not.
            response_steps:
                response time (refined to steps).
            corrects:
                whether recorded response was correct or not.
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
        data['matched'] = data['stim'].str.upper() == data['target'].str.upper()  # match or not

        # automatically calculate non-decision time (min RT - 1-step)
        if self.non_decision_time == 'auto':
            data['rt'] = data.groupby(['worker_id'])['rt'].transform(self._remove_non_decision_time,
                                                                     response_step_interval=self.response_step_interval)
        data['response_step'] = (data['rt'] / self.response_step_interval).apply(np.round)

        # discard trials with invalid targets, e.g., burn-in trials
        valid_trials_mask = data['target'].notna().index

        # to numpy
        subject_ids = data['worker_id'].astype('category').cat.codes
        stimulus = data['stim'].str.upper().astype('category').cat.codes
        trial_types = data['matched'].loc[valid_trials_mask].values
        responses = data['key_press'].astype('category').cat.codes.loc[valid_trials_mask].values
        response_steps = data['response_step'].loc[valid_trials_mask].values
        corrects = data['correct'].loc[valid_trials_mask].values

        # to tensors
        subject_ids = torch.tensor(subject_ids, dtype=torch.long)
        stimulus = torch.tensor(stimulus, dtype=torch.long)
        trial_types = torch.tensor(trial_types, dtype=torch.long)
        responses = torch.tensor(responses, dtype=torch.long)
        response_steps = torch.tensor(response_steps, dtype=torch.long)
        corrects = torch.tensor(corrects, dtype=torch.bool)

        return (subject_ids, stimulus, trial_types, responses, response_steps, corrects)
