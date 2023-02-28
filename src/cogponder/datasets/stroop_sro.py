import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
import numpy as np
from .utils import remove_non_decision_time


class StroopSRODataset(Dataset):
    """Self-regulation ontology Stroop dataset -- a categorization task.

    """

    def __init__(
        self,
        subject_ids: list[int] = [],  # [] for all
        response_step_interval=10,
        non_decision_time: Union[str, int] = 'auto',
        data_file='data/Self_Regulation_Ontology/stroop.csv.gz'
    ):

        """Initialize and load the SRO Stroop dataset.

        Args:
            subject_ids (list): Subject identifiers to fetch. Defaults to [] (all).
            response_step_interval (int):
                Size of the bins for the conversion of the response time to steps; in millis.
            non_decision_time (int or 'auto'):
                If int, in millis, it it will be subtracted from the response time.
                If 'auto', it will be estimated as the minimum of response times per subject,
                mapping values to 1..(max-min+1).
                If None, no subtraction will be performed.
            data_file (File, optional):
                Overrides path to the SRO compressed file containing the data; original file name: stroop.csv.gz.
        """

        self.subject_ids = subject_ids
        self.response_step_interval = response_step_interval
        self.non_decision_time = non_decision_time
        self.data_file = data_file

        # load and cleanup the data
        self._data = self.prepare_data(self.n_subjects)

    def __len__(self):
        """Get the number of samples.
        """
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        _item = []

        for x in self._data:
            _item.append(x[idx])

        return tuple(_item)

    def prepare_data(self,
                     n_subjects,
                     colors_order=['red', 'green', 'blue'],
                     color_codes={66: 'blue', 71: 'green', 82: 'red'},
                     **kwargs):
        """[summary]

        Args:
            color_order (list):
                order of colors in the dataset.
            color_codes (dict):
                mapping of color codes to color names.

        Returns:
            subject_ids:
                subject index.
            stimulus:
                Stroop stimulus features. dim0 is the ink color (i.e., target) and dim1 is the word.
                shape: (-1, 2).
            trial_types:
                whether it was a congruent or incongruent trial. To match tensor datatypes, 0=incongruent, 1=congruent.
                shape: (n_trials)
            responses:
                the pressed key, representing the color of the word.
            response_steps:
                response time (refined to steps).
            corrects:
                whether recorded response was correct or not. Either 0 (incorrect) or  1 (correct).
        """

        # TODO support multi subject by querying worker_id
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
                          'exp_stage == "test"').copy()

        # make sure everything is in the right order
        data['key_press'] = data['key_press'].astype('int').map(color_codes).astype('category')
        data['key_press'] = data['key_press'].cat.reorder_categories(colors_order)
        data['stim_color'] = data['stim_color'].cat.reorder_categories(colors_order)
        data['stim_word'] = data['stim_word'].cat.reorder_categories(colors_order)
        # encode as integers
        data['stim_color'] = data['stim_color'].cat.codes.values
        data['stim_word'] = data['stim_word'].cat.codes.values

        # making sure "incongruent" casts to 0 and "congruent" casts to 1
        data['condition'] = data['condition'].cat.reorder_categories(['incongruent', 'congruent'])

        # automatically calculate non-decision time (min RT - 1-step)
        if self.non_decision_time == 'auto':
            data['rt'] = data.groupby(['worker_id'])['rt'].transform(remove_non_decision_time,
                                                                     response_step_interval=self.response_step_interval)
        data['response_step'] = (data['rt'] / self.response_step_interval).apply(np.round)

        # discard trials with invalid targets, e.g., burn-in trials
        data = data.query('key_press.notna() and rt>=0').copy()

        # to numpy
        worker_ids = data['worker_id'].astype('category').cat.codes.values
        stimuli = data[['stim_color', 'stim_word']].values
        trial_types = data['condition'].cat.codes.values
        responses = data['key_press'].cat.codes.values
        response_steps = data['response_step'].values
        corrects = data['correct'].values

        # to tensors
        worker_ids = torch.tensor(worker_ids, dtype=torch.long).reshape(-1,)
        stimuli = torch.tensor(stimuli, dtype=torch.float).reshape(-1, 2)
        trial_types = torch.tensor(trial_types, dtype=torch.long).reshape(-1,)
        responses = torch.tensor(responses, dtype=torch.long).reshape(-1,)
        response_steps = torch.tensor(response_steps, dtype=torch.long).reshape(-1,)
        corrects = torch.tensor(corrects, dtype=torch.bool).reshape(-1,)

        return (worker_ids, stimuli, trial_types, responses, response_steps, corrects)
