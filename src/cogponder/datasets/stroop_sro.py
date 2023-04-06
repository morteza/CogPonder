import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
from .utils import remove_non_decision_time


class StroopSRODataset(Dataset):
    """Self-regulation ontology Stroop dataset -- a categorization task.

    Args:
        worker_ids (list): Subject identifiers to fetch. Defaults to [] (all).
        response_step_interval (int):
            Size of the bins for the conversion of the response time to steps; in millis.
        non_decision_time (int or 'auto'):
            If int (in seconds) it it will be subtracted from the response time.
            If 'auto', it will be estimated as the minimum of response times per subject,
            mapping values to 1..(max-min+1).
            If None, no subtraction will be performed.
        data_path (File, optional):
            Overrides path to the SRO compressed file containing the data; original file name: stroop.csv.gz.

    """

    def __init__(
        self,
        n_subjects: int = -1,  # -1 means all
        response_step_interval=10,
        non_decision_time: Union[str, int] = 'auto',
        data_path='data/Self_Regulation_Ontology/stroop.csv.gz'
    ):

        self.n_subjects = n_subjects
        self.response_step_interval = response_step_interval
        self.non_decision_time = non_decision_time
        self.data_path = data_path

        self._data = self.preprocess()

    def __len__(self):
        """Get the total number of observations (i.e., trials).
        """
        return self._data.dims['observation']

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        subset = self._data.isel(observation=idx)
        items = [torch.tensor(subset[v].values) for v in subset.data_vars]

        return tuple(items)

    def preprocess(self):

        data = pd.read_csv(self.data_path, index_col=0)

        worker_ids = data['worker_id'].unique()[:self.n_subjects]  # noqa

        # select only worker_ids and test trials
        data = data.query('worker_id in @worker_ids and exp_stage == "test"').copy()

        data.sort_index(inplace=True)
        data['trial_index'] = data.groupby('worker_id').cumcount() + 1

        sro_conditions = {'incongruent': 0, 'congruent': 1}
        sro_colors = {-1: 'timeout', 66: 'blue', 71: 'green', 82: 'red'}

        # map key_press to color names
        data['key_press'] = data['key_press'].map(sro_colors)

        # set categories
        data['worker_id'] = data['worker_id'].astype('category')
        data['condition'] = data['condition'].astype('category').cat.set_categories(sro_conditions.keys(), ordered=True)
        data['key_press'] = data['key_press'].astype('category').cat.set_categories(sro_colors.values(), ordered=True)
        data['stim_color'] = data['stim_color'].astype('category').cat.set_categories(sro_colors.values(), ordered=True)
        data['stim_word'] = data['stim_word'].astype('category').cat.set_categories(sro_colors.values(), ordered=True)

        # encode categorical variables
        data['worker_id'] = data['worker_id'].cat.codes.astype('int') + 1  # start at 1
        data['condition'] = data['condition'].cat.codes.astype('int')
        data['key_press'] = data['key_press'].cat.codes.astype('int')
        data['stim_color'] = data['stim_color'].cat.codes.astype('float32')
        data['stim_word'] = data['stim_word'].cat.codes.astype('float32')
        data['correct'] = data['correct'].astype('int')

        # compute response steps
        data['response_step'] = data['rt'] // self.response_step_interval
        data['response_step'] = data['response_step'].apply(np.floor).astype('int')

        mappings = {
            'trial_ids': ['trial_index'],
            'subject_ids': ['worker_id'],
            'contexts': ['condition'],
            'stimuli': ['stim_color', 'stim_word'],
            'responses': ['key_press'],
            'response_steps': ['response_step'],  # requires post-processing to be in steps
            'corrects': ['correct'],
        }

        dimensions = {
            'trial_ids': ('observation'),
            'subject_ids': ('observation'),
            'contexts': ('observation'),
            'stimuli': ('observation', 'stimulus_modality'),
            'responses': ('observation'),
            'response_steps': ('observation'),
            'corrects': ('observation')
        }

        preprocessed_data = xr.Dataset()

        for k, v in mappings.items():
            preprocessed_data[k] = (dimensions[k], data[v].values.squeeze())

        return preprocessed_data

        # TODO REWRITE and REMOVE
        # automatically calculate non-decision time (min RT - 1-step)
        # if self.non_decision_time == 'auto':
        #     data['rt'] = data.groupby(['worker_id'])['rt'].transform(remove_non_decision_time,
        #                                                              response_step_interval=self.response_step_interval)
        # data['response_step'] = (data['rt'] / self.response_step_interval).apply(np.round)

        # discard trials with invalid targets, e.g., burn-in trials
        # data = data.query('key_press.notna() and rt>=0').copy()
