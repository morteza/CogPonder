from collections import namedtuple
import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset, TensorDataset
import numpy as np
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

    # feature names mapping from the original dataset (value) to the tensor dataset (key)
    mappings = {
        'subject_ids': ['worker_id'],
        'trial_ids': ['trial_index'],
        'contexts': ['condition'],
        'inputs': ['stim_color', 'stim_word'],
        'outputs': ['key_press'],
        'response_steps': ['response_step'],
        'correct_outputs': ['correct_response'],
    }

    sro_conditions = {'incongruent': 0, 'congruent': 1}
    sro_colors = {66: 'blue', 71: 'green', 82: 'red', -1: 'timeout'}

    def __init__(
        self,
        n_subjects: int = -1,  # -1 means all TODO: use mixture of list[str]/int for reproducibility
        step_duration=10,
        non_decision_time: Union[str, int] = 'auto',
        shuffle_subjects: bool = True,
        data_path='data/Self_Regulation_Ontology/stroop.csv.gz'
    ):

        self.n_subjects = n_subjects
        self.step_duration = step_duration
        self.non_decision_time = non_decision_time
        self.shuffle_subjects = shuffle_subjects
        self.data_path = data_path

        self._data = self.to_tensor_dataset(self.preprocess())

    def __len__(self):
        """Get the total number of observations (i.e., trials).
        """
        return self._data.__len__()

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        items = self._data[idx]

        return items

    def inverse_transform_context(self, contexts):
        """Inverse transform context values to original labels.
        """

        inv_map = {v: k for k, v in self.sro_conditions.items()}
        return [inv_map[int(c)] for c in contexts]

    def preprocess(self):

        data = pd.read_csv(self.data_path, index_col=0)

        # shuffle and select a subset of subjects
        worker_ids = data['worker_id'].unique()
        if self.shuffle_subjects:
            np.random.shuffle(worker_ids)
        worker_ids = worker_ids[:self.n_subjects]

        # select only selected worker_ids and test trials
        data = data.query('worker_id in @worker_ids and exp_stage == "test"').copy()

        data.sort_index(inplace=True)
        data['trial_index'] = data.groupby('worker_id').cumcount()

        # map key_press to color names
        data['key_press'] = data['key_press'].map(self.sro_colors)
        data['correct_response'] = data['correct_response'].map(self.sro_colors)

        # set categories
        data['worker_id'] = data['worker_id'].astype('category')
        data['condition'] = data['condition'].astype('category').cat.set_categories(
            list(self.sro_conditions.keys()), ordered=True)
        data['key_press'] = data['key_press'].astype('category').cat.set_categories(
            list(self.sro_colors.values()), ordered=True)
        data['correct_response'] = data['correct_response'].astype('category').cat.set_categories(
            list(self.sro_colors.values()), ordered=True)
        data['stim_color'] = data['stim_color'].astype('category').cat.set_categories(
            list(self.sro_colors.values()), ordered=True)
        data['stim_word'] = data['stim_word'].astype('category').cat.set_categories(
            list(self.sro_colors.values()), ordered=True)

        # compute response steps
        data['response_step'] = data['rt'] // self.step_duration
        data['response_step'] = data['response_step'].apply(np.floor).astype('int')

        return data

    def to_tensor_dataset(self, data):
        """Helper to convert a preprocessed data mapping to a TensorDataset.
        """

        # encode categorical variables
        data['worker_id'] = data['worker_id'].cat.codes.astype('int')   # start at 0
        data['condition'] = data['condition'].cat.codes.astype('int')   # start at 0

        data[['correct_response', 'key_press']] = data[['correct_response', 'key_press']].apply(
            lambda x: x.cat.codes.astype('int'))
        data['stim_color'] = data['stim_color'].cat.codes.astype('float32')
        data['stim_word'] = data['stim_word'].cat.codes.astype('float32')

        preprocessed = {k: data[v].values.squeeze() for k, v in self.mappings.items()}

        # reshape stimuli to (trials, seq, feature)
        preprocessed['inputs'] = preprocessed['inputs'].reshape(-1, 1, 2)

        tensors = [torch.Tensor(v) for k, v in preprocessed.items()]

        return TensorDataset(*tensors)

        # TODO REWRITE and REMOVE
        # automatically calculate non-decision time (min RT - 1-step)
        # if self.non_decision_time == 'auto':
        #     data['rt'] = data.groupby(['worker_id'])['rt'].transform(remove_non_decision_time,
        #                                                              response_step_interval=self.response_step_interval)
        # data['response_step'] = (data['rt'] / self.response_step_interval).apply(np.round)
