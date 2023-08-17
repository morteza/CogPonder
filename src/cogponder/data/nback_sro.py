import torch
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
from typing import Union
import numpy as np
from .utils import remove_non_decision_time
from numpy.lib.stride_tricks import sliding_window_view


class NBackSRODataset(Dataset):
    """Self-regulation ontology adaptive N-back dataset -- a binary classification task.

    Args:
        n_subjects (int): Number of subjects.
        n_back (int): Number of items in the N-back sequence. Defaults to 2.
        step_duration (int): Duration of each step in milliseconds. Defaults to 10.
        non_decision_time (str or int): Non-decision time in milliseconds. Defaults to 'auto'.
    """

    # feature names mapping from the original dataset (value) to the tensor dataset (key)
    mappings = {
        'subject_ids': ['worker_id'],
        'trial_ids': ['trial_index'],
        'contexts': ['is_match'],
        'stimuli': ['stim'],
        'responses': ['key_press'],
        'response_steps': ['response_step'],
        'corrects': ['correct']
    }

    def __init__(
        self,
        n_subjects: int = -1,  # -1 means all
        n_back=2,
        step_duration=10,
        non_decision_time: Union[str, int] = 'auto',
        data_path='data/Self_Regulation_Ontology/adaptive_n_back.csv.gz'
    ):

        self.n_subjects = n_subjects
        self.n_back = n_back
        self.step_duration = step_duration
        self.non_decision_time = non_decision_time
        self.data_path = data_path

        self._data = self.to_tensor_dataset(self.preprocess(n_back=self.n_back))

    def __len__(self):
        """Get the number of samples.
        """
        return self._data.__len__()

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        items = self._data[idx]

        return items

    def preprocess(self, n_back):

        data = pd.read_csv(self.data_path, index_col=0)

        worker_ids = data['worker_id'].unique()[:self.n_subjects]  # noqa

        data = data.query('worker_id in @worker_ids and '
                          'exp_stage == "adaptive" and '
                          'load == @n_back and target.notna()').copy()

        data.sort_index(inplace=True)

        sro_keys = {37: 'match', 40: 'non-match'}

        # mapping
        data['block_index'] = data.groupby(['worker_id']).cumcount()
        data['trial_index'] = data.groupby(['worker_id']).cumcount()
        data['is_match'] = data['stim'].str.upper() == data['target'].str.upper()  # match or not
        data['key_press'] = data['key_press'].map(sro_keys)
        data['response_step'] = data['rt'] // self.step_duration
        data['response_step'] = data['response_step'].apply(np.floor).astype('int')

        # set categories
        data['worker_id'] = data['worker_id'].astype('category')
        data['key_press'] = data['key_press'].astype('category').cat.set_categories(
            list(sro_keys.values()), ordered=True)
        data['stim'] = data['stim'].str.upper().astype('category')

        # encode categorical variables
        data['worker_id'] = data['worker_id'].cat.codes.astype('int')   # start at 0
        data['is_match'] = data['is_match'].astype('int')   # start at 0
        data['key_press'] = data['key_press'].cat.codes.astype('int')
        data['stim'] = data['stim'].cat.codes.astype('float32') + 1  # start at 1, 0 is reserved for burn-in padding
        data['correct'] = data['correct'].astype('int')

        preprocessed = {k: data[v].values.squeeze() for k, v in self.mappings.items()}

        # sliding window stimuli
        stim = data.groupby(['worker_id', 'block_num'])['stim'].apply(
            lambda x: sliding_window_view(
                np.pad(x, (self.n_back, 0), 'constant', constant_values=0),
                n_back + 1).tolist(),
        ).to_list()

        # reshape stimuli to (trials, seq, feature)
        stim = np.concatenate(stim).reshape(-1, self.n_back + 1, 1)
        preprocessed['stimuli'] = stim  # type: ignore

        return preprocessed

    def to_tensor_dataset(self, preprocessed):
        """Helper to convert a preprocessed data mapping to a TensorDataset.
        """

        # create a sequence of previous N stimuli
        tensors = [torch.Tensor(v) for k, v in preprocessed.items()]

        return TensorDataset(*tensors)
