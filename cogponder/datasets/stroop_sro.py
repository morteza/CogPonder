import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class StroopSRODataset(Dataset):
    """Self-regulation ontology Stroop dataset -- a categorization task.

    """

    def __init__(
        self,
        n_subjects,
        rt_bin_size=20,
        data_file='data/Self_Regulation_Ontology/stroop.csv.gz'
    ):

        """Initialize the dataset.

        Args:
            n_subjects (int): Number of subjects.
            rt_bin_size (int): Size of the bins for the response time in millis.
        """
        self.n_subjects = n_subjects
        self.rt_bin_size = rt_bin_size
        self.data_file = data_file

        # load and cleanup the data
        self.X, self.conditions, self.is_corrects, self.response_times = \
            self.prepare_data(self.rt_bin_size, self.data_file)

    def __len__(self):
        """Get the number of samples.
        """
        return self.n_subjects

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        return (self.X[idx],
                self.conditions[idx],
                self.is_corrects[idx],
                self.response_times[idx, :])

    @classmethod
    def prepare_data(cls, rt_bin_size, data_file):
        """[summary]
        # TODO required data columns:
                # condition,
                # correct,
                # rt,
                # stim_color,
                # stim_word

        Args:
            data_file (File): [description]

        Returns:
            X: stimulus features
                Stroop stimulus features (dim0: color, dim1: word).
                shape: (n_subjects, n_trials, 2).
            conditions: whether it was a congruent or incongruent trial.
                To match tensor datatypes, 0 is congruent, and 1 is incongruent.
                shape: (n_subjects, n_trials)
            is_corrects: target labels, either 0 or 1
                shape: (n_subjects, n_trials)
            response_times: response time of each trial
                shape: (n_subjects, n_trials)
        """

        # TODO support multi subject by querying worker_id

        data = pd.read_csv(data_file, index_col=0)
        data = data.query('worker_id == worker_id.unique()[-1] and exp_stage == "test"')

        data = data.sort_index(ascending=True)
        stim_color = data['stim_color'].astype('category').cat.codes.values
        stim_color = torch.tensor(stim_color).reshape(-1, 1)

        stim_word = data['stim_word'].astype('category').cat.codes.values
        stim_word = torch.tensor(stim_word).reshape(-1, 1)

        X = torch.cat((stim_color, stim_word), dim=1).float()

        # FIXME: WORKAROUND for single-subject data
        X = X.reshape(1, -1, 2)

        # TODO make sure "congruent" casts to 0 and "incongruent" casts to 1
        conditions = torch.tensor(data['condition'].astype('category').cat.codes.values)
        conditions = conditions.reshape(1, -1).float()
        is_corrects = torch.tensor(data['correct'].values).reshape(1, -1).float()

        # response time
        # convert RTs to steps; time resolution is 50ms
        # TODO move time resolution (100ms) to hyper-parameters
        response_times = torch.tensor(data['rt'].values).reshape(1, -1)
        response_steps = torch.round(response_times / rt_bin_size).int()

        return (
            X,
            conditions,
            is_corrects,
            response_steps
        )
