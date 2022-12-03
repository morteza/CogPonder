from urllib import response
import torch
import pandas as pd
from torch.utils.data import Dataset


class StroopSRODataset(Dataset):
    """Self-regulation ontology Stroop dataset -- a categorization task.

    """

    def __init__(
        self,
        n_subjects=-1,
        response_step_interval=10,
        non_decision_time=200,  # in millis
        data_file='data/Self_Regulation_Ontology/stroop.csv.gz'
    ):

        """Initialize and load the SRO Stroop dataset.

        Args:
            n_subjects (int): Number of subjects. Defaults to -1 (all).
            response_step_interval (int):
                Size of the bins for the conversion of the response time to steps; in millis.
            non_decision_time (int):
                in millis, is considered to be the time between making decision and
                observing corresponding effects (e.g., after some motor movements).
        """

        self.n_subjects = n_subjects
        self.response_step_interval = response_step_interval
        self.data_file = data_file

        # load and cleanup the data
        data = self.prepare_data(self.n_subjects, self.response_step_interval, self.data_file)
        self.X, self.trial_types, self.is_corrects, self.responses, self.response_steps = data

    def __len__(self):
        """Get the number of samples.
        """
        return self.n_subjects

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        return (self.X[idx],
                self.trial_types[idx],
                self.is_corrects[idx],
                self.responses[idx],
                self.response_steps[idx, :])

    @classmethod
    def prepare_data(cls,
                     n_subjects,
                     response_step_interval,
                     data_file,
                     colors_order=['red', 'green', 'blue'],
                     color_codes={66: 'blue', 71: 'green', 82: 'red'}
                     ):
        """[summary]
        Args:
            response_step_interval (int):
                Size of the bins for the response time in millis.
            data_file (File):
                SRO compressed file containing the data; original file name: stroop.csv.gz.

        Returns:
            X: stimulus features
                Stroop stimulus features. dim0 is the ink color (i.e., target) and dim1 is the word.
                shape: (n_subjects, n_trials, 2).
            trial_types: whether it was a congruent or incongruent trial.
                To match tensor datatypes, 0=incongruent, 1=congruent.
                shape: (n_subjects, n_trials)
            is_corrects: target labels, either 0 (incorrect) or  1 (correct).
                shape: (n_subjects, n_trials)
            responses: the pressed key, representing the color of the word.
                shape: (n_subjects, n_trials)
            response_times: response time of each trial.
                shape: (n_subjects, n_trials)
        """

        # TODO support multi subject by querying worker_id
        data = pd.read_csv(
            data_file,
            index_col=0,
            dtype={
                'worker_id': 'str',
                'stim_color': 'category',
                'stim_word': 'category',
                'condition': 'category'})

        selected_worker_ids = data['worker_id'].unique()[:n_subjects]  # noqa: F841
        n_selected_subjects = len(selected_worker_ids)

        # filter out practice and no-response trials
        data = data.query('worker_id in @selected_worker_ids and '
                          'exp_stage == "test"').copy()

        data['key_press'] = data['key_press'].astype('int').map(color_codes).astype('category')

        data['key_press'] = data['key_press'].cat.reorder_categories(colors_order)
        data['stim_color'] = data['stim_color'].cat.reorder_categories(colors_order)
        data['stim_word'] = data['stim_word'].cat.reorder_categories(colors_order)
        data['condition'] = data['condition'].cat.reorder_categories(['incongruent', 'congruent'])

        data = data.sort_index(ascending=True)
        stim_color = data['stim_color'].cat.codes
        stim_color = torch.tensor(stim_color.values).reshape(-1, 1)

        stim_word = data['stim_word'].cat.codes
        stim_word = torch.tensor(stim_word.values).reshape(-1, 1)

        worker_ids = data['worker_id'].apply(lambda s: int(s.replace('s', '')))
        worker_ids = torch.tensor(worker_ids.values).reshape(-1, 1)

        X = torch.cat((worker_ids, stim_color, stim_word), dim=1).float()

        X = X.reshape(n_selected_subjects, -1, 3)  # (n_subjects, n_trials, 2)

        # TODO make sure "incongruent" casts to 0 and "congruent" casts to 1
        trial_types = torch.tensor(data['condition'].cat.codes.values)
        trial_types = trial_types.reshape(n_selected_subjects, -1).float()  # (n_subjects, n_trials)

        is_corrects = torch.tensor(data['correct'].values)
        is_corrects = is_corrects.reshape(n_selected_subjects, -1).float()  # (n_subjects, n_trials)

        responses = torch.tensor(data['key_press'].cat.codes.values)
        responses = responses.reshape(n_selected_subjects, -1)  # (n_subjects, n_trials)

        # convert RTs to steps
        response_times = torch.tensor(data['rt'].values)
        response_times = response_times.reshape(n_selected_subjects, -1)  # (n_subjects, n_trials)
        response_steps = torch.round(response_times / response_step_interval).int()

        return (
            X,
            trial_types,
            is_corrects,
            responses,
            response_steps
        )
