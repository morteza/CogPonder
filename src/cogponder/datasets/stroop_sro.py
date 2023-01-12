from urllib import response
import torch
import pandas as pd
from torch.utils.data import Dataset


class StroopSRODataset(Dataset):
    """Self-regulation ontology Stroop dataset -- a categorization task.

    """

    def __init__(
        self,
        subject_ids: list[int] = [],  # [] for all
        response_step_interval=10,
        non_decision_time='auto',  # int (in millis) or 'auto'
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
        """

        self.subject_ids = subject_ids
        self.response_step_interval = response_step_interval
        self.non_decision_time = non_decision_time
        self.data_file = data_file

        # load and cleanup the data
        data = self.prepare_data()
        self.contexts, self.stimuli, self.trial_types, self.is_corrects, self.responses, self.response_steps = data

    def __len__(self):
        """Get the number of samples.
        """
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """Get a feature vector and it's target label.
        """

        if isinstance(idx, slice):
            idx = torch.unique(self.contexts)[idx]

        masker = torch.isin(self.contexts, idx)

        # squeeze subjects into one array
        # contexts = contexts.reshape(-1,)
        # stimuli = stimuli.reshape(-1, stimuli.shape[-1])
        # trial_types = trial_types.reshape(-1,)
        # is_corrects = is_corrects.reshape(-1,)
        # responses = responses.reshape(-1,)
        # response_steps = response_steps.reshape(-1,)

        return (
            self.contexts[masker],
            self.stimuli[masker, :],
            self.trial_types[masker],
            self.is_corrects[masker],
            self.responses[masker],
            self.response_steps[masker])

    def prepare_data(self,
                     colors_order=['red', 'green', 'blue'],
                     color_codes={66: 'blue', 71: 'green', 82: 'red'}):
        """[summary]
        Args:
            response_step_interval (int):
                Size of the bins for the response time in millis.
            data_file (File):
                SRO compressed file containing the data; original file name: stroop.csv.gz.

        Returns:
            contexts: contextual information for each trial.
                This can include for example subject, task, and environment.
                shape: (n_subjects * n_trials)
            stimuli: stimulus features
                Stroop stimulus features. dim0 is the ink color (i.e., target) and dim1 is the word.
                shape: (n_subjects * n_trials, 2).
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
            self.data_file,
            index_col=0,
            dtype={
                'worker_id': 'str',
                'stim_color': 'category',
                'stim_word': 'category',
                'condition': 'category'})

        # convert work_id from 'sX' string to numerical X
        data['worker_id'] = data['worker_id'].apply(lambda w: w.replace('s', '')).astype('int')

        # load all subjects if none specified
        if len(self.subject_ids) == 0:
            self.subject_ids = data['worker_id'].unique()

        # filter out practice and no-response trials
        data = data.query('worker_id in @self.subject_ids and '
                          'exp_stage == "test"').copy()

        data['worker_id'] = data['worker_id'].astype('category')

        data['key_press'] = data['key_press'].astype('int').map(color_codes).astype('category')

        data['key_press'] = data['key_press'].cat.reorder_categories(colors_order)
        data['stim_color'] = data['stim_color'].cat.reorder_categories(colors_order)
        data['stim_word'] = data['stim_word'].cat.reorder_categories(colors_order)

        # making sure "incongruent" casts to 0 and "congruent" casts to 1
        data['condition'] = data['condition'].cat.reorder_categories(['incongruent', 'congruent'])

        data = data.sort_index(ascending=True)
        stim_color = data['stim_color'].cat.codes
        stim_color = torch.tensor(stim_color.values).reshape(-1, 1)

        stim_word = data['stim_word'].cat.codes
        stim_word = torch.tensor(stim_word.values).reshape(-1, 1)

        contexts = torch.tensor(data['worker_id'].cat.codes.values)
        contexts = contexts.reshape(len(self.subject_ids), -1).long()  # (n_subjects, n_trials)

        stimuli = torch.cat((stim_color, stim_word), dim=1).float()

        stimuli = stimuli.reshape(len(self.subject_ids), -1, 2)  # (n_subjects, n_trials, 2)

        trial_types = torch.tensor(data['condition'].cat.codes.values)
        trial_types = trial_types.reshape(len(self.subject_ids), -1).float()  # (n_subjects, n_trials)

        is_corrects = torch.tensor(data['correct'].values)
        is_corrects = is_corrects.reshape(len(self.subject_ids), -1).float()  # (n_subjects, n_trials)

        responses = torch.tensor(data['key_press'].cat.codes.values)
        responses = responses.reshape(len(self.subject_ids), -1)  # (n_subjects, n_trials)

        # convert RTs to steps
        response_times = torch.tensor(data['rt'].values)
        response_times = response_times.reshape(len(self.subject_ids), -1)  # (n_subjects, n_trials)

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
            contexts,
            stimuli,
            trial_types,
            is_corrects,
            responses,
            response_steps
        )
