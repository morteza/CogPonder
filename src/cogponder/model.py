# WARNING: this is work-in-progress

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from .losses import ResponseLoss, ResponseTimeLoss
import torchmetrics


class CogPonderModel(LightningModule):
    def __init__(
        self,
        config,
        example_input_array=None
    ):
        """CogPonder model written in PyTorch Lightning

        """

        super().__init__()
        self.save_hyperparameters(ignore=['example_input_array'], logger=False)

        self.inputs_dim = config['inputs_dim']
        self.embeddings_dim = config['embeddings_dim']
        self.outputs_dim = config['outputs_dim']
        self.resp_loss_beta = config['resp_loss_beta']
        self.time_loss_beta = config['time_loss_beta']
        self.loss_by_trial_type = config['loss_by_trial_type']
        self.learning_rate = config['learning_rate']
        self.max_response_step = config['max_response_step']
        self.n_subjects = config['n_subjects']
        self.task = config['task']

        self.example_input_array = example_input_array

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.resp_loss_fn = ResponseLoss()
        self.time_loss_fn = ResponseTimeLoss(self.max_response_step)

        # init nodes: halt_node, output_node, recurrent_node
        self.halt_node = nn.Sequential(
            nn.Linear(self.embeddings_dim, self.embeddings_dim),
            nn.ReLU(),
            nn.Linear(self.embeddings_dim, 1),
            nn.Sigmoid()
        )

        self.output_node = nn.Sequential(
            nn.Linear(self.embeddings_dim, self.embeddings_dim),
            nn.ReLU(),
            nn.Linear(self.embeddings_dim, self.outputs_dim)
        )

        self.recurrent_node = nn.GRUCell(self.inputs_dim, self.embeddings_dim)

        self.subject_embedding = nn.Embedding(self.n_subjects,
                                              self.embeddings_dim,
                                              device=self.device)

    def forward(self, x):

        """_summary_

        Args:
            X (torch.Tensor): input data of shape (batch_size, n_subjects, inputs_dim)
        Returns
        -------
        _type_
            _description_
        """

        batch_size = x.size(0)
        subject_idx = torch.zeros(batch_size, ).to(self.device).long()  # debug x[:, 0].long()

        h = self.subject_embedding(subject_idx)
        p_continue = torch.ones(batch_size, device=self.device)
        halt_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        y_list = []
        p_list = []

        for n in range(1, self.max_response_step + 1):

            if n == self.max_response_step:
                lambda_n = torch.ones((batch_size,), device=self.device)
            else:
                lambda_n = self.halt_node(h)[:, 0]

            y_step = self.output_node(h)
            h = self.recurrent_node(x[:, 1:], h)  # x[:, 1:] would skip subject index

            y_list.append(y_step)
            p_list.append(p_continue * lambda_n)
            p_continue = p_continue * (1 - lambda_n)

            # update halt_steps
            halt_steps = torch.max(
                halt_steps,
                ((halt_steps == 0) * n * torch.bernoulli(lambda_n)).to(torch.long)
            )

            # IGNORE: for debugging
            if False & (halt_steps > 0).sum() == batch_size:
                break

        # prepare outputs of the forward pass
        y_steps = torch.stack(y_list)  # (step, batch)
        p_steps = torch.stack(p_list)  # (step, batch)

        # the probability of halting at all the steps sums to 1
        for i in range(batch_size):
            halt_step = halt_steps[i] - 1
            p_steps[halt_step:, i] = 0.0
            p_steps[halt_step, i] = 1 - p_steps[:halt_step, i].sum()

        # y = torch.functional.F.sigmoid(y_steps)

        return y_steps, p_steps, halt_steps

    def training_step(self, batch, batch_idx):

        # unpack task-specific batch
        match self.task:
            case 'nback':
                X, trial_types, _, responses, rt_true = batch
                y_true = responses.long()
            case 'stroop':
                X, trial_types, _, responses, rt_true = batch

                # remove invalid trials (no response)
                valid_response_mask = (responses != -1)
                X = X[valid_response_mask]
                trial_types = trial_types[valid_response_mask]
                responses = responses[valid_response_mask]
                rt_true = rt_true[valid_response_mask]

                y_true = responses.long()
            case _:
                raise Exception(f'Invalid cognitive task: {self.task}')

        # forward pass
        y_steps, p_halts, rt_pred = self.forward(X)

        # compute losses
        resp_loss = self.resp_loss_fn(p_halts, y_steps, y_true)
        time_loss = self.time_loss_fn(p_halts, rt_true, logger=self.logger.experiment, step=self.global_step)
        loss = self.resp_loss_beta * resp_loss + self.time_loss_beta * time_loss

        # log losses
        self.log('train/resp_loss', resp_loss, on_epoch=True, on_step=False)
        self.log('train/time_loss', time_loss, on_epoch=True, on_step=False)
        self.log('train/total_loss', loss, on_epoch=True, logger=True, on_step=False)

        # compute and log accuracy (assuming binary classification)
        # y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
        # accuracy = (y_pred.int() == y_true.int()).float().mean()
        # self.log('train/accuracy', accuracy, on_epoch=True)
        # self.train_accuracy(y_pred, y_true.int())
        # self.log('train/accuracy', self.train_accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # unpack task-specific batch
        match self.task:
            case 'nback':
                X, trial_types, is_targets, responses, rt_true = batch
                y_true = responses.long()
            case 'stroop':
                X, trial_types, is_corrects, responses, rt_true = batch
                # remove invalid trials (no response)
                valid_response_mask = (responses != -1)
                X = X[valid_response_mask, :]
                trial_types = trial_types[valid_response_mask]
                responses = responses[valid_response_mask]
                rt_true = rt_true[valid_response_mask]

                y_true = responses.long()
            case _:
                raise Exception(f'Invalid cognitive task: {self.task}')

        # forward pass
        y_steps, p_halts, rt_pred = self.forward(X)
        y_pred = torch.argmax(y_steps, dim=-1).gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)

        # compute losses
        resp_loss = self.resp_loss_fn(p_halts, y_steps, y_true)
        time_loss = self.time_loss_fn(p_halts, rt_true)
        loss = self.resp_loss_beta * resp_loss + self.time_loss_beta * time_loss

        # log losses
        self.log('val/resp_loss', resp_loss, on_epoch=True, on_step=False)
        self.log('val/time_loss', time_loss, on_epoch=True, on_step=False)
        self.log('val/total_loss', loss, on_epoch=True, logger=True, on_step=False)

        match self.task:
            case 'nback':
                pass
            case 'stroop':

                is_corrects_pred = (y_pred.long() == y_true).float()
                cong_is_corrects = torch.where(trial_types == 1, is_corrects_pred, torch.nan)
                incong_is_corrects = torch.where(trial_types == 0, is_corrects_pred, torch.nan)

                accuracy = torch.nanmean(is_corrects_pred)
                cong_accuracy = torch.nanmean(cong_is_corrects)
                incong_accuracy = torch.nanmean(incong_is_corrects)

                self.log('val/accuracy', accuracy, on_epoch=True)
                self.log('val/accuracy_congruent', cong_accuracy, on_epoch=True)
                self.log('val/accuracy_incongruent', incong_accuracy, on_epoch=True)

        # compute and log accuracy (assuming binary classification)
        # y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
        # accuracy = (y_pred.int() == y_true.int()).float().mean()
        # self.log('val/accuracy', accuracy, on_epoch=True)
        # self.val_accuracy(y_pred, y_true.int())
        # self.log('val/accuracy', self.val_accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
