# WARNING: this is work-in-progress

import torch
from torch import nn
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

    def forward(self, x):

        batch_size = x.size(0)

        h = torch.zeros(batch_size, self.embeddings_dim, device=self.device)
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
            h = self.recurrent_node(x, h)

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
                X = X[:, 1:]  # FIXME this removes the first column (subject index)
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
        time_loss = self.time_loss_fn(p_halts, rt_true)
        loss = self.resp_loss_beta * resp_loss + self.time_loss_beta * time_loss

        # compute accuracy and log metrics (only in the case of binary classification)
        if torch.unique(y_true).shape[0] == 2:
            y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
            self.train_accuracy(y_pred, y_true.int())
            self.log('train/accuracy', self.train_accuracy, on_epoch=True)

        self.log('train/resp_loss', resp_loss, on_epoch=True)
        self.log('train/time_loss', time_loss, on_epoch=True)
        self.log('train/total_loss', loss, on_epoch=True, logger=True)

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
                X = X[valid_response_mask]
                X = X[:, 1:]  # FIXME this removes the first column (subject index)
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
        time_loss = self.time_loss_fn(p_halts, rt_true)
        loss = self.resp_loss_beta * resp_loss + self.time_loss_beta * time_loss

        # compute accuracy and log metrics (only in the case of binary classification)
        if torch.unique(y_true).shape[0] == 2:
            y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
            self.train_accuracy(y_pred, y_true.int())
            self.log('val/accuracy', self.val_accuracy, on_epoch=True)

        self.log('val/resp_loss', resp_loss, on_epoch=True)
        self.log('val/time_loss', time_loss, on_epoch=True)
        self.log('val/total_loss', loss, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
