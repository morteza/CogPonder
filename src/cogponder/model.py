# WARNING: this is work-in-progress

import torch
from torch import nn
from pytorch_lightning import LightningModule
from .losses import ResponseLoss, ResponseTimeLoss
# import torchmetrics


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
        self.n_contexts = config['n_contexts']
        self.task = config['task']

        self.example_input_array = example_input_array

        # self.train_accuracy = torchmetrics.Accuracy(task='multiclass')
        # self.val_accuracy = torchmetrics.Accuracy(task='multiclass')

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

        self.embeddings = nn.Embedding(
            self.n_contexts,
            self.embeddings_dim,
            device=self.device)

    def forward(self, context, x):

        """_summary_

        Args
        ----
            context (torch.Tensor): contextual information of shape (batch_size,)
            x (torch.Tensor): input data of shape (batch_size, n_contexts, inputs_dim)

        Returns
        -------
            (y_steps, p_steps, halt_steps)
        """

        batch_size = x.size(0)

        h = self.embeddings(context)
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

            # IGNORE: enable for debugging or stopping the recurrent loop upon halt.
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

        return y_steps, p_steps, halt_steps

    def training_step(self, batch, batch_idx):

        contexts, X, trial_types, is_corrects, responses, rt_true = batch
        y_true = responses.long()

        # task-specific cleanups
        match self.task:
            case 'nback':
                valid_response_mask = (rt_true > 0) & (responses != 0)
                contexts = contexts[valid_response_mask, ...]
                X = X[valid_response_mask, ...]
                trial_types = trial_types[valid_response_mask]
                y_true = y_true[valid_response_mask]
                rt_true = rt_true[valid_response_mask]
            case 'stroop':
                # remove invalid trials (no response)
                valid_response_mask = (responses != -1)
                contexts = contexts[valid_response_mask, ...]
                X = X[valid_response_mask, :]
                trial_types = trial_types[valid_response_mask]
                y_true = y_true[valid_response_mask]
                rt_true = rt_true[valid_response_mask]
            case _:
                raise Exception(f'Invalid cognitive task: {self.task}')

        # forward pass
        y_steps, p_halts, rt_pred = self.forward(contexts, X)
        
        print(y_steps.shape, p_halts.shape, rt_pred.shape)
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

        contexts, X, trial_types, is_corrects, responses, rt_true = batch
        print('val contexts', torch.unique(contexts))
        y_true = responses.long()

        # task-specific cleanups
        match self.task:
            case 'nback':
                valid_response_mask = (rt_true > 0) & (responses != 0)
                contexts = contexts[valid_response_mask, ...]
                X = X[valid_response_mask, ...]
                trial_types = trial_types[valid_response_mask]
                y_true = y_true[valid_response_mask]
                rt_true = rt_true[valid_response_mask]
            case 'stroop':
                # remove invalid trials (no response)
                valid_response_mask = (responses != -1)
                contexts = contexts[valid_response_mask, ...]
                X = X[valid_response_mask, :]
                trial_types = trial_types[valid_response_mask]
                y_true = y_true[valid_response_mask]
                rt_true = rt_true[valid_response_mask]
            case _:
                raise Exception(f'Invalid cognitive task: {self.task}')

        # forward pass
        y_steps, p_halts, rt_pred = self.forward(contexts, X)
        y_pred = torch.argmax(y_steps, dim=-1).gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)

        print(y_steps.shape, p_halts.shape, rt_pred.shape)

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
                # compute and log accuracy (assuming binary classification)
                # y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
                # accuracy = (y_pred.int() == y_true.int()).float().mean()
                # self.log('val/accuracy', accuracy, on_epoch=True)
                # self.val_accuracy(y_pred, y_true.int())
                # self.log('val/accuracy', self.val_accuracy, on_epoch=True)
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

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
