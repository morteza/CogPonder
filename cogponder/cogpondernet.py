# WARNING: this is work-in-progress

import torch
from torch import nn
from pytorch_lightning import LightningModule
from cogponder.losses import ReconstructionLoss, CognitiveLoss
import torchmetrics


class CogPonderNet(LightningModule):
    def __init__(
        self,
        config,
        example_input_array=None
    ):
        """CogPonder model written in PyTorch Lightning

        Parameters
        ----------
        max_response_step : _type_
            _description_
        learning_rate : float, optional
            _description_, by default 1e-4
        """

        super().__init__()
        self.save_hyperparameters(ignore=['example_input_array'], logger=False)

        self.inputs_dim = config['inputs_dim']
        self.embeddings_dim = config['embeddings_dim']
        self.rec_loss_beta = config['rec_loss_beta']
        self.cog_loss_beta = config['cog_loss_beta']
        self.loss_by_trial_type = config['loss_by_trial_type']
        self.learning_rate = config['learning_rate']
        self.max_response_step = config['max_response_step']

        self.example_input_array = example_input_array
        
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.cog_loss_fn = CognitiveLoss(self.max_response_step)
        self.rec_loss_fn = ReconstructionLoss()

        # init nodes: halt_node, output_node, recurrent_node
        self.halt_node = nn.Sequential(
            nn.Linear(self.embeddings_dim, 1),
            nn.Sigmoid()
        )

        self.output_node = nn.Sequential(
            nn.Linear(self.embeddings_dim, 1),
            nn.Sigmoid()
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

            y_step = self.output_node(h)[:, 0]
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
        y = torch.stack(y_list)  # (step, batch)
        p = torch.stack(p_list)  # (step, batch)

        # the probability of halting at all the steps sums to 1
        for i in range(batch_size):
            halt_step = halt_steps[i] - 1
            p[halt_step:, i] = 0.0
            p[halt_step, i] = 1 - p[:halt_step, i].sum()

        return y, p, halt_steps

    def training_step(self, batch, batch_idx):
        X, trial_types, is_targets, responses, rt_true = batch
        y_true = responses.float()
        y_steps, p_halts, rt_pred = self.forward(X)

        rec_loss = self.rec_loss_fn(p_halts, y_steps, y_true)
        cog_loss = self.cog_loss_fn(p_halts, rt_true)
        loss = self.rec_loss_beta * rec_loss + self.cog_loss_beta * cog_loss

        y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
        self.train_accuracy(y_pred, y_true.int())
        self.log('train_accuracy', self.train_accuracy, on_epoch=True)

        self.log('train_loss_rec', rec_loss, on_epoch=True)
        self.log('train_loss_cog', cog_loss, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, trial_types, is_targets, responses, rt_true = batch
        y_true = responses.float()
        y_steps, p_halts, rt_pred = self.forward(X)

        rec_loss = self.rec_loss_fn(p_halts, y_steps, y_true)
        cog_loss = self.cog_loss_fn(p_halts, rt_true)
        loss = self.rec_loss_beta * rec_loss + self.cog_loss_beta * cog_loss

        y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
        self.val_accuracy(y_pred, y_true.int())
        self.log('val_accuracy', self.val_accuracy, on_epoch=True)

        self.log('val_loss_rec', rec_loss, on_epoch=True)
        self.log('val_loss_cog', cog_loss, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
