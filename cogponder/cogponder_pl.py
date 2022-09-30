# WARNING: this is work-in-progress

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from .loss import ReconstructionLoss, RegularizationLoss


class CogPonderNet(LightningModule):
    def __init__(
        self,
        decision_model,
        embeddings_dim,
        max_steps,
        lambda_p=0.5,
        loss_beta=.2,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        """CogPonder model written in PyTorch Lightning

        Parameters
        ----------
        decision_cls : _type_
            _description_
        embeddings_dim : _type_
            _description_
        n_outputs : _type_
            _description_
        max_steps : _type_
            _description_
        learning_rate : float, optional
            _description_, by default 1e-4
        """

        super().__init__()
        self.save_hyperparameters(ignore=['decision_model'])

        self.embeddings_dim = embeddings_dim
        self.decision_model = decision_model
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.loss_beta = loss_beta
        self.learning_rate = learning_rate

        # the halting node predicts the probability of halting conditional on not having
        # halted before. It computes overall probability of halting at each step.
        self.halt_node = nn.Sequential(
            nn.Linear(self.embeddings_dim, 1),
            nn.Sigmoid()
        )

    def ponder_step(self, x, h, step):
        """A single pondering step.
        """

        batch_size = x.shape[0]

        y, h = self.decision_model(x, h)

        if step == self.max_steps:
            lambda_n = torch.ones((batch_size,))
        else:
            lambda_n = self.halt_node(h).squeeze()

        return y, h, lambda_n

    def forward(self, x):

        batch_size = x.size(0)

        h = torch.zeros(1, batch_size, self.embeddings_dim)
        _, h = self.decision_model(x, h)  # initialize hidden state

        p_halt = torch.zeros(batch_size)
        p_continue = torch.ones(batch_size)

        y_steps = []
        p_halts = []

        halt_steps = torch.zeros((batch_size,))  # stopping step

        for step in range(1, self.max_steps + 1):

            y_n, h, lambda_n = self.ponder_step(x, h, step)

            if step == self.max_steps:
                halt_steps = torch.empty((batch_size,)).fill_(step).int()
            else:
                _halt_step_dist = torch.distributions.Geometric(lambda_n)
                halt_steps = torch.maximum(_halt_step_dist.sample(), halt_steps).int()

            p_halt = p_continue * lambda_n  # p_halt = ...(1-p)p
            p_continue = p_continue * (1 - lambda_n)  # update p_continue = ...(1-p)(1-p)

            y_steps.append(y_n)
            p_halts.append(p_halt)

            if (halt_steps <= step).all():
                break

        # prepare outputs of the forward pass
        y_steps = torch.stack(y_steps).transpose(0, 1)  # -> (batch,step)
        p_halts = torch.stack(p_halts).transpose(0, 1)  # -> (batch,step)

        # the probability of halting at all the steps sums to 1
        for i in range(batch_size):
            halt_step_idx = halt_steps[i] - 1
            p_halts[i, halt_step_idx:] = 0.0
            p_halts[i, halt_step_idx] = 1 - p_halts[i, :halt_step_idx].sum()

        return y_steps, p_halts, halt_steps

    def training_step(self, batch, batch_idx):
        X = batch['X']
        y = batch['is_target']
        resp = batch['response']
        resp_step = batch['response_step']
        y_steps, p_halt, halt_steps = self.forward(X)
        loss_rec_fn = ReconstructionLoss(nn.BCELoss(reduction='mean'))
        loss_reg_fn = RegularizationLoss(lambda_p=self.lambda_p, max_steps=self.max_steps)

        loss_rec = loss_rec_fn(p_halt, y_steps, y)
        loss_reg = loss_reg_fn(p_halt, halt_steps, resp, resp_step)
        loss = loss_rec + self.loss_beta * loss_reg

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch['X']
        y = batch['is_target']
        resp = batch['response']
        resp_step = batch['response_step']
        y_steps, p_halt, halt_steps = self.forward(X)
        loss_rec_fn = ReconstructionLoss(nn.BCELoss(reduction='mean'))
        loss_reg_fn = RegularizationLoss(lambda_p=self.lambda_p, max_steps=self.max_steps)

        loss_rec = loss_rec_fn(p_halt, y_steps, y)
        loss_reg = loss_reg_fn(p_halt, halt_steps, resp, resp_step)
        loss = loss_rec + self.loss_beta * loss_reg

        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)