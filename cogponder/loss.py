# from the PonderNet paper
# We don’t regularize PonderNet to explicitly minimize the number of computing
# steps, but incentivize exploration instead. The pressure of using computation
# efficiently happens naturally as a form of Occam’s razor.

import torch
from torch import nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Weighted average of classification losses."""

    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p_halt, y_steps, y_true):
        """Compute reconstruction loss.

        Args:
        ----------
        p_halt : torch.Tensor
            probability of halting. Shape (steps, batch_size).
        y_steps: torch.Tensor
            predicted y at each step. Shape (batch_size, steps)
        y_true:
            Ground truth y. Shape (batch_size, steps)

        Returns:
        ----------
        loss : torch.Tensor
            Scalar loss.
        """

        _, max_steps = p_halt.shape
        total_loss = p_halt.new_tensor(0.0)

        for n in range(max_steps):
            y_n = y_steps[:, n, 1]
            step_loss = p_halt[:, n] * self.loss_func(y_n, y_true.float())  # (batch_size,)
            total_loss += step_loss.mean()  # (1,)

        return total_loss


class RegularizationLoss(nn.Module):
    """Regularization loss for the halting steps.
    ----------
    lambda_p : float
        Hyperparameter for the the geometric distribution. Expected to be in [0, 1].
    max_steps : int
    """

    def __init__(self, lambda_p, max_steps, by_trial_type=True):
        super().__init__()

        self.lambda_p = lambda_p
        self.max_steps = max_steps
        self.by_trial_type = by_trial_type
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        # 1. stores the geometric distribution at each step
        p_g = torch.zeros((self.max_steps,))

        not_halted = 1.0  # initial probability of not having halted

        for step in range(self.max_steps):
            p_g[step] = not_halted * self.lambda_p
            not_halted = not_halted * (1 - self.lambda_p)
        self.register_buffer('p_g', p_g)  # persist p_g

    def forward(self, p_halts, halt_steps, trial_types, response_steps):
        """Compute reg_loss.

        Notes: `reg_loss = KL(p_rt_pred || p_rt_true)`, which can also be separated by trial_type.

        Args:
        ----------
        p_halts : torch.Tensor
            probability of halting. Shape (steps, batch_size).
        halt_steps : torch.Tensor
            steps at which the network halted. Shape (batch_size,).
        is_targets : torch.Tensor
            trial_type
        response_steps : torch.Tensor
            Response times in steps. Shape (batch_size, steps) of type int.
        """

        _, steps = p_halts.shape

        if self.by_trial_type:
            # TODO: use torch.unique() to get the unique trial_type

            loss = torch.tensor(0.)
            for trial_type in torch.unique(trial_types):
                _response_steps = response_steps[trial_type]
                _halt_steps = halt_steps[trial_type]
                loss += self._compute_rt_loss(_response_steps, _halt_steps, steps)
        else:
            loss = self._compute_rt_loss(response_steps, halt_steps, steps)

        return loss

    def _compute_rt_loss(self, rt_true, rt_pred, steps):
        """CogPonder regularizer: KL between halt steps and empirical RT distribution.
        """
        p_rt_true = torch.zeros((self.max_steps + 1,))
        p_rt_pred = torch.zeros((self.max_steps + 1,))

        # counting RTs; this is a quick version of a loop over RTs, borrowed from:
        # https://stackoverflow.com/questions/66315038
        rt_idx, rt_cnt = torch.unique(rt_true, return_counts=True)
        p_rt_true[rt_idx.long()] += rt_cnt

        # counting halting steps (rt_pred)
        rt_pred_idx, rt_pred_cnt = torch.unique(rt_pred, return_counts=True)
        p_rt_pred[rt_pred_idx.long()] += rt_pred_cnt

        # cap at maximum halting step for the batch
        # TODO each batch item must be capped separately
        p_rt_true = p_rt_true[1:steps + 1]
        p_rt_pred = p_rt_pred[1:steps + 1]

        # normalize the probability distributions
        p_rt_true = F.normalize(p_rt_true, p=1, dim=0)
        p_rt_pred = F.normalize(p_rt_pred, p=1, dim=0)

        rt_loss = self.kl_div(p_rt_pred, p_rt_true)

        return rt_loss
