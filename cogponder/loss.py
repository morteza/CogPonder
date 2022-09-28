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
            step_loss = p_halt[:, n] * self.loss_func(y_n, y_true)  # (batch_size,)
            total_loss += step_loss.mean()  # (1,)

        return total_loss


class RegularizationLoss(nn.Module):
    """Regularization loss for the halting steps.
    ----------
    lambda_p : float
        Hyperparameter for the the geometric distribution. Expected to be in [0, 1].
    max_steps : int
    """

    def __init__(self, lambda_p, max_steps):
        super().__init__()

        self.max_steps = max_steps
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        # 1. stores the geometric distribution at each step
        p_g = torch.zeros((self.max_steps,))

        not_halted = 1.0  # initial probability of not having halted

        for step in range(self.max_steps):
            p_g[step] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)
        self.register_buffer('p_g', p_g)  # persist p_g

    def forward(self, p_halts, halt_steps, responses, response_times):
        """Compute reg_loss.

        Notes: `reg_loss = KL(p_halts || p_g) + KL(halt_steps || p_rt_empirical)`

        Args:
        ----------
        p_halts : torch.Tensor
            probability of halting. Shape (steps, batch_size).
        halt_steps : torch.Tensor
            steps at which the network halted. Shape (batch_size,).
        responses : torch.Tensor

        response_times : torch.Tensor
            Response times converted to steps. Shape (batch_size, steps) of type int.
        """

        _, steps = p_halts.shape

        # component 1.  KL between p_halt and geometric distribution (p_g)
        p_g_batch = self.p_g[:steps, ].expand_as(p_halts)  # (batch_size, steps)
        p_g_loss = self.kl_div(p_halts, p_g_batch)

        # component 2. KL between halt steps and empirical RT distribution
        p_rt_empirical = torch.zeros((self.max_steps + 1,))
        p_rt_pred = torch.zeros((self.max_steps + 1,))

        # counting RTs; this is a quick version of a loop over RTs, borrowed from:
        # https://stackoverflow.com/questions/66315038
        rt_idx, rt_cnt = torch.unique(response_times, return_counts=True)
        p_rt_empirical[rt_idx.long()] += rt_cnt

        # counting halting steps
        rt_pred_idx, rt_pred_cnt = torch.unique(halt_steps, return_counts=True)
        p_rt_pred[rt_pred_idx.long()] += rt_pred_cnt

        # cap at maximum halting step for the batch
        p_rt_empirical = p_rt_empirical[1:steps + 1]
        p_rt_pred = p_rt_pred[1:steps + 1]

        # normalize the probability distributions
        p_rt_empirical = F.normalize(p_rt_empirical, p=1, dim=0)
        p_rt_pred = F.normalize(p_rt_pred, p=1, dim=0)

        empirical_loss = self.kl_div(p_rt_pred, p_rt_empirical)

        # Note: Original PonderNet returns only p_g_loss
        return p_g_loss + empirical_loss
