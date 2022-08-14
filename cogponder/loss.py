import torch
from torch import nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Weighted average of classification losses."""

    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p, y_steps, y_true):
        """Compute reconstruction loss.
        
        Args:
        ----------
        p:
            shape (batch_size, steps)
        y_pred:
            shape (batch_size, steps)
        y_true:
            shape (batch_size)

        Returns:
        ----------
        loss : torch.Tensor
            Scalar loss.
        """
        _, max_steps = p.shape
        total_loss = p.new_tensor(0.0)

        for n in range(max_steps):
            y_n = y_steps[:, n, 1]
            step_loss = p[n] * self.loss_func(y_n, y_true)  # (batch_size,)
            total_loss = total_loss + step_loss.mean()  # (1,)

        return total_loss


class RegularizationLoss(nn.Module):
    """Regularization loss for the halting steps.
    ----------
    lambda_p : float
        Hyperparameter for the the geometric distribution. Expected to be in [0, 1].
    max_steps : int
    """

    def __init__(self, lambda_p, max_steps=20):
        super().__init__()

        self.max_steps = max_steps
        self.p_g = torch.zeros((max_steps,))

        not_halted = 1.0
        for k in range(max_steps):
            self.p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        # self.register_buffer('p_g', p_g)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p, responses, response_times):
        """Compute reg_loss.

        Args:
        ----------
        p : torch.Tensor
            probability of halting. Shape (steps, batch_size).
        """

        steps, _ = p.shape

        # build an empirical RT distribution
        p_rt = torch.zeros((self.max_steps,))
        for rt in response_times:
            p_rt[rt.long() % self.max_steps] += 1
        p_rt = F.normalize(p_rt, p=1, dim=0)

        # REMOVE: geometric P_G (from PonderNet paper)
        # p_g_batch = self.p_g[:steps, ].expand_as(p)  # (batch_size, steps)
        # return self.kl_div(p.log(), p_g_batch)

        p_rt_batch = p_rt[:steps, ].expand_as(p)  # (batch_size, steps)
        return self.kl_div(p.log(), p_rt_batch)
