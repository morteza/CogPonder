import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    """Weighted average of classification losses."""

    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p, y_pred, y_true):
        """Compute reconstruction loss.
        
        Args:
        ----------
        p:
            (batch_size, steps)
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
            step_loss = p[n] * self.loss_func(y_pred[:, n, :], y_true)  # (batch_size,)
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

        p_g = torch.zeros((max_steps,))
        not_halted = 1.0

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        # self.register_buffer('p_g', p_g)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p):
        """Compute reg_loss.

        Args:
        ----------
        p : torch.Tensor
            probability of halting. Shape (steps, batch_size).
        """

        steps, batch_size = p.shape
        p = p.transpose(0, 1)  # (batch_size, steps)
        p_g_batch = self.p_g[None, :steps].expand_as(p)  # (batch_size, steps)

        return self.kl_div(p.log(), p_g_batch)
