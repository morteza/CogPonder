import torch
from torch import nn
import torch.nn.functional as F


class CognitiveLoss(nn.Module):
    """Regularization loss for the halting steps.
    ----------
    lambda_p : float
        Hyperparameter for the the geometric distribution. Expected to be in [0, 1].
    max_steps : int
    """

    def __init__(self, lambda_p, max_steps):
        super().__init__()

        self.lambda_p = lambda_p
        self.max_steps = max_steps
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=True)

        # 1. stores the geometric distribution at each step
        p_g = torch.zeros((self.max_steps,))
        p_continue = 1.0  # initial probability of not having halted
        for step in range(self.max_steps):
            p_g[step] = p_continue * self.lambda_p
            p_continue = p_continue * (1 - self.lambda_p)
        self.register_buffer('p_g', p_g)  # persist p_g

    def forward(self, rt_pred, rt_true, trial_types=None):
        """Compute reg_loss.

        Args:
        ----------
        halt_steps : torch.Tensor
            steps at which the network halted. Shape (batch_size,).
        rt_true : torch.Tensor
            Ground truth response times digitized in steps.
            Shape (batch_size, steps) of type int.
        """

        # steps = halt_steps.max().item()  # maximum number of steps in the batch

        if trial_types is not None:
            total_loss = torch.tensor(0.)
            for tt in torch.unique(trial_types):
                _mask = (trial_types == tt)
                _halt_steps = rt_pred[_mask]
                _rt_true = rt_true[_mask]
                loss = (_halt_steps - _rt_true).abs().float().mean()
                total_loss += loss
        else:
            total_loss = loss = (rt_pred - rt_true).abs().float().mean()

        return total_loss
