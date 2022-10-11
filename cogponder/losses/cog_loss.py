import torch
from torch import nn


class CognitiveLoss(nn.Module):
    """Regularization loss for the halting steps.
    ----------
    max_steps : int
    """

    def __init__(self, max_steps):
        super().__init__()

        self.max_steps = max_steps
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=True)

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
            total_loss = (rt_pred - rt_true).abs().float().mean()

        return total_loss

    def rt_dist_loss(self, rt_pred, rt_true):
        """Compute the KL divergence between the predicted and true distributions.

        Args:
        ----------
        halt_steps : torch.Tensor
            steps at which the network halted. Shape (batch_size,).
        rt_true : torch.Tensor
            Ground truth response times digitized in steps.
            Shape (batch_size, steps) of type int.
        """

        steps = rt_true.max().item()  # maximum number of steps in the batch

        # 1. compute distributions
        rt_true_dist = torch.zeros((steps,))
        rt_true_idx, rt_true_cnt = torch.unique(rt_true, return_counts=True)
        rt_true_dist[rt_true_idx.long()] += rt_true_cnt

        rt_pred_dist = torch.zeros((steps,))
        rt_pred_idx, rt_pred_cnt = torch.unique(rt_pred, return_counts=True)
        rt_pred_dist[rt_pred_idx.long()] += rt_pred_cnt

        # 3. compute the KL divergence between the two distributions
        loss = self.kl_div(rt_pred_dist, rt_true_dist)

        return loss
