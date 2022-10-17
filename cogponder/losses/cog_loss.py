import torch
from torch import nn
import torch.nn.functional as F


class CognitiveLoss(nn.Module):
    """Regularization loss for the halting steps.
    ----------
    max_steps : int
    """

    def __init__(self, max_steps, parametric=False):
        super().__init__()

        self.max_steps = max_steps
        self.parametric = parametric
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p_halts, rt_true, trial_types=None):
        """Compute reg_loss.

        Args:
        ----------
        p_halts : torch.Tensor
            probs of steps at which the network halted. Shape (steps, batch_size).
        rt_true : torch.Tensor
            Ground truth response times digitized in steps.
            Shape (batch_size, steps) of type int.
        """

        # steps = halt_steps.max().item()  # maximum number of steps in the batch

        if trial_types is not None:
            total_loss = torch.tensor(0.)
            for tt in torch.unique(trial_types):
                _mask = (trial_types == tt)
                _halt_steps = p_halts[_mask]
                _rt_true = rt_true[_mask]
                loss = (_halt_steps - _rt_true).abs().float().mean()
                total_loss += loss
        else:
            # total_loss = (rt_pred.float().mean() - rt_true.float().mean()).abs()
            # total_loss = (rt_pred - rt_true).abs().float().mean()
            total_loss = self.rt_dist_loss(p_halts, rt_true)

        return total_loss

    def rt_dist_loss(self, p_halts, rt_true):
        """Compute the KL divergence between the predicted and true distributions.

        Args:
        ----------
        p_halts : torch.Tensor
            probs of steps at which the network halted. Shape (steps,batch_size).
        rt_true : torch.Tensor
            Ground truth response times digitized in steps.
            Shape (batch_size, steps) of type int.
        """

        steps = rt_true.max().item()  # maximum number of steps in the batch

        # 1. compute normal distributions

        if self.parametric:
            # TODO 
            # rt_true_norm = torch.distributions.Normal(rt_true.float().mean(), rt_true.float().std())
            # rt_pred_norm = torch.distributions.Normal(rt_pred.float().mean(), rt_pred.float().std())
            # rt_true_dist = rt_true_norm.log_prob(torch.arange(0, self.max_steps + 1)).exp()
            pass

        rt_true_dist = rt_true.new_zeros((self.max_steps,), dtype=torch.float)
        rt_true_idx, rt_pred_cnt = torch.unique(rt_true, return_counts=True)
        rt_true_dist[rt_true_idx.long()] += rt_pred_cnt
        rt_true_dist = F.normalize(rt_true_dist, p=1, dim=0)  # normalize

        p_halts = p_halts.transpose(0, 1)

        # 2. compute the KL divergence between the two normalized distributions
        # loss = torch.distributions.kl_divergence(rt_pred_norm.log(), rt_true_norm)
        loss = self.kl_div(p_halts, rt_true_dist)

        return loss
