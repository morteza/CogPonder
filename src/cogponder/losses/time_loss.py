import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


class ResponseTimeLoss(nn.Module):
    """Regularization loss for the halting steps.
    ----------
    max_steps : int
    """

    def __init__(self, max_steps, parametric=False):
        super().__init__()

        self.max_steps = max_steps
        self.parametric = parametric

    def forward(self, p_halts, rt_true, trial_types=None, logger=None, step=None):
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
            total_loss = self.rt_dist_loss(p_halts, rt_true, logger=logger, step=step)

        return total_loss

    def log_distribution(self, logger, label, p, step):
        if logger is None or step is None:
            return

        sns.lineplot(x=range(p.cpu().size(0)), y=p.cpu().numpy())
        logger.add_figure(tag=label, figure=plt.gcf(), global_step=step)

    def rt_dist_loss(self, p_halts, rt_true, logger=None, step=None):
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

        p_halts = p_halts.transpose(0, 1)  # -> (batch_size, steps)

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
        rt_true_dist = rt_true_dist.expand(p_halts.size(0), p_halts.size(1))  # -> (batch_size, steps)

        # cumulative sum on steps
        p_halts = torch.cumsum(p_halts, dim=1).flip(1)
        rt_true_dist = torch.cumsum(rt_true_dist, dim=1).flip(1)

        # normalize
        # rt_true_dist = rt_true_dist / rt_true_dist.max(dim=1, keepdim=True)[0]
        # p_halts = p_halts / p_halts.max(dim=1, keepdim=True)[0]
        rt_true_dist = F.normalize(rt_true_dist, p=1, dim=1)  # normalize
        p_halts = F.normalize(p_halts, p=1, dim=1)  # normalize

        # log distributions
        self.log_distribution(logger, 'rt_true_dist', rt_true_dist.detach().mean(dim=0), step)
        self.log_distribution(logger, 'p_halts', p_halts.detach().mean(dim=0), step)

        # 2. compute the KL divergence between the two normalized distributions
        loss = F.kl_div(p_halts.log(), rt_true_dist, reduction='batchmean', log_target=False)

        return loss
