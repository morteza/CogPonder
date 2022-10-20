# TODO

import torch
from torch import nn


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

    def forward(self, trial_types, p_halts, halt_steps, response_steps):
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
            total_loss = torch.tensor(0.)
            for tt in torch.unique(trial_types):
                _mask = (trial_types == tt)
                _response_steps = response_steps[_mask]
                _halt_steps = halt_steps[_mask]
                loss = self._compute_rt_loss(_response_steps, _halt_steps, steps)
                total_loss += loss
        else:
            total_loss = self._compute_rt_loss(response_steps, halt_steps, steps)

        return total_loss

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
        # p_rt_true = F.normalize(p_rt_true, p=1, dim=0)
        # p_rt_pred = F.normalize(p_rt_pred, p=1, dim=0)

        # ALT: rt_loss = torch.abs(rt_pred.float() - rt_true.float())
        loss = self.kl_div(p_rt_pred, p_rt_true)

        return loss
