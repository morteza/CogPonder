from torch import nn


class ReconstructionLoss(nn.Module):
    """

    # Note
    from the PonderNet paper
    We don't regularize PonderNet to explicitly minimize the number of computing
    steps, but incentivize exploration instead. The pressure of using computation
    efficiently happens naturally as a form of Occam's razor.

    """

    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p_halts, y_steps, y_true):
        """Compute reconstruction loss.

        Args:
        ----------
        p_halt : torch.Tensor
            probability of halting at each step, of shape (batch_size, steps).
        y_steps: torch.Tensor
            predicted y at each step, of shape (batch_size, steps).
        y_true:
            Ground truth y, of shape (batch_size,).

        Returns:
        ----------
        loss : torch.Tensor
            Scalar loss.
        """

        _, steps = p_halts.shape
        total_loss = p_halts.new_tensor(0.0)

        for n in range(steps):
            y_n = y_steps[:, n, 1]
            p_n = p_halts[:, n]
            step_loss = p_n * self.loss_func(y_n, y_true.float())  # (batch_size,)
            total_loss += step_loss.mean()  # (1,)

        return total_loss
