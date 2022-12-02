from torch import nn


class ResponseLoss(nn.Module):
    """

    # Note
    from the PonderNet paper
    We don't regularize PonderNet to explicitly minimize the number of computing
    steps, but incentivize exploration instead. The pressure of using computation
    efficiently happens naturally as a form of Occam's razor.

    """

    def __init__(self, loss_func=nn.CrossEntropyLoss()):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p_halts, y_steps, y_true):
        """Compute response reconstruction loss.

        Args:
        ----------
        p_halts : torch.Tensor
            probability of halting at each step, of shape (steps, batch_size).
        y_steps: torch.Tensor
            predicted y at each step, of shape (steps, batch_size).
        y_true:
            Ground truth y, of shape (batch_size,) of type int (class).

        Returns:
        ----------
        loss : torch.Tensor
            Scalar loss.
        """

        steps, _ = p_halts.shape
        total_loss = p_halts.new_tensor(0.0)

        for n in range(steps):
            _loss = p_halts[n] * self.loss_func(y_steps[n], y_true)
            # _loss = self.loss_func(y_steps[n], y_true)
            total_loss += _loss.mean()

        return total_loss
