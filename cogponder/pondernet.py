import torch
from torch import nn
import torch.nn.functional as F


class PonderNet(nn.Module):
  def __init__(self, output_cls, n_inputs, n_embeddings, n_outputs, max_steps):
    super(PonderNet, self).__init__()

    self.n_embeddings = n_embeddings
    self.n_outputs = n_outputs
    self.max_steps = max_steps

    self.output_node = output_cls(n_inputs, n_embeddings, n_outputs)

    # the halting node predicts the probability of halting conditional on not having
    # halted before. It computes overall probability of halting at each step.
    self.halt_node = nn.Sequential(
        nn.Linear(n_embeddings, 1),
        nn.Sigmoid()
    )

  def step(self, x, h, n):
    """A single pondering step.

    Args:
    -------
    x: current input of shape (batch_size, n_inputs)
    h: previous hidden state of shape (batch_size, n_embeddings, n_xxx)

    Returns
    -------
    lambda_n : float
        probability of the "continue->halt" transition
    """

    batch_size = x.shape[0]

    y, h = self.output_node(x, h)

    if n == self.max_steps:
      lambda_n = torch.ones((batch_size,))
    else:
      lambda_n = self.halt_node(h).squeeze()

    return y, h, lambda_n

  def forward(self, x):

    batch_size = x.size(0)

    h = torch.zeros(1, batch_size, self.n_embeddings)
    _, h = self.output_node(x, h)  # initialize hidden state

    p_halt = torch.zeros(batch_size)
    p_continue = torch.ones(batch_size)

    y_steps = []
    p_halts = []

    halt_steps = torch.zeros((batch_size,))  # stopping step

    for n in range(1, self.max_steps + 1):

      y_n, h, lambda_n = self.step(x, h, n)

      if n == self.max_steps:
        halt_steps = torch.empty((batch_size,)).fill_(n).int()
      else:
        _halt_step_dist = torch.distributions.Geometric(lambda_n)
        halt_steps = torch.maximum(_halt_step_dist.sample(), halt_steps).int()

      p_halt = p_continue * lambda_n  # p_halt = ...(1-p)p
      p_continue = p_continue * (1 - lambda_n)  # update p_continue = ...(1-p)(1-p)

      y_steps.append(y_n)
      p_halts.append(p_halt)

      if (halt_steps <= n).all():
        break

    # prepare outputs of the forward pass
    y_steps = torch.stack(y_steps).transpose(0, 1)
    p_halts = torch.stack(p_halts).transpose(0, 1)

    # the probability of halting at all the steps sums to 1
    p_halts[:, -1] = 1 - p_halts[:, :-1].sum(dim=1)

    # halt_step_idx = halt_step.reshape(-1).to(torch.int64) - 1
    # y_pred = y_steps[0, halt_step_idx].squeeze()

    return y_steps, p_halts, halt_steps

# DEBUG
# from icom import ICOM
# model = PonderNet(ICOM, 5, 3, 2, 100)
# X = torch.randint(0, 5, (10, 3))
# y_steps, p_halts, halt_step = model(X)
