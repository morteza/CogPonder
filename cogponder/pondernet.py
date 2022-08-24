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

    # the halting node predicts the probability of halting conditional on not having halted before. It exactly computes overall probability of halting at each step as a geometric distribution.
    self.halt_node = nn.Sequential(
      nn.Linear(n_embeddings, 1),
      nn.Sigmoid()
    )

    # loss:  we don’t regularize PonderNet to explicitly minimize the number of computing steps, but incentivize exploration instead. The pressure of using computation efficiently happens naturally as a form of Occam’s razor.

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

    y_n, h = self.output_node(x, h)

    if n == self.max_steps:
      lambda_n = torch.ones((batch_size,))
    else:
      lambda_n = self.halt_node(h).squeeze()

    return y_n, h, lambda_n

  def forward(self, x):

    batch_size = x.size(0)

    h = torch.zeros(1, batch_size, self.n_embeddings)
    # h = self.output_node(x, h)

    p_halt = torch.zeros(batch_size)
    p_continue = torch.ones(batch_size)

    y_steps = []
    p_halts = []
    lambdas = []

    halt_step = torch.zeros((batch_size,)) # stopping step

    for n in range(1, self.max_steps + 1):

      y_n, h, lambda_n = self.step(x, h, n)

      if n == self.max_steps:
        halt_step = torch.empty((batch_size,)).fill_(n)
      else:
        _halt_step_dist = torch.distributions.Geometric(lambda_n / 5)
        halt_step = torch.maximum(_halt_step_dist.sample(), halt_step)

      p_halt = p_continue * lambda_n  # p_halt = (1-p)p
      p_continue = p_continue * (1 - lambda_n)  # update

      y_steps.append(y_n)
      # lambdas.append(lambda_n)
      p_halts.append(p_halt)

      if (halt_step <= n).all():
        break

    # prepare outputs of the forward pass
    halt_step_idx = halt_step.reshape(-1).to(torch.int64) - 1
    y_steps = torch.stack(y_steps).transpose(0, 1)
    # lambdas = torch.stack(lambdas).transpose(0, 1)

    # FIXME p_halt is not correct
    p_halts = torch.stack(p_halts).transpose(0, 1).squeeze()

    # y_pred = y_steps[0, halt_step_idx].squeeze()

    return y_steps, p_halts, halt_step_idx

# DEBUG
# from icom import ICOM
# model = PonderNet(ICOM, 5, 3, 2, 100)
# X = torch.randint(0, 5, (10, 3))
# y_steps, p_halts, halt_step = model(X)
