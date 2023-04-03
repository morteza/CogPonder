import torch  # noqa
from torch import nn


class RecurrenceModule(nn.Module):
    def __init__(self, inputs_dim, embeddings_dim):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.embeddings_dim = embeddings_dim

        self.model = nn.GRUCell(self.inputs_dim, self.embeddings_dim)

    def forward(self, x, h):
        out = self.model(x, h)
        return out
