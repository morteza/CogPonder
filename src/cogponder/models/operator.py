import torch  # noqa
from torch import nn


class SimpleOperatorModule(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, embeddings_dim=8):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.embeddings_dim = embeddings_dim
        self.outputs_dim = outputs_dim

        self.model = nn.Sequential(
            nn.Linear(self.inputs_dim, self.embeddings_dim),
            nn.ReLU(),
            nn.Linear(self.embeddings_dim, self.outputs_dim)
        )

    def forward(self, x):
        out = self.model(x)
        return out
