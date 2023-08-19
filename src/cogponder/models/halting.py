import torch  # noqa
from torch import nn


class HaltingModule(nn.Module):
    def __init__(self, embeddings_dim, max_response_step):
        super().__init__()
        self.embeddings_dim = embeddings_dim
        self.max_response_step = max_response_step

        self.model = nn.Sequential(
            nn.Linear(self.embeddings_dim, self.embeddings_dim),
            nn.ReLU(),
            nn.Linear(self.embeddings_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h, step):

        batch_size = h.squeeze(0).size(0)

        if step == self.max_response_step:
            lambda_n = torch.ones((batch_size,))
        else:
            lambda_n = self.model(h.squeeze(0))[:, 0]

        return lambda_n
