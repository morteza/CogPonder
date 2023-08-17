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


class SpatioTemporalOperatorModule(nn.Module):
    def __init__(self,
                 inputs_dim, outputs_dim,
                 space_embedding_dim, time_embedding_dim):

        super().__init__()
        self.inputs_dim = inputs_dim
        self.space_embedding_dim = space_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.outputs_dim = outputs_dim

        # simple CNN to encode space features
        self.space_encoder = nn.Sequential(
            nn.Conv1d(inputs_dim, space_embedding_dim, kernel_size=3),
            nn.ReLU(),
        )

        # simple LSTM to encode time features
        self.time_encoder = nn.LSTM(
            space_embedding_dim,
            time_embedding_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_embedding_dim, outputs_dim),
        )

    def forward(self, x):
        # x: (batch_size, seq_len, features)

        # space
        y_space = self.space_encoder(x.transpose(1, 2))  # (batch_size, features, seq_len)
        y_space = y_space.transpose(1, 2)  # (batch_size, seq_len, features)

        # time
        y_time, _ = self.time_encoder(y_space)  # (batch_size, seq_len, features)
        y = y_time[:, -1, :]  # (batch_size, space_embedding_dim, time_embedding_dim)

        # output
        y = self.fc(y)

        return y
