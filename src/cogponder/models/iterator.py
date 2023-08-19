import torch  # noqa
from torch import nn


class IteratorModule(nn.Module):
    def __init__(self, inputs_dim, embeddings_dim, n_subjects, subject_embeddings_dim):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.embeddings_dim = embeddings_dim
        self.n_subjects = n_subjects
        self.subject_embeddings_dim = subject_embeddings_dim

        # init subject embeddings (if applicable)
        if self.n_subjects is not None:
            self.subject_embeddings = nn.Embedding(self.n_subjects,
                                                   self.subject_embeddings_dim,
                                                   dtype=torch.float)

        self.model = nn.GRU(self.inputs_dim + self.subject_embeddings_dim,
                            self.embeddings_dim,
                            batch_first=True)

    def forward(self, x, h, subject_ids):
        """Forward pass of the recurrence module

        Parameters
        ----------
        x : Tensor
            input data of shape (batch_size, seq_len, inputs_dim)
        h : Tensor
            contextual information of shape (batch_size, embeddings_dim)

        Returns
        -------
        Tensor
            next contextual information of shape (batch_size, embeddings_dim)
        """

        batch_size, seq_len, _ = x.shape

        # append subject-specific embeddings
        subject_features = self.subject_embeddings(subject_ids.int())
        subject_features = subject_features.unsqueeze(1).repeat(1, seq_len, 1)
        x_subject = torch.cat([x, subject_features], dim=-1)

        y, _ = self.model(x_subject, h.unsqueeze(0))
        h = y[:, -1, :]  # (batch_size, embeddings_dim)

        return h
