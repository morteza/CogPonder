import torch  # noqa
from torch import nn


class RecurrenceModule(nn.Module):
    def __init__(self, inputs_dim, embeddings_dim, n_subjects, subject_embeddings_dim):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.embeddings_dim = embeddings_dim
        self.n_subjects = n_subjects
        self.subject_embeddings_dim = subject_embeddings_dim

        # init subject embeddings (if applicable)
        if self.n_subjects is not None:
            self.subject_embeddings = nn.Embedding(self.n_subjects, self.subject_embeddings_dim, dtype=torch.float)

        self.model = nn.GRUCell(self.inputs_dim + self.subject_embeddings_dim, self.embeddings_dim)

    def forward(self, x, h, subject_ids):
        """Forward pass of the recurrence module

        Parameters
        ----------
        x : Tensor
            input data of shape (batch_size, inputs_dim)
        h : Tensor
            contextual information of shape (batch_size, embeddings_dim)

        Returns
        -------
        Tensor
            next contextual information of shape (batch_size, embeddings_dim)
        """
        # append subject-specific embeddings
        subject_features = self.subject_embeddings(subject_ids.int())
        x = torch.cat([x, subject_features], dim=1)

        out = self.model(x, h)

        return out
