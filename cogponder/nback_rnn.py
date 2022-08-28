import torch
from torch import nn


class NBackRNN(nn.Module):
    def __init__(self, n_stimuli, n_embeddings, n_outputs):

        super(NBackRNN, self).__init__()

        self.n_stimuli = n_stimuli
        self.n_embeddings = n_embeddings

        self.embed = nn.Embedding(n_stimuli, n_embeddings)
        self.rnn = nn.RNN(n_embeddings, n_embeddings, batch_first=False)
        self.decode = nn.Sequential(
            nn.Linear(n_embeddings, n_outputs),
            nn.Sigmoid()
        )

        self.h = self.init_h()

    def forward(self, x, h=None):

        # shapes:
        #   X: seq_size, batch_size, input_size
        #   H: 1, batch_size, hidden_size
        #   Y: seq_size, batch_size, output_size

        # tODO append X to H
        x = self.embed(x)
        y, self.h = self.rnn(x, self.h)
        y = self.decode(x)

        return y

    def init_h(self):

        h0 = torch.zeros(1, self.n_embeddings)
        return h0


# DEBUG
# model = NBackRNN(n_stimuli=10, n_embeddings=5, n_outputs=2)
# X = torch.randint(0, 10, (10,))
# y, h = model(X)

# print(X, y, h)
