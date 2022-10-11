import torch
from torch import nn
import torch.nn.functional as F


class ICOM(nn.Module):
    def __init__(self, n_inputs, n_embeddings, n_outputs):

        super(ICOM, self).__init__()

        self.n_inputs = n_inputs
        self.n_embeddings = n_embeddings
        self.n_outputs = n_outputs

        # encode: x -> sent_msg
        self.encode = nn.Sequential(
            nn.Linear(n_inputs, n_embeddings, bias=False)
        )

        # transmit: sent_msg -> rcvd_msg
        self.transmit = nn.RNN(n_embeddings, n_embeddings, batch_first=False)

        # decode: rcvd_msg -> response
        self.decode = nn.Sequential(
            nn.Linear(n_embeddings, n_outputs, bias=False),
            nn.Softmax(dim=2)
        )

    def forward(self, x, h=None):

        # shapes:
        #   X: seq_size, batch_size, input_size
        #   H: 1, batch_size, hidden_size
        #   Y: seq_size, batch_size, output_size

        batch_size = x.size(0)

        if h is None:
            h = self._init_h(batch_size)

        msg = F.one_hot(x, num_classes=self.n_inputs).type(torch.float)
        print('[1]', msg.shape)
        msg = self.encode(msg)
        print('[2]', msg.shape)
        msg = msg.transpose(0, 1)  # reshape for RNN
        print('[3]', msg.shape)
        msg, h = self.transmit(msg, h)
        print('[4]', msg.shape)
        msg = msg.transpose(0, 1)  # reshape for linear layer
        print('[5]', msg.shape)
        msg = self.decode(msg)
        print('[6]', msg.shape)
        y = msg[:, -1, :]  # last output (in n-back)
        # y = y.argmax(dim=1).float()
        return y, h

    def _init_h(self, batch_size):

        h0 = torch.zeros(1, batch_size, self.n_embeddings)
        return h0


# DEBUG
# model = ICOM(n_inputs=10, n_embeddings=4, n_outputs=2)
# X = torch.randint(0, 10, (10, 3))
# y_pred, h = model(X)
