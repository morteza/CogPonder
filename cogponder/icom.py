import torch
from torch import nn
import torch.nn.functional as F


class ICOM(nn.Module):
    def __init__(self, n_inputs, n_channels, n_outputs):

        super(ICOM, self).__init__()

        self.n_inputs = n_inputs
        self.n_channels = n_channels
        self.n_outputs = n_outputs

        # encode: x -> sent_msg
        self.encode = nn.Sequential(
            nn.Linear(n_inputs, n_channels, bias=False)
        )

        # transmit: sent_msg -> rcvd_msg
        self.transmit = nn.RNN(n_channels, n_channels, batch_first=False)

        # decode: rcvd_msg -> response
        self.decode = nn.Sequential(
            nn.Linear(n_channels, n_outputs, bias=False),
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
        msg = self.encode(msg)
        msg = msg.transpose(0, 1)  # reshape for RNN
        msg, h = self.transmit(msg, h)
        msg = msg.transpose(0, 1)  # reshape for linear layer
        y = self.decode(msg)
        y = y[:, -1, :]  # last time step
        # y = y.argmax(dim=1).float()
        return y, h

    def _init_h(self, batch_size):

        h0 = torch.zeros(1, batch_size, self.n_channels)
        return h0


# DEBUG
# model = ICOM(n_inputs=10, n_channels=4, n_outputs=2)
# X = torch.randint(0, 10, (10, 3))
# y_pred, h = model(X)
