import torch
from torch import nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from cogponder import ICOM, NBackDataset
from torch.utils.data import TensorDataset

from cogponder.nback_rnn import NBackRNN


def test_nback_rnn(n_epochs=1000, n_stimuli=10):

    logs = SummaryWriter()

    X_train = torch.randint(0, n_stimuli, (100,))
    X_test = torch.randint(0, n_stimuli, (20,))
    y_train = torch.randint(0, 2, (100,)).float()
    y_test = torch.randint(0, 2, (20,)).float()

    model = NBackRNN(n_stimuli, n_stimuli, n_outputs=2)
    loss_fn = nn.BCELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(n_epochs), desc='Epochs'):

        model.train()
        y_pred = model(X_train)

        loss = loss_fn(y_pred[:, 1], y_train)

        # logs
        logs.add_scalar('loss/train', loss, epoch)

        # forward + backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_accuracy = accuracy_score(y_pred.detach().numpy().argmax(1), y_train.detach().numpy())

        logs.add_scalar('accuracy/train', train_accuracy, epoch)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            loss = loss_fn(y_pred[:, 1], y_test)
            logs.add_scalar('loss/test', loss.detach(), epoch)
            test_accuracy = accuracy_score(y_pred.detach().numpy().argmax(1), y_test.detach().numpy())
            logs.add_scalar('accuracy/test', test_accuracy, epoch)

    logs.close()


def test_nback_icom(n_epochs=10000, n_stimuli=6):

    logs = SummaryWriter()

    dataset = NBackDataset(n_subjects=2, n_trials=100, n_stimuli=n_stimuli)

    X, targets, responses, response_times = dataset[0]
    dataset = TensorDataset(X, targets.float(), responses, response_times)

    n_outputs = torch.unique(targets).size()[0]

    # split params
    train_size = int(len(dataset) * .8)
    test_size = len(dataset) - train_size

    train_subset, test_subset = random_split(dataset, lengths=(train_size, test_size))
    X_train, y_train, r_train, rt_train = dataset[train_subset.indices]
    X_test, y_test, r_test, rt_test = dataset[test_subset.indices]

    model = ICOM(n_stimuli + 1, n_stimuli, n_outputs)
    loss_fn = nn.BCELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(n_epochs), desc='Epochs'):

        model.train()
        y_pred, _ = model(X_train)

        loss = loss_fn(y_pred[:, 1], y_train)

        # logs
        logs.add_scalar('loss/train', loss, epoch)

        # forward + backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_accuracy = accuracy_score(y_pred.detach().numpy().argmax(1), y_train.detach().numpy())

        logs.add_scalar('accuracy/train', train_accuracy, epoch)

        model.eval()
        with torch.no_grad():
            y_pred, _ = model(X_test)
            loss = loss_fn(y_pred[:, 1], y_test)
            logs.add_scalar('loss/test', loss.detach(), epoch)
            test_accuracy = accuracy_score(y_pred.detach().numpy().argmax(1), y_test.detach().numpy())
            logs.add_scalar('accuracy/test', test_accuracy, epoch)

    logs.close()
