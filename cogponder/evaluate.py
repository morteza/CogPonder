import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loss import ReconstructionLoss, RegularizationLoss


def evaluate(
    model,
    dataset,
    optimizer,
    n_epochs=1000,
    batch_size=4,
    max_steps=100,
    loss_beta=.3,
    lambda_p=.5,
    logs=SummaryWriter(),
    randomized_split=False,
):

    # split params
    train_size = int(len(dataset) * .8)
    test_size = len(dataset) - train_size

    if randomized_split:
        train_subset, test_subset = random_split(dataset, lengths=(train_size, test_size))
        X_train, y_train, r_train, rt_train = dataset[train_subset.indices]
        X_test, y_test, r_test, rt_test = dataset[test_subset.indices]
    else:
        X_train, y_train, r_train, rt_train = dataset[:train_size]
        X_test, y_test, r_test, rt_test = dataset[train_size:]

    loss_rec_fn = ReconstructionLoss(nn.BCELoss(reduction='mean'))
    loss_reg_fn = RegularizationLoss(lambda_p=lambda_p, max_steps=max_steps)

    for epoch in tqdm(range(n_epochs), desc='Epochs'):

        # running_loss = 0.0
        # for X_batch, y_batch, r_batch, rt_batch in DataLoader(train_subset,
        #                                                       batch_size=batch_size,
        #                                                       shuffle=True):

        model.train()
        y_steps, p_halt, halt_step = model(X_train)
        halt_step_idx = halt_step.reshape(-1).to(torch.int64) - 1

        loss_rec = loss_rec_fn(p_halt, y_steps, y_train)
        loss_reg = loss_reg_fn(p_halt, r_train, rt_train)
        loss = loss_rec + loss_beta * loss_reg
        # running_loss += loss.item()

        # logs
        logs.add_scalar('loss/rec_train', loss_rec, epoch)
        logs.add_scalar('loss/reg_train', loss_reg, epoch)
        logs.add_scalar('loss/train', loss, epoch)

        # forward + backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = y_steps.detach()[0, halt_step_idx].argmax(dim=1)
        batch_accuracy = accuracy_score(y_pred, y_train)

        logs.add_scalar('accuracy/train', batch_accuracy, epoch)

        model.eval()
        with torch.no_grad():
            y_steps, p_halt, halt_step = model(X_test)
            halt_step_idx = halt_step.reshape(-1).to(torch.int64) - 1

            loss_rec = loss_rec_fn(p_halt, y_steps, y_test)
            loss_reg = loss_reg_fn(p_halt, r_test, rt_test)
            loss = loss_rec + loss_beta * loss_reg
            logs.add_scalar('loss/test', loss.detach(), epoch)

            y_pred = y_steps.detach()[0, halt_step_idx].argmax(dim=1)
            test_accuracy = accuracy_score(y_pred, y_test)

            logs.add_scalar('accuracy/test', test_accuracy, epoch)

    return model, X_train, X_test, y_train, y_test, r_train, r_test, rt_train, rt_test
