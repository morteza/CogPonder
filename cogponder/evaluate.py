import torch
from torch import nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.auto import tqdm

from .loss import ReconstructionLoss, RegularizationLoss


def evaluate(
    model,
    dataset,
    optimizer,
    n_epochs=1000,
    logs=SummaryWriter()
):

    # split params
    train_size = int(len(dataset) * .8)
    test_size = len(dataset) - train_size

    train_subset, test_subset = random_split(dataset, lengths=(train_size, test_size))

    X_train, r_train, y_train, rt_train = dataset[train_subset.indices]
    X_test, r_test, y_test, rt_test = dataset[test_subset.indices]

    loss_rec_fn = ReconstructionLoss(nn.BCELoss(reduction='mean'))
    loss_reg_fn = RegularizationLoss(lambda_p=.5, max_steps=20)
    loss_beta = .2

    for epoch in tqdm(range(n_epochs), desc='Epochs'):

        model.train()
        y_steps, p_halt, halt_step = model(X_train)

        loss_rec = loss_rec_fn(p_halt, y_steps, y_train)
        loss_reg = loss_reg_fn(p_halt, r_train, rt_train)
        loss = loss_rec + loss_beta * loss_reg

        logs.add_scalar('loss/rec_train', loss_rec, epoch)
        logs.add_scalar('loss/reg_train', loss_reg, epoch)
        logs.add_scalar('loss/train', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = y_steps.detach()[0, halt_step].argmax(dim=1)
        train_accuracy = accuracy_score(y_pred, y_train)

        logs.add_scalar('accuracy/train', train_accuracy, epoch)  

        model.eval()
        with torch.no_grad():
            y_steps, p_halt, halt_step = model(X_test)
            loss_rec = loss_rec_fn(p_halt, y_steps, y_test)
            loss_reg = loss_reg_fn(p_halt, r_test, rt_test)
            loss = loss_rec + loss_beta * loss_reg
            logs.add_scalar('loss/test', loss.detach(), epoch)

            y_pred = y_steps.detach()[0, halt_step].argmax(dim=1)
            test_accuracy = accuracy_score(y_pred, y_test)

            logs.add_scalar('accuracy/test', test_accuracy, epoch)
