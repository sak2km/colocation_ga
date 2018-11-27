"""Perform logistic regression
"""
import click
import torch
from torch import nn
from torch.nn import functional

from . import segment_datafeeder
from ..utils import cli, io


class LRModel(nn.Module):
    """Logistic regression
    """

    def __init__(self, segment_length):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(segment_length * segment_length, 2)

    # pylint: disable=W0221
    def forward(self, x):
        """forward
        """
        return functional.sigmoid(self.linear(x))


def train(train_dataloader, model, optimizer, log_interval, callback=None):
    """train a logistic model
    """
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = functional.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        if callback and i % log_interval == 0:
            callback(i, loss.item())


def test(test_loader, model):
    """Run test
    """
    model.eval()
    loss = 0

    positive_positive = 0
    positive_counts = 0
    negative_negative = 0
    negative_counts = 0

    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x)
            loss += functional.mse_loss(y_hat, y).item()
            pred = y_hat.max(1, keepdim=True)[1]
            label = y.max(1, keepdim=True)[1]

            # Metrics
            corrects = pred == label
            positive = label == 1
            negative = label == 0
            positive_positive += (corrects * positive).sum().item()
            positive_counts += positive.sum().item()
            negative_negative += (corrects * negative).sum().item()
            negative_counts += negative.sum().item()

    loss /= len(test_loader.dataset)
    positive_rate = positive_positive / positive_counts
    negative_rate = negative_negative / negative_counts
    return loss, positive_rate, negative_rate


@click.command("LR-outer")
@cli.option_input_file()
@click.option("--segment-length", default=40)
@click.option("--n-epoch", default=1)
@cli.option_verbose()
def lr_outer(input_file, segment_length, n_epoch, verbose):
    """Run Logistic Regression on the outer product of segments
    """
    trainset = segment_datafeeder.KETISegmentsDataSet(
        input_file, segment_length, rooms=list(range(1))
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    test_dataset = segment_datafeeder.KETISegmentsDataSet(
        input_file, segment_length, rooms=list(range(1, 2))
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    model = LRModel(segment_length).double()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epoch):
        train(
            train_loader,
            model,
            optimizer,
            5000,
            lambda i, loss: print(f"E[{epoch}] {i}:\t{loss}"),
        )
        io.vput("start testing...", verbose=verbose)
        loss, same_room_accuracy, diff_room_accuracy = test(test_loader, model)

        io.vput(
            "EPOCH [{}]: loss = {} | accuracies = [ {}, {} ]",
            epoch,
            loss,
            same_room_accuracy,
            diff_room_accuracy,
            verbose=verbose,
        )
