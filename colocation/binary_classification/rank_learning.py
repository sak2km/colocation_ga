"""Rank Learning
"""
import logging

from scipy import stats
import numpy as np
import torch
from torch import nn
from torch.nn import functional
import tensorboardX

from .segment_datafeeder import KETISegmentsDataSet


LOG = logging.getLogger("M.Classifier.RankModel")


class RankModel(nn.Module):
    """Rank Learning
    """

    def __init__(self, seg_length):
        super(RankModel, self).__init__()
        self.transform = nn.Linear(seg_length, 40)
        self.logits = nn.Linear(40, 2)

    def forward(self, x):  # pylint: disable=W0221
        """Forward"""
        a_0 = functional.relu(self.transform(x[:, 0]))
        a_1 = functional.relu(self.transform(x[:, 1]))
        diff = a_1 - a_0
        return torch.sigmoid(self.logits(diff))


class CNNRankModel(nn.Module):
    """Convolutional Neural Network
    """

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool1d(20, stride=10)
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=2)
        self.maxpool2 = nn.MaxPool1d(5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.maxpool3 = nn.MaxPool1d(3)
        self.embeder = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

    def _nn_layer(self, x):
        # LOG.debug("Input Dimension = %s", str(x.shape))
        x = x.view(x.size(0), 1, x.size(1))  # Turn each sample to 2D (pytorch require)
        # x = self.maxpool(x)
        x = self.conv(x)
        # LOG.debug("After conv1 = %s", str(x.shape))
        x = functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv2(x)
        # LOG.debug("After conv2 = %s", str(x.shape))
        x = functional.relu(x)
        x = self.maxpool3(x)
        x = self.embeder(x)
        # LOG.debug("Output Dimension = %s", str(x.shape))
        return x

    def forward(self, x):  # pylint: disable=W0221
        """Forward"""
        a_0 = self._nn_layer(x[:, 0])
        a_1 = self._nn_layer(x[:, 1])

        diff = a_0.view(a_0.size(0), -1) - a_1.view(a_1.size(0), -1)
        distance = diff.norm(dim=1)
        return distance


def _param_l2(model):
    return sum(p.norm() for p in model.parameters())


def train(train_dataloader, model, optimizer, get_loss, cuda, log_interval):
    """Train
    """

    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        if cuda:
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = get_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            LOG.info("batch=%d train_l=%f param_l2=%f", i, loss, _param_l2(model))

        if i >= 10000:
            break


def test(test_loader, model, get_loss, cuda=False, callback=None):
    """Run test
    """
    model.eval()
    loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if cuda:
                x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            # LOG.debug("y_hat = %s", str(y_hat))
            loss += get_loss(y_hat, y).item()
            if callback:
                callback(
                    batch_id=i, batch_size=test_loader.batch_size, y_hat=y_hat, y=y
                )

    loss /= len(test_loader.dataset)

    return loss


def run_training(input_file, step, seg_length, n_epoch, cuda):
    """Run the training
    """
    trainset = KETISegmentsDataSet(
        input_file, seg_length, rooms=list(range(40)), step=step, cuda=cuda
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testset = KETISegmentsDataSet(
        input_file, segment_length=0, rooms=list(range(40, 50)), step=step, cuda=cuda
    )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    model = CNNRankModel().double()
    model = model.cuda() if cuda else model
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    loss_func = _contrastive_loss
    distance_matrix = _DistanceMatrix(testset)

    loss = test(test_loader, model, loss_func, cuda, callback=distance_matrix.add)
    same_stats, diff_stats = distance_matrix.evaluate()
    LOG.info(
        "Pre-Training: test_loss=%.2f, stats(same=%s, diff=%s)",
        loss,
        _short_stats(same_stats),
        _short_stats(diff_stats),
    )

    for epoch in range(n_epoch):
        train(train_loader, model, optimizer, loss_func, cuda, 100)

        LOG.debug("start testing...")
        distance_matrix.matrix.fill(0)
        loss = test(test_loader, model, loss_func, cuda, callback=distance_matrix.add)
        same_stats, diff_stats = distance_matrix.evaluate()
        LOG.info(
            "epoch %d: test_loss=%.2f, stats(same=%s, diff=%s)",
            epoch,
            loss,
            _short_stats(same_stats),
            _short_stats(diff_stats),
        )


def _contrastive_loss(distance, label):
    """Contrastive Loss from https://arxiv.org/ftp/arxiv/papers/1709/1709.08761.pdf
    """
    y = label
    same_room = torch.pow(distance, 2)
    diff_room = torch.pow((1 - distance).clamp(min=0), 2)
    return (0.5 * (y * same_room + (1 - y) * diff_room)).sum()


def _short_stats(s):
    return "[{min:.2f} <-- {mean:.2f} --> {max:.2f}]".format(
        min=s.minmax[0], mean=s.mean, max=s.minmax[1]
    )


class _DistanceMatrix(object):
    def __init__(self, dataset: KETISegmentsDataSet):
        self.sensor_count = dataset.sample_size
        self.matrix = np.zeros((self.sensor_count, self.sensor_count))
        self.dataset = dataset

    def add(self, batch_id, batch_size, y_hat, *_, **__):
        if batch_id * batch_size >= self.dataset.pair_count:
            return
        ids = np.arange(batch_id * batch_size, batch_id * batch_size + batch_size)
        ids = ids[ids < self.dataset.pair_count]
        # LOG.debug("ids = %s", str(ids))
        i, j, _ = self.dataset.disassemble_id(ids)
        distance = y_hat.cpu().numpy()[: len(ids)]
        assert distance.ndim == 1, str(distance)
        self.matrix[i, j] = distance

    def evaluate(self):
        ids = np.arange(self.dataset.pair_count)
        i, j, seg = self.dataset.disassemble_id(ids)
        assert (seg == 0).all()

        same_r = self.dataset._is_same_room(i, j)
        same_stats = stats.describe(self.matrix[i[same_r], j[same_r]])
        diff_stats = stats.describe(self.matrix[i[~same_r], j[~same_r]])

        return same_stats, diff_stats
