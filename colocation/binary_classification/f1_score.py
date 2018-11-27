"""Util to calculate F1 score
"""

import numpy as np


def _f1_score(precision, recall):
    return 2 * _safe_div(precision * recall, precision + recall)


def _safe_div(a, b):
    return a / b if b != 0 else 0


class F1Score(object):
    def __init__(self):
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

    def add(self, truth, prediction):
        true_positive = np.sum(truth & prediction)
        self.false_positive += np.sum(prediction) - true_positive
        self.false_negative += np.sum(truth) - true_positive
        self.true_negative += np.sum(truth == prediction) - true_positive
        self.true_positive += true_positive

    def add_tensor(self, truth, prediction):
        true_positive = (truth & prediction).sum().item()
        self.false_positive += prediction.sum().item() - true_positive
        self.true_negative += truth.sum().item() - true_positive
        self.true_negative += truth.eq(prediction).sum().item() - true_positive
        self.true_positive += true_positive

    @property
    def count(self):
        return (
            self.false_positive
            + self.false_negative
            + self.true_negative
            + self.true_positive
        )

    @property
    def recall(self):
        return _safe_div(self.true_positive, self.true_positive + self.false_negative)

    @property
    def precision(self):
        return _safe_div(self.true_positive, self.true_positive + self.false_positive)

    @property
    def f1(self):
        return _f1_score(self.precision, self.recall)

    @property
    def accuracy(self):
        return _safe_div(self.true_positive + self.true_negative, self.count)

    def __str__(self):
        if self.count == 0:
            return "F1Score(Empty)"
        return f"F1Score(f1={self.f1:.2f:.2f}, recall={self.recall:.2f}, precision={self.precision:.2f}, accuracy={self.accuracy:.2f}"

