"""Tests for rank learning
"""

import pathlib
import pytest

WORKSPACE_DIR = pathlib.Path("C:/repo/co-location")


def _is_near(a, b, threshold):
    return abs(a - b) < threshold


def test_class_balancer():
    import torch
    from ..binary_classification.segment_datafeeder import KETISegmentsDataSet

    dataset = KETISegmentsDataSet("data/time_series/resids.npz", 200, step=10000)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

    total_count = 0
    positive_count = 0

    for _, y in train_loader:
        total_count += 1
        positive_count += y.max(1, keepdim=True)[1].item()

    assert _is_near(total_count / positive_count, 2, 0.2)
