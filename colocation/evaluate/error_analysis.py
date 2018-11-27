"""Perform error analysis on room assignment
"""

import enum
import numpy as np
from ..utils import io
from ..genetic_algorithm.core.calcs import calculate_accuracy


class RPattern(enum.IntEnum):
    """Pattern types
    """

    CORRECT = 0
    ONE_ERROR = 1
    OTHERS = 2


def analyze_error(assignment: np.ndarray, room_count: int, type_count: int):
    """Perform error analysis on room assignment
    """
    assert assignment.ndim == 2
    assert isinstance(assignment.shape, tuple)
    assert isinstance(room_count, int)
    assert isinstance(type_count, int)
    assert assignment.shape == (room_count, type_count)

    accuracy = calculate_accuracy(assignment)
    io.vput("Accuracy = {}", accuracy, verbose=True)

    # By Pattern
    patterns, one_err_ids = _pattern(assignment)
    for p_type in RPattern:
        count = np.count_nonzero(patterns == int(p_type))
        io.vput("| {!s}\t{}", p_type, count, verbose=True)

    # Switch one
    one_ids, rest_ids = one_err_ids[patterns == RPattern.ONE_ERROR].T
    switch_count = 0
    for i, j in zip(one_ids, rest_ids):
        switch_count += 1 if rest_ids[one_ids == j] == i else 0

    io.vput("| switch one: {}", switch_count, verbose=True)


def _pattern(room: np.ndarray):
    patterns = np.empty((room.shape[0]), dtype=int)
    one_err_ids = np.empty((room.shape[0], 2), dtype=int)

    for i, r in enumerate(room):
        room_ids = r // len(r)
        uniques, counts = np.unique(np.sort(room_ids), return_counts=True)
        patterns[i] = (
            RPattern.CORRECT
            if len(uniques) == 1
            else RPattern.ONE_ERROR
            if len(uniques) == 2 and (counts == 1).any()
            else RPattern.OTHERS
        )

        one_err_ids[i] = (
            (0, 0)
            if patterns[i] != RPattern.ONE_ERROR
            else uniques
            if counts[0] == 1
            else uniques[::-1]
        )

    return patterns, one_err_ids
