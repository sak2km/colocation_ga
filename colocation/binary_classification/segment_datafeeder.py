"""Feed data as segments
"""
import numpy as np
import torch
import torch.utils.data

from ..utils import io, functional


class KETISegmentsDataSet(torch.utils.data.Dataset):
    """Data Feeder
    """

    _one_hot_label = {
        # True: torch.DoubleTensor([0.0, 1.0]),
        # False: torch.DoubleTensor([1.0, 0.0]),
        True: 1.0,
        False: 0.0,
    }

    def __init__(
        self, data_file, segment_length=0, type_count=4, rooms=None, step=1, cuda=False
    ):
        """Construct a KETI Segment Dataset
        
        Args:
            data_file (str): the path to the data fiel
            segment_length (int, optional): Defaults to 0. Use full length if zero.
            type_count (int, optional): Defaults to 4. 
            rooms (list, optional): Choose which rooms to load. 
                Use all rooms if ``rooms`` is ``None``.
                Defaults to None.
            step (int, optional): Defaults to 1. Number of datapoints to skip when
                segmenting the time series.
            cuda (bool): Whether to use cuda
        """

        self.raw_data = _select_room(io.read_file(data_file), rooms, type_count)
        self.raw_data = _normalize(self.raw_data)

        self.cuda = cuda
        self.type_count = type_count
        self.step = step
        self.room_count = len(self.raw_data) // type_count
        self.sample_size, self.ts_length = self.raw_data.shape

        self.segment_length = segment_length if segment_length else self.ts_length
        self.segment_count = (
            functional.window_count(self.ts_length, self.segment_length, stride=step)
            if segment_length
            else 1
        )
        self.pair_count = self.segment_count * (
            self.sample_size * (self.sample_size - 1)
        )

        self.data_tensor = torch.DoubleTensor(self.raw_data)
        # if cuda:
        #     self._one_hot_label[True] = self._one_hot_label[True].cuda()
        #     self._one_hot_label[False] = self._one_hot_label[False].cuda()
        #     self.data_tensor = self.data_tensor.cuda()

    def disassemble_id(self, idx):
        """Disassemble an id to the corresponding sensor ids and segment ids
        """
        pair_id, seg_id = np.divmod(idx, self.segment_count)
        first_id, second_id = np.divmod(pair_id, self.sample_size - 1)

        if isinstance(second_id, np.ndarray):
            second_id[second_id >= first_id] += 1
        elif second_id >= first_id:
            second_id += 1

        return first_id, second_id, seg_id

    def __len__(self):
        return self.pair_count * 2

    def _get_segment(self, sensor_id, seg_id):
        start_id = seg_id * self.step
        return self.data_tensor[sensor_id, start_id : start_id + self.segment_length]

    def _get_same_room_pair(self, idx):
        r_id = idx % self.room_count
        seg_id = idx % self.segment_count
        t_id1 = idx % self.type_count
        t_id2 = (t_id1 + 1) % self.type_count
        return (
            self._get_segment(r_id * self.type_count + t_id1, seg_id),
            self._get_segment(r_id * self.type_count + t_id2, seg_id),
            self._one_hot_label[True],
        )

    def _get_pair_normal(self, idx):
        first_id, second_id, seg_id = self.disassemble_id(idx)

        return (
            self._get_segment(first_id, seg_id),
            self._get_segment(second_id, seg_id),
            self._one_hot_label[self._is_same_room(first_id, second_id)],
        )

    def __getitem__(self, idx):
        seg1, seg2, label = (
            self._get_same_room_pair(idx - self.pair_count)
            if idx >= self.pair_count
            else self._get_pair_normal(idx)
        )

        pair = torch.stack((seg1, seg2))

        return pair, label  # pylint: disable=no-member

    def _is_same_room(self, first_id, second_id):
        return first_id // self.type_count == second_id // self.type_count


def _select_room(data, rooms, type_count):
    if rooms is None:
        return data
    else:
        ids = (
            np.array([[r * type_count + t for t in range(type_count)] for r in rooms])
            .astype(int)
            .reshape(-1)
        )
        return data[ids]


def _normalize(data):
    for row in data:
        row[:] = (row - np.mean(row)) / np.linalg.norm(row)
    return data
