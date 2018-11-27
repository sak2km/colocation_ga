from pathlib import Path
import pandas as pd
import numpy as np


def concatenate_data(root_dir):
    root_dir = Path(root_dir)
    ahu_dir: Path = root_dir.joinpath("AHU")
    vav_dir: Path = root_dir.joinpath("VAV")

    ahu_files = list(ahu_dir.iterdir())
    vav_files = list(vav_dir.iterdir())

    n_ahu = len(ahu_files)

    a_data = np.array(
        [pd.read_csv(str(a_file), header=None).values[:, 1] for a_file in ahu_files]
    )
    a_ids = np.array([])

    def parse_vav_name(name):
        a_id, v_id = name.stem.lstrip("VAV ").split("-")
        a_id, v_id = int(a_id), int(v_id)
        return a_id, v_id

    v_data = np.empty((len(vav_files), a_data.shape[1]), dtype=float)
    v_counts = {}

    for i, v_file in enumerate(vav_files):
        a_id, v_id = parse_vav_name(v_file)
        v_data[i] = pd.read_csv(str(v_file), header=None).values[:, 1]
        if a_id not in v_counts:
            v_counts[a_id] = 0
        v_counts[a_id] += 1

    data = np.concatenate((a_data, v_data), axis=0)

    v_counts = np.array(list(v_counts.values()), dtype=int)
    return data, v_counts


def read_dezhi_csv(data, n_ahu):
    """Read CSV from Dezhi's Events
    """
    ahu_data = data[:n_ahu, :-1]
    ahu_labels = data[:n_ahu, -1]
    ahu_data = ahu_data[np.argsort(ahu_labels)]

    vav_data = data[n_ahu:, :-1]
    vav_labels = data[n_ahu:, -1]
    sorted_id = np.argsort(vav_labels)
    vav_data = vav_data[sorted_id]
    vav_labels = vav_labels[sorted_id]

    _, count = np.lib.arraysetops.unique(vav_labels, return_counts=True)

    data = np.concatenate([ahu_data, vav_data])

    return data, count
