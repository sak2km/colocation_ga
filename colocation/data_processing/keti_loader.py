"""Data Loaders
"""

import pathlib

import progressbar

from ..utils import io


def room_ids():
    return [
        "413",
        "415",
        "417",
        "419",
        "421",
        "422",
        "423",
        "424",
        "442",
        "446",
        "448",
        "452",
        "454",
        "456",
        "458",
        "462",
        "510",
        "513",
        "552",
        "554",
        "556",
        "558",
        "562",
        "564",
        "621",
        "621A",
        "621C",
        "621D",
        "621E",
        "640",
        "644",
        "648",
        "656A",
        "656B",
        "664",
        "666",
        "668",
        "717",
        "719",
        "721",
        "722",
        "723",
        "724",
        "726",
        "734",
        "746",
        "748",
        "752",
        "754",
        "776",
    ]


def csv_path_generator(room_paths, show_progress=False):
    """Emits csv file path one by one

    Args:
        room_paths (List[Path]): list of room dirs

    Yields: file path of each csv file
    """

    valid_sensors = ["co2", "humidity", "light", "temperature"]
    bar = (
        progressbar.ProgressBar(max_value=(len(valid_sensors) * len(room_paths)))
        if show_progress
        else None
    )

    i = 0
    for room_dir in room_paths:
        for sensor in valid_sensors:
            yield room_dir.joinpath(sensor + ".csv")
            if bar is not None:
                bar.update(i)
                i += 1


def load(path, show_progress=False):
    """Load KETI one week raw data

    Args:
        path (str): the path to the directory containing raw data

    Raises:
        FileNotFoundError: If the directory is not a valid directory

    Returns:
        List[np.ndarray]: a list of numpy arrays
    """

    path = pathlib.Path(path)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError("{} is not a valid directory.".format(path))

    room_paths = sorted(path.iterdir())

    data_list = [
        io.read_csv(csv_path)
        for csv_path in csv_path_generator(room_paths, show_progress=show_progress)
    ]

    return data_list
