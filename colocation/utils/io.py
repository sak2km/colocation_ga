"""Matrix Loaders
"""
import pathlib

import numpy as np
import pandas as pd
import scipy.io

import rapidjson as json


def read_csv(input_filename):
    """Read a csv file and return a numpy array
    """
    return pd.read_csv(input_filename, header=None).values


def save_csv(data: np.ndarray, out_filename):
    """Save a numpy array as a csv
    """
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(out_filename, index=False, header=False)


def read_matlab(file):
    """Load matlab matrix file
    """
    return np.fabs(scipy.io.loadmat(file)["corr"])  # pylint: disable=no-member


def read_npy(file):
    """read numpy dump file
    """
    return np.load(file)


def save_npy(data, path):
    """save numpy dump file
    """
    with touch(path).open("wb") as path:
        np.array(data).dump(path)


def save_npz(data, path):
    """Save data as compressed numpy array
    """
    with touch(path).open("wb") as path:
        np.savez_compressed(path, data)


def read_npz(path):
    """Read the npz format
    """
    data = np.load(path)["arr_0"]
    return data


def save_json(dict_file, path):
    """Save a dict file as json
    """
    with touch(path).open("w") as file:
        json.dump(dict_file, file, indent=4)

def read_json(path):
    """Read json
    """
    path = pathlib.Path(path)
    with path.open("rb") as file:
        return json.load(file)


def touch(path_name):
    """Make sure a file exists
    """
    path = pathlib.Path(path_name)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.touch()
    return path


def make_dir(path_name):
    """Make sure that a dir exist
    """
    path = pathlib.Path(path_name)
    path.mkdir(exist_ok=True, parents=True)
    return path


def vput(fmt: str, *args, verbose=False):
    """verbose print
    """
    if verbose:
        print(fmt.format(*args))


def save_txt(string: str, path):
    """Save text as is
    """
    with touch(path).open("w") as file:
        file.write(string)


_READ_FUNC = {".npz": read_npz, ".npy": read_npy, ".csv": read_csv, ".mat": read_matlab}


def read_file(path: str):
    """Read file in a generic manner
    """
    file_type = pathlib.Path(path).suffix
    return _READ_FUNC[file_type](path)


def save_variables(path: str, iteration, **vars):
    """Save a bunch of variables
    """
    output_path = make_dir(path)
    for v_name, v_val in vars.items():
        save_npz(v_val, output_path.joinpath(f"{iteration}_{v_name}.npz"))
