import numpy as np
import pdb

# with np.load('raw_abs.npz') as data:
# 	print(data)

# data = np.load('raw_abs.npz')["arr_0"]
# print(data)
pdb.set_trace()
txt_file = "score_list.txt'"

raw_pt_a = [i[1:].strip(']').split(',') for i in open(txt_file).readlines()]

pdb.set_trace()

# print(data[0])

	# pdb.set_trace()
    # a = data['a']


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