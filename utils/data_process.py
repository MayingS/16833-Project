import numpy as np
import os
import torch.utils.data as data


class DPFDataset(data.Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.dataset = make_dataset(filename)


def make_dataset(filename):
    data = dict(np.load(filename))
    pass


if __name__ == '__main__':
    make_dataset('../data/100s/nav01_train.npz')

