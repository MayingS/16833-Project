import numpy as np
import os
import argparse

from modeling.DPF import *
from utils.data_process import *


def args_parse():
    parser = argparse.ArgumentParser('Train the DPF')
    parser.add_argument(
        '--data_dir', help='The directory of the data file.', default='data/100s'
    )
    parser.add_argument(
        '--train_file', help='The filename of the train data', default='nav01_train'
    )
    parser.add_argument(
        '--data_slipt_ratio', help='The ratio to split the data into train and validation set',
        default=0.9
    )

    args = parser.parse_args()
    return args


def main():
    args = args_parse()

    data_dir = args.data_dir
    assert data_dir
    train_file = args.train_file
    train_file = os.path.join(data_dir, train_file + '.npz')
    assert train_file

    # load the data
    sta, obs, act = make_dataset(train_file)
    # split the dataset into train and eval set
    N = sta.shape[0]
    split_ind = int(N*args.data_slipt_ratio)
    train_dataset = DPFDataset(sta[:split_ind], obs[:split_ind], act[:split_ind])
    eval_dataset = DPFDataset(sta[split_ind:], obs[split_ind:], act[split_ind])

    dpf = DPF(train_set=train_dataset, eval_set=eval_dataset)

    pass


if __name__ == '__main__':
    main()
