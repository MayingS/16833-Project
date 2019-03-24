import numpy as np
import os
import modeling.DPF as DPF
import argparse


def args_parse():
    parser = argparse.ArgumentParser('Train the DPF')
    parser.add_argument(
        '--data_dir', help='The directory of the data file.', default='data/100s'
    )
    parser.add_argument(
        '--train_file', help='The filename of the train data', default='nav01_train'
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

    pass


if __name__ == '__main__':
    main()
