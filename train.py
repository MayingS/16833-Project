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

    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs - obs_mean, axis=tuple(range(len(obs.shape) - 1)))
    obs = (obs - obs_mean) / obs_std

    sta_mean = np.mean(sta, axis=0)
    sta_mean[2] = 0
    sta_std = np.std(sta - sta_mean, axis=tuple(range(len(sta.shape) - 1)))

    act_mean = np.mean(act, axis=0)
    act_mean[2] = 0
    act_std = np.std(act - act_mean, axis=tuple(range(len(act.shape) - 1)))

    means = {'o': obs_mean, 'a': act_mean, 's': sta_mean}
    stds = {'o': obs_std, 'a': act_std, 's': sta_std}

    # split the dataset into train and eval set
    N = sta.shape[0]
    split_ind = int(N*args.data_slipt_ratio)
    train_dataset = DPFDataset(sta[:split_ind], obs[:split_ind], act[:split_ind])
    eval_dataset = DPFDataset(sta[split_ind:], obs[split_ind:], act[split_ind])

    dpf = DPF(train_set=train_dataset, eval_set=eval_dataset, means=means, stds=stds)
    # test train_likelihood_estimator
    # dpf.train_likelihood_estimator()


if __name__ == '__main__':
    main()
