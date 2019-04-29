import numpy as np
import os
import argparse

from modeling.DPF import *
from utils.data_process import *
import torch.backends.cudnn as cudnn


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
    parser.add_argument(
        '--visualize', help='To visualize the model output', default=False
    )
    parser.add_argument(
        '--dynamics_model_path', help='Path to trained dynamics model',
        default='model/motion_model/mode_1/dynamic_model.pth'
    )

    args = parser.parse_args()
    return args


def main():
    rand_seed = 1234
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    cudnn.deterministic = True

    args = args_parse()

    data_dir = args.data_dir
    assert data_dir
    train_file = args.train_file
    train_file = os.path.join(data_dir, train_file + '.npz')
    assert train_file
    vis = args.visualize

    # load the data
    sta, obs, act = make_dataset(train_file)

    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs - obs_mean, axis=tuple(range(len(obs.shape) - 1)))
    obs = (obs - obs_mean) / obs_std

    sta_mean = np.mean(sta, axis=0)
    sta_mean[2] = 0
    sta_std = np.std(sta - sta_mean, axis=tuple(range(len(sta.shape) - 1)))
    sta_min = np.min(sta, axis=0)
    sta_max = np.max(sta, axis=0)

    act_mean = np.mean(act, axis=0)
    act_mean[2] = 0
    act_std = np.std(act - act_mean, axis=tuple(range(len(act.shape) - 1)))

    means = {'o': obs_mean, 'a': act_mean, 's': sta_mean}
    stds = {'o': obs_std, 'a': act_std, 's': sta_std}
    
    state_step_sizes = []
    for i in range(3):
        steps = sta[1:, i] - sta[:-1, i]
        if i == 2:
            steps = wrap_angle(steps)
        state_step_sizes.append(np.mean(abs(steps)))
    state_step_sizes[0] = state_step_sizes[1] = (state_step_sizes[0] + state_step_sizes[1]) / 2
    state_step_sizes = np.array(state_step_sizes)

    # split the dataset into train and eval set
    N = sta.shape[0]
    split_ind = int(N*args.data_slipt_ratio)
    train_dataset = DPFDataset(sta[:split_ind], obs[:split_ind], act[:split_ind])
    eval_dataset = DPFDataset(sta[split_ind:], obs[split_ind:], act[split_ind:])

    dpf = DPF(train_set=train_dataset, eval_set=eval_dataset, means=means, stds=stds, visualize=vis,
              state_step_sizes_=state_step_sizes, state_min=sta_min, state_max=sta_max)
    # # test train_likelihood_estimator
    dpf.train_likelihood_estimator()
    # # test train_motion_model
    # dpf.train_motion_model(mode=0)
    # dpf.train_motion_model(mode=1, phrase=0)
    dpf.train_motion_model(mode=1, phrase=1, dynamics_model_path=args.dynamics_model_path)
    # dpf.train_particle_proposer()


if __name__ == '__main__':
    main()
