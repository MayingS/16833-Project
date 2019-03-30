import numpy as np
import os
import torch.utils.data as data

import config.set_parameters as sp


class DPFDataset(data.Dataset):
    def __init__(self, filename):
        self.filename = filename
        states, observations, actions = make_dataset(filename)
        self.dataset = [(sta, obs, act) for sta, obs, act in zip(states, observations, actions)]

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def wrap_angle(angle):
    """ Warp the angle to the range of -pi to pi

    Args:
      angle: angle value in radian
    Returns:
      angle value in range [-pi, pi)
    """
    return ((angle - np.pi) % (2 * np.pi)) - np.pi


def make_dataset(filename):
    """ Extract the states, observations, actions from the dataset

    Args:
      filename: the filename of the dataset
    Returns:
      states: numpy array in size (N, T-1, 3)
      observations: numpy array in size (N, T-1, H, W, 3)
      actions: numpy array array in size (N, T-1, 3)
    """
    params = sp.Params()
    seq_length = params.train['seq_length']
    data = dict(np.load(filename))
    for key in data.keys():
        # print(key, data[key].shape) -> vel: (100000, 3), pose: (100000, 3), rgbd: (100000, 32, 32, 4)
        data[key] = np.reshape(data[key], [-1, seq_length] + list(data[key].shape[1:])).astype('float32')
        # print(data[key].shape) -> vel: (5000, 20, 3), pose: (5000, 20, 3), rgbd: (5000, 20, 32, 32, 4)

    # convert degrees into radian
    for key in ['pose', 'vel']:
        data[key][:, :, 2] *= np.pi / 180
    # warp angles to range of -pi to pi
    data['pose'][:, :, 2] = wrap_angle(data['pose'][:, :, 2])

    # calculate the dist on x and y directions between s_t and s_{t+1}
    d_x = (data['pose'][:, 1:, 0] - data['pose'][:, :-1, 0])
    d_y = (data['pose'][:, 1:, 1] - data['pose'][:, :-1, 1])
    d_theta = wrap_angle(data['pose'][:, 1:, 2] - data['pose'][:, :-1, 2])
    s = np.sin(data['pose'][:, :-1, 2])
    c = np.cos(data['pose'][:, :-1, 2])
    # calculate the movement on particle's local coordinates
    rel_d_x = s * d_x - c * d_y
    rel_d_y = c * d_x + s * d_y

    rel_d_x = np.expand_dims(rel_d_x, axis=2)
    rel_d_y = np.expand_dims(rel_d_y, axis=2)
    d_theta = np.expand_dims(d_theta, axis=2)

    states = data['pose'][:, 1:, :]
    observations = data['rgbd'][:, 1:, :, :, :3]
    actions = np.concatenate([rel_d_x, rel_d_y, d_theta], axis=-1)

    return states, observations, actions


if __name__ == '__main__':
    D = DPFDataset('data/100s/nav01_train.npz')
