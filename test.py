import argparse
import torch
import torch.utils.data

from utils.data_process import *
from utils.visualize import *
from modeling.DPF import *
import modeling.measurement_update as measurement


def args_parse():
    parser = argparse.ArgumentParser('To test the models using test data')
    parser.add_argument(
        '--observation_encoder', help='the path of the trained observation encoder',
        default='likelihood_estimator_checkpoint/encoder_checkpoint_570.pth'
    )
    parser.add_argument(
        '--likelihood_estimator', help='the path of the trained likelihood estimator',
        default='likelihood_estimator_checkpoint/estimator_checkpoint_570.pth'
    )
    parser.add_argument(
        '--data_dir', help='The directory of the data file.', default='data/100s'
    )
    parser.add_argument(
        '--train_file', help='The filename of the train data', default='nav01_train'
    )
    parser.add_argument(
        '--test_file', help='The filename of the test data', default='nav01_test'
    )
    parser.add_argument(
        '--visualize', help='To visualize the model output', default=False
    )
    parser.add_argument(
        '--vis_outdir', help='The output directory to store the visualization result', default='test_vis'
    )
    args = parser.parse_args()

    return args


def main():
    """ Main Function """
    args = args_parse()

    assert args.observation_encoder
    assert args.likelihood_estimator

    data_dir = args.data_dir
    assert data_dir
    test_file = args.test_file
    test_file = os.path.join(data_dir, test_file + '.npz')
    assert test_file
    train_file = args.train_file
    train_file = os.path.join(data_dir, train_file + '.npz')
    assert train_file

    # load the train data to get the mean and std
    train_sta, train_obs, train_act = make_dataset(train_file)

    obs_mean = np.mean(train_obs, axis=0)
    obs_std = np.std(train_obs - obs_mean, axis=tuple(range(len(train_obs.shape) - 1)))

    sta_mean = np.mean(train_sta, axis=0)
    sta_mean[2] = 0
    sta_std = np.std(train_sta - sta_mean, axis=tuple(range(len(train_sta.shape) - 1)))

    act_mean = np.mean(train_act, axis=0)
    act_mean[2] = 0
    act_std = np.std(train_act - act_mean, axis=tuple(range(len(train_act.shape) - 1)))

    means = {'o': obs_mean, 'a': act_mean, 's': sta_mean}
    stds = {'o': obs_std, 'a': act_std, 's': sta_std}

    # create the whole model instance
    dpf = DPF(means=means, stds=stds)

    # load the test data and create the dataset
    sta, obs, act = make_dataset(test_file)
    obs = (obs - obs_mean) / obs_std
    test_dataset = DPFDataset(sta, obs, act)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    observation_encoder_path = args.observation_encoder
    likelihood_estimator_path = args.likelihood_estimator

    # load trained model
    observation_encoder = measurement.ObservationEncoder()
    likelihood_estimator = measurement.ObservationLikelihoodEstimator()
    observation_encoder.load_state_dict(torch.load(observation_encoder_path))
    likelihood_estimator.load_state_dict(torch.load(likelihood_estimator_path))

    dpf.observation_encoder = observation_encoder.double()
    dpf.likelihood_estimator = likelihood_estimator.double()

    vis_outdir = args.vis_outdir
    if not os.path.exists(vis_outdir):
        os.path.exists(vis_outdir)

    test(dpf, test_loader, args.visualize, args.vis_outdir)


def test(dpf, test_loader, vis=False, vis_outdir=None):
    """ Test the model given the test data

    Args:
      dpf: an instance of DPF class
      test_loader: DataLoader for test data
      vis: True or False, indicating whether to visualize model or not
      vis_outdir: the output directory of the visualization result
    Returns:
    """
    for batch_id, (s, o, a) in enumerate(test_loader):
        dpf.observation_encoder.eval()
        dpf.likelihood_estimator.eval()
        if dpf.use_cuda:
            s = s.cuda()
            o = o.cuda()

        # test the measurement
        w = dpf.get_likelihood(s, o)
        w = w.data.cpu().numpy()
        # visualize the output of the measurement model
        if vis:
            # visualize output of measurement
            meas_vis_dir = os.path.join(vis_outdir, 'measurement')
            if not os.path.exists(meas_vis_dir):
                os.makedirs(meas_vis_dir)
            for i in range(w.shape[0]):
                plot_measurement(w[i], save_image=True, outdir=meas_vis_dir, batch=batch_id, ind=i)


if __name__ == '__main__':
    main()
