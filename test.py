import argparse
import torch
import torch.utils.data

from utils.data_process import *
from utils.visualize import *
from modeling.DPF import *
import modeling.measurement_update as measurement
import modeling.motion_model.motion_model as motion
import config.set_parameters as sp


def args_parse():
    parser = argparse.ArgumentParser('To test the models using test data')
    parser.add_argument(
        '--observation_encoder', help='the path of the trained observation encoder',
        default='models/encoder_checkpoint_620.pth'
    )
    parser.add_argument(
        '--likelihood_estimator', help='the path of the trained likelihood estimator',
        default='models/estimator_checkpoint_620.pth'
    )
    parser.add_argument(
        '--motion_model', help='the path of the trained motion_model',
        default='models/motion_model.pth'
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
    sta_min = np.min(train_sta, axis=0)
    sta_max = np.max(train_sta, axis=0)

    act_mean = np.mean(train_act, axis=0)
    act_mean[2] = 0
    act_std = np.std(train_act - act_mean, axis=tuple(range(len(train_act.shape) - 1)))

    means = {'o': obs_mean, 'a': act_mean, 's': sta_mean}
    stds = {'o': obs_std, 'a': act_std, 's': sta_std}
    
    state_step_sizes = []
    for i in range(3):
        steps = train_sta[1:, i] - train_sta[:-1, i]
        if i == 2:
            steps = wrap_angle(steps)
        state_step_sizes.append(np.mean(abs(steps)))
    state_step_sizes[0] = state_step_sizes[1] = (state_step_sizes[0] + state_step_sizes[1]) / 2
    state_step_sizes = np.array(state_step_sizes)
    
    # create the validation dataset
    N = train_sta.shape[0]
    split_ind = int(N*0.9)
    eval_dataset = DPFDataset(train_sta[split_ind:], train_obs[split_ind:], train_act[split_ind:])
    val_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    # create the whole model instance
    dpf = DPF(eval_set=eval_dataset, means=means, stds=stds, visualize=args.visualize,
              state_step_sizes_=state_step_sizes, state_min=sta_min, state_max=sta_max)

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
    motion_model_path = args.motion_model

    # load trained model
    observation_encoder = measurement.ObservationEncoder()
    likelihood_estimator = measurement.ObservationLikelihoodEstimator()
    motion_model = motion.MotionModel()
    observation_encoder.load_state_dict(torch.load(observation_encoder_path))
    likelihood_estimator.load_state_dict(torch.load(likelihood_estimator_path))
    motion_model.load_state_dict(torch.load(motion_model_path))

    dpf.observation_encoder = observation_encoder.double()
    dpf.likelihood_estimator = likelihood_estimator.double()
    dpf.motion_model = motion_model

    vis_outdir = args.vis_outdir
    if not os.path.exists(vis_outdir):
        os.makedirs(vis_outdir)

    # test_measurement(dpf, val_loader, args.visualize, args.vis_outdir)
    # test_motion_model(val_loader, motion_model, args.vis_outdir, stds, means, state_step_sizes)
    test_connect_model(dpf, test_loader, args.visualize, args.vis_outdir)


def test_measurement(dpf, test_loader, vis=False, vis_outdir=None):
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


def test_motion_model(val_loader, motion_model, vis_outdir, stds, means, state_step_sizes):
    """Test the motion model
    """
    particle_num = sp.Params().train['particle_num']
    mode = sp.Params().train['train_motion_model_mode']
    for batch_idx, (sta, obs, act) in enumerate(val_loader):
        # Shape: (batch_size, seq_len, 1, 3)
        act = act.unsqueeze(2)
        sta = sta.unsqueeze(2)
        # Shape: (batch_size, seq_len, num_particle, 3)
        actions = act.repeat(1, 1, particle_num, 1).float()
        states = sta.repeat(1, 1, particle_num, 1).float()
        # Shape: (batch_size*(seq_len-1), num_particle, 3)
        actions = actions[:, 1:, :, :].contiguous().view(-1, particle_num, act.size(3))
        particles = states[:, :-1, :, :].contiguous().view(-1, particle_num, sta.size(3))
        states = states[:, 1:, :, :].contiguous().view(-1, particle_num, sta.size(3))

        moved_particles = motion_model(actions,
                                       particles,
                                       states,
                                       stds,
                                       means,
                                       state_step_sizes,
                                       mode)

        particles = particles.cpu().detach().numpy()
        states = states.cpu().detach().numpy()
        moved_particles = moved_particles.cpu().detach().numpy()
        
        plot_motion_model('nav01', vis_outdir, particles, states, moved_particles)

        break


def test_connect_model(dpf, test_loader, vis=False, vis_outdir=None):
    """ Test the whole system

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
        dpf.motion_model.eval()

        particle_num = sp.Params().test['particle_num']
        if dpf.use_cuda:
            s = s.cuda()
            o = o.cuda()
            a = a.cuda()

        particle_list, particle_probs_list, pred_state = dpf.connect_modules(particle_num,
                                                                             s, o, a,
                                                                             motion_mode=0,
                                                                             phrase=None)
        plot_particle_filter('nav01', particle_list, particle_probs_list, pred_state, s,
                             save_image=vis, outdir=vis_outdir, batch=batch_id)


if __name__ == '__main__':
    main()
