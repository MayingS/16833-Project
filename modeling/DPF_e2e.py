import numpy as np
import os
import modeling.motion_model.motion_model as motion
import shutil
import math
import modeling.measurement_update as measurement
import modeling.resampling as resample
import config.set_parameters as sp
import utils.data_process as data_process
from utils.visualize import *

import torch
import torch.utils
import torch.utils.data
import torch

from tensorboardX import SummaryWriter


class DPF_e2e:
    def __init__(self, train_set=None, eval_set=None, means=None, stds=None, visualize=False,
                 state_step_sizes_=None, state_min=None, state_max=None):
        self.train_set = train_set
        self.eval_set = eval_set
        self.means = means
        self.stds = stds
        self.state_step_sizes_ = state_step_sizes_
        self.state_min = state_min
        self.state_max = state_max
        self.visualize = visualize

        self.motion_model = motion.MotionModel()
        self.observation_encoder = measurement.ObservationEncoder()
        self.particle_proposer = measurement.ParticleProposer()
        self.likelihood_estimator = measurement.ObservationLikelihoodEstimator()
        self.resampling = resample.particle_resampling

        params = sp.Params()
        self.globalparam = params.globalparam
        self.trainparam = params.train
        self.testparam = params.test

        self.end2end = False
        # self.use_cuda = torch.cuda.is_available()
        self.use_cuda = False

        self.log_freq = 10  # Steps
        self.test_freq = 2  # Epoch

    def particles_to_state(self, particle_list, particle_probs_list):
        """ Get predicted state from the particles

        Args:
          particle_list: Tensor with size (N, T, particle_num, 3), containing the particles at different time step
          particle_probs_list: Tensor with size (N, T, particle_num), corresponds to the particle probabilities
        Returns:
          Tensor with size (N, T, 4), each state is 4-dim with (x, y, cos(theta), sin(theta))
        """
        particle_probs_list = particle_probs_list.view(particle_probs_list.size(0), particle_probs_list.size(1),
                                                       particle_probs_list.size(2), 1)
        mean_position = torch.sum(particle_probs_list.repeat(1, 1, 1, 2)
                                  * particle_list[:, :, :, :2], 2)
        mean_orientation = torch.atan2(
            torch.sum(particle_probs_list * torch.cos(particle_list[:, :, :, 2:]), 2),
            torch.sum(particle_probs_list * torch.sin(particle_list[:, :, :, 2:]), 2))
        return torch.cat([mean_position, mean_orientation], 2)

    def connect_modules(self, particle_num, sta, obs, act, motion_mode=0, phrase=None):
        """ Connect all the modules together to form the whole DPF system
        Args:
          sta: Tensor with size (N, T, 3), states
          obs: Tensor with size (N, T, 3, H, W), observations
          act: Tensor with size (N, T, 3), actions
        Returns:
          particle_list: Tensor with size (N, T, particle_num, 3), particles at different time step
          particle_probs_list: Tensor with size (N, T, particle_num),
                               the particle probabilities at different time step
        """

        for t in range(1, sta.size(1)):
            act_ = act.unsqueeze(2)
            sta_ = sta.unsqueeze(2)
            act_ = act_.repeat(1, 1, particle_num, 1)
            sta_ = sta_.repeat(1, 1, particle_num, 1)

            # motion update
            # particles: Shape (N, particle_num, 3)
            particles = self.motion_model(act_[:, t:t+1, :, :].view(-1, particle_num, 3).float(),
                                          sta_[:, t-1:t, :, :].view(-1, particle_num, 3).float(),
                                          sta_[:, t:t+1, :, :].view(-1, particle_num, 3).float(),
                                          self.stds,
                                          self.means,
                                          self.state_step_sizes_,
                                          motion_mode,
                                          phrase)

            # measurement update
            # particle_probs (N, 1, resample_num)
            particle_probs = self.get_likelihood(particles, obs[:, t:t+1, :, :, :].float())
            # (N, resample_num)
            particle_probs = particle_probs.squeeze()

            # normalize probabilities
            particle_probs /= torch.sum(particle_probs, dim=1, keepdim=True)
            if t == 1:
                particle_list = particles.view(particles.size(0), 1, particles.size(1), particles.size(2))
                particle_probs_list = particle_probs.view(particle_probs.size(0), 1, particle_probs.size(1))
            else:
                particle_list = torch.cat((particle_list,
                                           particles.view(particles.size(0), 1, particles.size(1), particles.size(2))),
                                           1)
                particle_probs_list = torch.cat((particle_probs_list,
                                                 particle_probs.view(particle_probs.size(0), 1, particle_probs.size(1))),
                                                 1)

        return particle_list, particle_probs_list
        
    def get_likelihood(self, sta, obs):
        """ Process the data input and get the model output

        Args:
          sta: Tensor with size (N, sta_num, 3), states
          obs: Tensor with size (N, obs_num, 3, H, W), observations
        Returns:
            w: Tensor with size (N, obs_num, sta_num).
               The diagonal entries are likelihood of observations at their states.
               Other entries are likelihood of observations not at their states.
        """
        # obs (32, obs_num, 24, 24, 3) -> (32*obs_num, 3, 24, 24)
        o = obs.permute(0, 1, 4, 2, 3)
        o = o.view(-1, 3, 24, 24)
        e = self.observation_encoder(o.float())
        # get e (32*obs_num, 128)
        # get all the combinations of states and observations
        # -> (32, obs_num, 128)
        e = e.view(obs.size()[0], obs.size()[1], -1)
        # -> (32, obs_num, sta_num, 128)
        e = e.view(obs.size()[0], obs.size()[1], 1, e.size()[2]).repeat(1, 1, sta.size()[1], 1)
        # sta (32, sta_num, 3) -> (32, sta_num, 4)
        s = torch.cat(((sta[:, :, :2].double() - torch.from_numpy(self.means['s'])[:2]) / torch.from_numpy(self.stds['s'])[:2],
                       torch.cos(sta[:, :, 2:3].double()), torch.sin(sta[:, :, 2:3].double())), -1)
        # -> (32, obs_num, sta_num, 4)
        s = s.view(s.size()[0], 1, s.size()[1], s.size()[2]).repeat(1, obs.shape[1], 1, 1)
        # get all the combinations of states and observations
        # cat_input (32, obs_num, sta_num, 132)
        cat_input = torch.cat((e.float(), s.float()), -1)
        # -> (32*obs_num*sta_num, 132)
        cat_input = cat_input.view(-1, cat_input.size()[-1])

        # get w (32*obs_num*sta_num, 1)
        w = self.likelihood_estimator(cat_input.float())
        # -> (32, obs_num, sta_num)
        w = w.view(sta.size()[0], obs.size()[1], sta.size()[1])

        return w

    def train_e2e(self, mode=0, phrase=None, motion_model_path=None, observation_encoder_path=None, likelihood_estimator_path=None):
        """ Train DPF end-to-end

        :return:
        """
        batch_size = self.trainparam['batch_size']
        epochs = self.trainparam['epochs']
        lr = self.trainparam['learning_rate']
        particle_num = self.trainparam['particle_num']
        state_step_sizes = self.state_step_sizes_
        
        log_dir = 'log/e2e/mode_{}/'.format(mode)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
      
        save_dir = 'model/e2e/mode_{}/'.format(mode)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        if motion_model_path is not None:
            self.motion_model = self.motion_model.load_state_dict(torch.load(motion_model_path))
            self.motion_model = self.motion_model.float()
        if observation_encoder_path is not None:
            self.observation_encoder = self.observation_encoder.load_state_dict(torch.load(observation_encoder_path))
            self.observation_encoder = self.observation_encoder.float()
        if likelihood_estimator_path is not None:
            self.likelihood_estimator = self.likelihood_estimator.load_state_dict(torch.load(likelihood_estimator_path))
            self.likelihood_estimator = self.likelihood_estimator.float()

        if self.use_cuda:
            self.motion_model = self.motion_model.cuda()
            self.observation_encoder = self.observation_encoder.cuda()
            self.likelihood_estimator = self.likelihood_estimator.cuda()

        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.globalparam['workers'],
            pin_memory=True,
            sampler=None)
        val_loader = torch.utils.data.DataLoader(
            self.eval_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.globalparam['workers'],
            pin_memory=True)

        optimizer = torch.optim.Adam(list(self.motion_model.parameters())+
                                     list(self.observation_encoder.parameters())+
                                     list(self.likelihood_estimator.parameters()), lr)

        niter = 0
        for epoch in range(epochs):
            self.motion_model.train()
            self.observation_encoder.train()
            self.likelihood_estimator.train()

            for iteration, (sta, obs, act) in enumerate(train_loader):
                # Build ground truth inputs of size (batch_size, num_particles, 3)
                # 
                # -actions action at current time step
                # -particles: true state at previous time step
                # -states: true state at current time step

                # particle_list: Shape (batch_size, seq_len-1, num_particles, 3) 
                # particle_probs_list: Shape (batch_size, seq_len-1, num_particles)
                particle_list, particle_probs_list = self.connect_modules(particle_num=particle_num, 
                                                                          sta=sta, 
                                                                          obs=obs, 
                                                                          act=act, 
                                                                          motion_mode=mode, 
                                                                          phrase=phrase)

                particle_list = particle_list.contiguous().view(-1, particle_num, 3)
                particle_probs_list = particle_probs_list.contiguous().view(-1, particle_num)

                sta_ = sta.unsqueeze(2)
                sta_ = sta_.repeat(1, 1, particle_num, 1)
                sta_ = sta_[:, 1:, :, :].contiguous().view(-1, particle_num, 3)

                std = 0.01
                # dists: Shape (batch_size*(seq_len-1), num_particle)
                dists = data_process.square_distance(particle_list.double(), sta_.double(), state_step_sizes)
                dist_probs = particle_probs_list.double() / ((2 * np.pi * std ** 2)**0.5) * torch.exp(-dists.double() / (2.0 * std ** 2))
                # Add e for numerical stability
                e = 1e-16
                # Compute most likelihood estimate loss
                loss = torch.mean(-torch.log(e + torch.sum(dist_probs, dim=-1)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                niter += 1
                
                if niter % self.log_freq == 0:
                    print("Epoch:{}, Iteration:{}, loss:{}".format(epoch, niter, loss))
                    writer.add_scalar('train/loss', loss, niter)

                if niter % 1000 == 0:
                    torch.save(self.motion_model.state_dict(), save_dir+'motion_model_' + repr(niter) + '.pth')
                    torch.save(self.observation_encoder.state_dict(), save_dir+'observation_encoder_' + repr(niter) + '.pth')
                    torch.save(self.likelihood_estimator.state_dict(), save_dir+'likelihood_estimator_' + repr(niter) + '.pth')
        





