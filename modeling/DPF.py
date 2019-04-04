import numpy as np
import os
import shutil
import modeling.state_prediction as motion
import modeling.measurement_update as measurement
import modeling.resampling as resample
import config.set_parameters as sp

import torch
import torch.utils
import torch.utils.data
import torch

from tensorboardX import SummaryWriter


class DPF:
    def __init__(self, train_set=None, eval_set=None, means=None, stds=None):
        self.train_set = train_set
        self.eval_set = eval_set
        self.means = means
        self.stds = stds

        self.action_sampler = motion.ActionSampler()
        self.dynamic_model = motion.DynamicModels()
        self.observation_encoder = measurement.ObservationEncoder()
        self.particle_proposer = measurement.ParticleProposer()
        self.likelihood_estimator = measurement.ObservationLikelihoodEstimator()
        self.resampling = resample.particle_resampling

        params = sp.Params()
        self.globalparam = params.globalparam
        self.trainparam = params.train
        self.testparam = params.test

        self.use_cuda = torch.cuda.is_available()

        self.log_freq = 10  # Steps
        self.test_freq = 2  # Epoch

    def connect_modules(self):
        """ Connect all the modules together to form the whole DPF system

        :return:
        """
        pass

    def train_motion_model(self):
        """ Train the motion model (action sampler) f.

        :return:
        """
        pass

    def train_dynamic_model(self):
        """ Train the dynamic model g.

        :return:
        """
        pass

    def train_particle_proposer(self):
        """ Train the particle proposer k.

        :return:
        """
        pass

    def train_likelihood_estimator(self):
        """ Train the observation likelihood estimator l (and h)

        :return:
        """
        batch_size = self.trainparam['batch_size']
        epochs = self.trainparam['epochs']
        lr = self.trainparam['learning_rate']

        self.observation_encoder = self.observation_encoder.double()
        self.likelihood_estimator = self.likelihood_estimator.double()
        if self.use_cuda:
            self.observation_encoder = self.observation_encoder.cuda()
            self.likelihood_estimator = self.likelihood_estimator.cuda()

        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.globalparam['workers'],
            pin_memory=True,
            sampler=None)
        val_loader = torch.utils.data.DataLoader(
            self.eval_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.globalparam['workers'],
            pin_memory=True)

        optimizer = torch.optim.Adam(list(self.observation_encoder.parameters())+
                                     list(self.likelihood_estimator.parameters()), lr)

        log_dir = 'likelihood_estimator_log'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        log_writer = SummaryWriter(log_dir)

        niter = 0
        for epoch in range(epochs):
            self.observation_encoder.train()
            self.likelihood_estimator.train()

            for i, (sta, obs, act) in enumerate(train_loader):
                if self.use_cuda:
                    sta = sta.cuda()
                    obs = obs.cuda()
                w = self.get_likelihood(sta, obs)

                # define loss (correct -> 1, incorrect -> 0) and optimizer
                correct_item = 0
                incorrect_item = 0
                for batch_ind in range(w.size()[0]):
                    correct_samples = torch.diag(w[batch_ind])
                    incorrect_samples = w[batch_ind] - torch.diag(torch.diag(w[batch_ind]))
                    correct_item += torch.sum(-torch.log(correct_samples))
                    incorrect_item += torch.sum(-torch.log(1.0 - incorrect_samples))
                loss = correct_item / w.size()[0] + incorrect_item / (w.size()[0]*(w.size()[0]-1))

                # log and visualize
                if niter % self.log_freq == 0:
                    print('Epoch {}/{}, Batch {}/{}: Train loss: {}'.format(epoch, epochs, i, len(train_loader), loss))
                    log_writer.add_scalar('train/loss', loss, niter)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                niter += 1

            if epoch % self.test_freq == 0:
                likelihood = self.eval_likelihood_estimator(val_loader)
                print('Epoch {}: Val likelihood: {}'.format(epoch, likelihood))
                log_writer.add_scalar('val/likelihood', likelihood, niter)

    def get_likelihood(self, sta, obs):
        """

        :param sta:
        :param obs:
        :return:
        """
        # obs (32, 20, 3, 24, 24) -> (32*20, 3, 24, 24)
        o = obs.view(-1, 3, 24, 24)
        e = self.observation_encoder(o)
        # get e (32*20, 128)
        # get all the combinations of states and observations
        # -> (32, 20, 128)
        e = e.view(obs.size()[0], obs.size()[1], -1)
        # -> (32, 20, 20, 128)
        e = e.view(obs.size()[0], obs.size()[1], 1, e.size()[2]).repeat(1, 1, sta.size()[1], 1)
        # sta (32, 20, 3) -> (32, 20, 4)
        s = torch.cat(((sta[:, :, :2] - torch.from_numpy(self.means['s'])[:2]) / torch.from_numpy(self.stds['s'])[:2],
                       torch.cos(sta[:, :, 2:3]), torch.sin(sta[:, :, 2:3])), -1)
        # -> (32, 20, 20, 4)
        s = s.view(s.size()[0], 1, s.size()[1], s.size()[2]).repeat(1, obs.shape[1], 1, 1)
        # get all the combinations of states and observations
        # cat_input (32, 20, 20, 132)
        cat_input = torch.cat((e, s), -1)
        # -> (32*20*20, 132)
        cat_input = cat_input.view(-1, cat_input.size()[-1])

        # get w (32*20*20, 1)
        w = self.likelihood_estimator(cat_input)
        # -> (32, 20, 20)
        w = w.view(sta.size()[0], sta.size()[1], sta.size()[1])

        return w

    def eval_likelihood_estimator(self, val_loader):
        """

        :param val_loader:
        :return:
        """
        likelihood_list = []
        self.observation_encoder.eval()
        self.likelihood_estimator.eval()

        for i, (sta, obs, act) in enumerate(val_loader):
            if self.use_cuda:
                sta = sta.cuda()
                obs = obs.cuda()
            w = self.get_likelihood(sta, obs)

            # calculate the likelihood
            correct_item = 0
            incorrect_item = 0
            for batch_ind in range(w.size()[0]):
                correct_samples = torch.diag(w[batch_ind])
                incorrect_samples = w[batch_ind] - torch.diag(torch.diag(w[batch_ind]))
                correct_item += torch.sum(torch.log(correct_samples))
                incorrect_item += torch.sum(torch.log(1.0 - incorrect_samples))
            likelihood = correct_item / w.size()[0] + incorrect_item / (w.size()[0] * (w.size()[0] - 1))
            likelihood_list.append(likelihood)

        likelihood = sum(likelihood_list) / len(likelihood_list)
        return likelihood

    def train_e2e(self):
        """ Train DPF end-to-end

        :return:
        """
        pass

    def predict(self):
        """ Predict the output given the trained model.

        :return:
        """
        pass
