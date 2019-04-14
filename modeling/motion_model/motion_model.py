import numpy as np
import os
import torch.nn as nn

from .state_prediction import *
from utils.data_process import *

class MotionModel(nn.Module):
    """
    Main class for motion model, consists of action sampler f
    and dynamics model g
    """
    def __init__(self):
        super(MotionModel, self).__init__()
        """
        Build motion model from networks f and g
        """
        self.action_sampler = ActionSampler()
        self.dynamics_model = DynamicsModel()

    @property
    def loss(self):
        return self._loss

    def forward(self, actions, particles, states, stds, means, state_step_sizes, mode, phrase=None, train=True):
        """
        Forward actions and particles through motion model to obtain new particle states
	Args:
	    actions: array of size (1,3) containing (x, y, theta) actions
	    particles: array of size (N-1, 3) containing particle states
	    state_step_sizes: array of size (N-1, 3) containing the expection of the difference between two particle states
            mode: 0 - disable learnable dynamics model; 1 - enable learnable dynamics model
	    phrase: only applicable when mode = 1, 0 - train dynamics model, 1 - train motion model
        """
        # Feedforward
        noisy_actions = self.action_sampler(actions, particles, stds, means)
        
        if mode == 0: 
            theta = particles[:, :, 2:3]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            new_x = particles[:, :, 0:1] + (noisy_actions[:, :, 0:1] * sin_theta + noisy_actions[:, :, 1:2] * cos_theta)
            new_y = particles[:, :, 1:2] + (-noisy_actions[:, :, 0:1] * cos_theta + noisy_actions[:, :, 1:2] * sin_theta)
            new_theta = wrap_angle(particles[:, :, 2:3] + noisy_actions[:, :, 2:3])
            moved_particles = torch.cat((new_x, new_y, new_theta), dim=-1)
    	    # Build loss
            if train:
                self._loss = self.build_mle_loss(moved_particles, states, state_step_sizes)
        
        elif mode == 1:
            if phrase == 0:
                moved_particles = self.dynamics_model(noisy_actions.detach(), particles, state_step_sizes, stds, means)
                # Build loss
                if train:
                    self._loss = self.build_mse_loss(moved_particles, states, state_step_sizes)

            elif phrase == 1:
                moved_particles = self.dynamics_model(noisy_actions, particles, state_step_sizes, stds, means)
                # Build loss
                if train:
                    self._loss = self.build_mle_loss(moved_particles, states, state_step_sizes)

        return moved_particles
    	
    def build_mle_loss(self, moved_particles, states, state_step_sizes):
    	# Compute distance between each particle state and ground truth state
        std = 0.01
        dists = square_distance(moved_particles, states, state_step_sizes)
        # Transform distances to probabilities sampled from a normal distribution
        dist_probs = (1 / float(moved_particles.size(1))) / ((2 * np.pi * std ** 2)**0.5) * torch.exp(-dists / (2.0 * std ** 2))
    	# Add e for numerical stability
        e = 1e-16
    	# Compute most likelihood estimate loss
        mle_loss = torch.mean(-torch.log(e + torch.sum(dist_probs, dim=-1)))
        
        return mle_loss

    def build_mse_loss(self, moved_particles, states, state_step_sizes):
        # Compute distance between each particle state and ground truth state
        dists = square_distance(moved_particles, states, state_step_sizes)
        # Compute min square loss
        mse_loss = torch.mean(dists)
        
        return mse_loss
