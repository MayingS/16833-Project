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
        return self.mle_loss

    def forward(self, actions, particles, states, stds, means, state_step_sizes):
    	"""
    	Forward actions and particles through motion model to obtain new particle states
		Args:
		  actions: array of size (1,3) containing (x, y, theta) actions
		  particles: array of size (N-1, 3) containing particle states
    	"""

    	# Feedforward
    	noisy_actions = self.action_sampler(actions,particles,stds,means)
    	moved_particles = self.dynamics_model(noisy_actions,particles,stds,means)

    	# Build loss
    	self.mle_loss = self.build_loss(moved_particles,
    									states,
    									state_step_sizes)

    	return moved_particles
    	
    def build_loss(self, moved_particles, states, state_step_sizes, std):
    	# Compute distance between each particle state and ground truth state
    	dists = square_distance(moved_particles, states, state_step_sizes)
    	# Transform distances to probabilities sampled from a normal distribution
    	dist_probs = (1 / moved_particles) / torch.sqrt(2 * np.pi * std ** 2) \
    				 * tf.exp(-dists / (2.0 * std ** 2))
    	
    	# Add e for numerical stability
    	e = 1e-12
    	# Compute most likelihood estimate loss
    	mle_loss = torch.mean(-torch.log(e + torch.sum(dist_probs)))

    	return mle_loss
