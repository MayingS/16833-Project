import numpy as np
import os
import torch.nn as nn

from utils.data_process import *

class ActionSampler(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ActionSampler, self).__init__()
        """
        Model to obtain random noisy actions from current action input
        """

        self.layers = nn.Sequential(
            nn.Linear(6,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,3),
            )

    def sample_noise(self, actions, particles, stds, means):
        """
        Create input for action sampler network f
        Args:
          actions: array of size (batch_size, 1, 3)
          particles: array of size (batch_size, num_particles, 3)

        returns:
          sampler_input: concatenated array of size (batch_size, num_particles, 6)
          actions_input: array of size (batch_size, num_particles, 3) containing
                         actions propagated over all particles
        """
        # Normalize actions
        actions_input = actions / stds["actions"]
        
        # Concatenate noise array to actions
        random_noise_input = torch.rand_like(actions_input)
        sampler_input = torch.cat((actions_input, random_noise_input), axis=-1)
        
        return sampler_input, actions_input

    def forward(self, actions, particles, stds, means):
        """
        Feed forward actions to action sampler network f
        """

        sampler_input, actions_input = self.sample_noise(actions, particles, stds, means)

        # Reshape concatenated array and pass to newtwork
        sampler_input = sampler_input.view(-1, 6)
        batch_size = sampler_input.size(0)
        delta_noise = self.layers(sampler_input)
        # Zero-centering of output noisy actions
        delta_noise = delta_noise - torch.mean(delta_noise)
        # Reshape output back into original size (batch_size, num_particles, 3)
        delta_noise = delta_noise.view(batch_size, -1, 3)

        noisy_actions = actions_input + delta_noise

        return noisy_actions


class DynamicsModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DynamicModels, self).__init__()
        """
        Dynamics model which models dynamics between actions and state
        """

        self.layers = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,3),
            )

    def transform_particles(self, particles, stds, means):
        """
        Transform particle state to network input form
        Args:
          particles: array of size (batch_size, num_particles, 3)

        returns:
          particles_input: tensor of size (batch_size, num_particles, 4) with 
                           channels (x, y, cos(theta), sin(theta))
        """

        norm_pos = (particles[:,:,:2] - means["states"][:,:,:2]) \
                    / stds["states"][:,:,:2]
        cos_theta = torch.cos(particles[:,:,2])
        sin_theta = torch.sin(particles[:,:,2])

        particles_input = torch.cat([norm_pos, 
                                    cos_theta,
                                    sin_theta], axis=-1)

        return particles_input

    def model_input(self, noisy_actions, particles, stds, means):
        """
        Create input for dynamics model g
        Args:
          noisy_actions: output of action sampler f, 
                         tensor of size (batch_size, num_particles, 3)
        returns:
          noisy_input: concatenated tensor of size (batch_size, num_particles, 8)
        """
        particles_input = self.transform_particles(particles, stds, means)
        noisy_actions = noisy_actions / stds["actions"]
        noisy_input = torch.cat([particles_input, noisy_actions], axis=-1)

        return noisy_input
    
    def forward(self, noisy_actions, particles, stds, means):
        """
        Feedforward action input to obtain "delta state"
        """
        noisy_input = self.model_input(noisy_actions, particles, stds, means)

        # Reshape concatenated tensor and pass to network
        noisy_input = noisy_input.view(-1, 8)
        batch_size = noisy_input.size(0)
        delta_state = self.layers(noisy_input)
        # Reshape output back to original size compatiable with particles array
        delta_state = delta_state.view(batch_size, -1, 3)

        moved_particles = particles + delta_state
        moved_particles[:,:,2] = wrap_angle(moved_particles[:,:,2])

        return moved_particles
