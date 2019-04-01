import numpy as np
import os
import torch.nn as nn


class ActionSampler(nn.Module):
    def __init__(self, actions, particles, *args, **kwargs):
        super(ActionSampler, self).__init__()
        """
        :param actions: array of size (1,3) with motion between states
        :param particles: array of state of individual particles
        """
        self.actions = actions
        self.num_particles = particles.shape[0]

        self.layers = nn.Sequential(
            nn.Linear(6,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,3),
            )

    def sample_noise(self):
        """
        Create noisy actions input
        """
        # Propagate action over particles
        std_act = np.std(self.actions)
        actions_input = np.repeat(self.actions, self.num_particles, axis=0) / std_act
        actions_input = actions_input[..., np.newaxis]
        
        # Concatenate noise array to actions
        random_noise_input = np.random.normal(size=self.actions.shape)
        sampler_input = np.stack([self.actions, random_noise], axis=-1)
        return sampler_input, actions_input

    def forward(self, x):
        """
        Feed noisy inputs to action sampler f
        """
        x = self.layers(x)
        return x

    def get_noisy_actions(self):
        noisy_input, actions_input = self.sample_noise()
        out_noise = self.forward(noisy_input)
        noisy_actions = actions_input + out_noise



class DynamicModels(nn.Module):
    def __init__(self, particles, noisy_actions, *args, **kwargs):
        super(DynamicModels, self).__init__()
        """
        :param particles: array of state of individual particles (x, y, theta)
        :param noisy_actions: output of action sampler f ()
        """
        self.particles = particles
        self.noisy_actions = noisy_actions

        self.layers = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,3),
            )

    def transform_particles(self):
        mean = np.mean(self.particles, axis=0)
        std = np.std(self.particles, axis=0)

        trans_pos = (self.particles[:,:,2] - mean[:2]) / std[:2]
        trans_theta_cos = np.cos(self.particles[:,:,2])
        trans_theta_sin = np.sin(self.particles[:,:,2])

        particles_input = np.stack([trans_pos, 
                                   trans_theta_cos,
                                   trans_theta_sin], axis=-1)
        return particles_input

    def forward(self, x):
        pass

