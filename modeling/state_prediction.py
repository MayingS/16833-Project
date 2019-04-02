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
        self.particles = particles

        self.layers = nn.Sequential(
            nn.Linear(6,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,3),
            )

    def sample_noise(self):
        """
        Create input for action sampler network f
        """
        # Propagate action over particles
        H, W = self.particles.shape[0], self.particles.shape[1]
        actions_input = np.tile(self.actions.ravel(), (H, W, 1))
        
        # Concatenate noise array to actions
        random_noise_input = np.random.normal(size=actions_input.shape)
        sampler_input = np.stack([actions_input, random_noise_input], axis=-1)
        
        return sampler_input, actions_input

    def forward(self, x):
        """
        Feed noisy inputs x to action sampler network f
        """
        x = torch.from_numpy(x)
        x = x.view(-1,6)
        x = self.layers(x)

        return x

    def get_noisy_actions(self):
        """
        Method to retrieve noisy actions from action sampler f
        """
        H, W = self.particles.shape[0], self.particles.shape[1]

        noisy_input, actions_input = self.sample_noise()

        out_noise = self.forward(noisy_input)

        out_noise = out_noise.numpy()
        out_noise = np.reshape((H, W, 3))
        out_noise = out_noise - np.mean(out_noise)

        noisy_actions = actions_input + out_noise

        return noisy_actions



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
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,3),
            )

    def transform_particles(self):
        """
        Transform particle state to network input form
        """
        trans_pos = self.particles[:,:,:2]
        trans_theta_cos = np.cos(self.particles[:,:,2])
        trans_theta_sin = np.sin(self.particles[:,:,2])

        particles_input = np.stack([trans_pos, 
                                   trans_theta_cos,
                                   trans_theta_sin], axis=-1)

        return particles_input

    def model_input(self):
        """
        Create network model g input
        """
        particles_input = self.transform_particles()
        action_input = self.noisy_actions
        input = np.stack([particles_input, actions_input], axis=-1)

        return input
    
    def forward(self, x):
        """
        Feedforward action input to obtain "delta state"
        """
        x = torch.from_numpy(x)
        x = x.view(-1,8)
        x = self.layers(x)

        return x

    def get_moved_particles(self):
        """
        Method to retrieve final particles with added motion
        """
        H, W = self.particles.shape[0], self.particles.shape[1]

        input = self.model_input()

        output_actions = self.forward(input)

        output_actions = torch.numpy(output_actions)
        output_actions = np.reshape((H, W, 3))
        moved_particles = self.particles + output_actions
        moved_particles[:,:,2] = wrap_angle(moved_particles[:,:,2])

        return moved_particles


