import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable


class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()
        self.Conv = nn.Sequential(
            # size from (3, 24, 24) to (16, 11, 11)
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            # size from (16, 11, 11) to (32, 5, 5)
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            # size from (32, 5, 5) to (64, 2, 2)
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7)
        )
        self.Fc = nn.Sequential(
            nn.Linear(2*2*64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.Conv(x)
        #print(x.size())
        x = x.view(-1, 2*2*64)
        #print(x.size())
        x = self.Fc(x)
        #print(x.size())

        return x


class ParticleProposer(nn.Module):
    def __init__(self, keepRatio=0.3):
        super(ParticleProposer, self).__init__()
        self.proposer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(1 - keepRatio),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
    
    def propose_particle(self, encoding, num_particles, state_mins, state_maxs):
        """
        Args:
            encoding: output of observation encoder tensor shape: (128, )
            num_particles: number of particles
            state_mins: minimum values of states, numpy array of shape (1, 2)
            state_maxs: maximum values of states, numpy array of shape (1, 2)
        Returns:
            proposed_particles: np array of new proposed states: (N, )
        """
        # encoding = Variable(encoding, requires_grad=False)
        encoding_rep = encoding.repeat(num_particles, 1)
        proposed_particles = self.forward(encoding_rep)

        # transform states 4 dim to 3 dim
        x = proposed_particles[:, 0] * \
            (state_maxs[0] - state_mins[0]) / 2.0 + (state_maxs[0] + state_mins[0]) / 2.0
        y = proposed_particles[:, 1] * \
            (state_maxs[1] - state_mins[1]) / 2.0 + (state_maxs[1] + state_mins[1]) / 2.0
        theta = torch.atan2(proposed_particles[:, 2], proposed_particles[:, 3])

        proposed_particles = torch.cat((x.unsqueeze(1), y.unsqueeze(1),\
             theta.unsqueeze(1)), 1)

        return proposed_particles

    def forward(self, x):
        return self.proposer(x)


class ObservationLikelihoodEstimator(nn.Module):
    def __init__(self):
        super(ObservationLikelihoodEstimator, self).__init__()
        self.Fc = nn.Sequential(
            nn.Linear(128+4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, min_obs_likelihood=0.004):
        x = self.Fc(x)
        # scale to range [min_obs_likelihood, 1.0]
        x = x * (1 - min_obs_likelihood) + min_obs_likelihood
        return x
