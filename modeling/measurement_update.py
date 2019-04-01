import numpy as np
import os
import torch.nn as nn


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
    def __init__(self, keepRatio):
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
