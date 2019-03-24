import numpy as np
import os
import modeling.state_prediction as motion
import modeling.measurement_update as measurement
import modeling.resampling as resample
import config.set_parameters as sp


class DPF:
    def __init__(self):
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
        pass

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
        pass

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
