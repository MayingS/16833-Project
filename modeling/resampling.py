import numpy as np
import os
import torch


def particle_resampling(particles, particle_probs, resample_num):
    """ Perform particle resampling

    Args:
      particles: Tensor with size (N, particle_num, 3)
      particle_probs: Tensor with size (N, particle_num)
      resample_num: int, the particle numbers to be sampled
    Returns:
      resampled_particles: Tensor with size (N, resample_num, 3)
    """
    batch_size = particles.size(0)
    particle_num = particles.size(1)

    basic_markers = torch.linspace(0.0, (resample_num - 1.0) / resample_num, resample_num)
    random_offset = torch.empty(batch_size, 1).uniform_(0.0, 1.0/resample_num)
    # shape: batch_size x resample_num
    basic_markers = basic_markers.view(1, -1).repeat([batch_size, 1])
    random_offset = random_offset.repeat([1, resample_num])
    markers = random_offset + basic_markers
    # calculate the cumulative sum, shape: batch_size x particle_num
    probs = torch.cumsum(particle_probs, dim=1)
    # shape: batch_size x resample_num x num_particles
    markers = markers.view(markers.size(0), markers.size(1), 1).repeat([1, 1, particle_num])
    probs = probs.view(probs.size(0), 1, probs.size(1)).repeat([1, resample_num, 1])
    marker_matching = markers < probs

    # batch_size x resample_num
    samples = torch.argmax(marker_matching, dim=2)
    resampled_particles = torch.zeros(batch_size, resample_num, particles.size(2))
    for i in range(batch_size):
        resampled_particles[i] = particles[i, samples[i]]

    return resampled_particles
