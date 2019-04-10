import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from modeling.DPF import *
from .visualization_utils import *


def plot_measurement(measurement_output, save_image=False, outdir=None, batch=None, ind=None):
    """ To plot the output of the observation likelihood estimator

    Args:
      measurement_output: numpy array with size (seq_length, seq_length),
                          entry (t1, t2) is the likelihood of t2-state has t1-observation.
      save_image: True or False, indicating whether to save the image into file
      outdir: the path of the output directory
      batch: the batch number, used to generate the output file name
      ind: the index of the output in the batch, used to generate the output file name
    Returns:
    """
    plt.figure('Measurement Model Output')
    plt.gca().clear()
    plt.imshow(measurement_output, interpolation="nearest", cmap="coolwarm")
    if save_image:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        plt.imsave('{}/measurement_model_output_{}_{}.jpg'.format(outdir, batch, ind),
                   measurement_output, cmap='coolwarm')

def plot_motion_model(task, vis_outdir, particles, states, moved_particles):
    """ To plot learned the motion model
    Args:
        task: name of the maze
        vis_outdir: the path of the output directory
        particles: states of the previous time step (s_{t-1})
        states: states of the current time step (s_t)
        moved_particles: the particles output by the motion model
    """
    quiv_kwargs_1 = {'scale_units':'xy', 'scale':1./40., 'width': 0.003, 'headlength': 5, 'headwidth': 3, 'headaxislength': 4.5}
    quiv_kwargs_2 = {'scale_units':'xy', 'scale':1./40., 'width': 0.003, 'headlength': 5, 'headwidth': 3, 'headaxislength': 4.5}
    marker_kwargs = {'markersize': 4.5, 'markerfacecolor':'None', 'markeredgewidth':1}

    num_particles = particles.shape[1]

    fig = plt.figure('Motion Model')
    plt.gca().clear()
    plot_maze(task, margin=5, linewidth=0.5)

    idxs = [0, 10, 50, 80, 130, 200, 300, 420, 540, 600]

    for i in idxs:

        plt.quiver(moved_particles[i, :, 0], moved_particles[i, :, 1], 0.7*np.cos(moved_particles[i, :, 2]),
                          0.7*np.sin(moved_particles[i, :, 2]), np.ones([num_particles]), cmap='viridis_r',
                   **quiv_kwargs_2,
                   alpha=1.0, clim=[0, 2])

        plt.quiver([particles[i, 0, 0]], [particles[i, 0, 1]], np.cos([particles[i, 0, 2]]),
                   np.sin([particles[i, 0, 2]]), color='black',
                   **quiv_kwargs_1)

        plt.quiver([states[i, 0, 0]], [states[i, 0, 1]], np.cos([states[i, 0, 2]]),
                   np.sin([states[i, 0, 2]]), color='red',
                   **quiv_kwargs_1)

        plt.plot([particles[i, 0, 0], states[i, 0, 0]], [particles[i, 0, 1], states[i, 0, 1]], '--', color='black', linewidth=0.3)
        plt.plot(particles[i, 0, 0], particles[i, 0, 1], 'o', color='black', linewidth=3, **marker_kwargs)
        plt.plot(states[i, 0, 0], states[i, 0, 1], 'o', color='red', linewidth=3, **marker_kwargs)

    plt.gca().axis('off')
    plt.gca().set_aspect('equal')
    plt.savefig(vis_outdir+'motion_model.png', transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)
