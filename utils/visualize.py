import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

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


def plot_particle_filter(task, particle_list, particle_probs_list, pred_state, sta,
                         save_image=False, outdir=None, batch=None):
    """ To plot the learned DPF

    Args:
      task: name of the maze
      particle_list: Tensor with size (N, T, particle_num, 3), particles at different time step
      particle_probs_list: Tensor with size (N, T, particle_num),
                           the particle probabilities at different time step
      pred_state: the predicted state obtained from the particles, Tensor with size (N, T, 4)
      sta: the ground truth state, Tensor with size (N, T, 3)
    Returns:
    """
    particle_list = particle_list.data.cpu().numpy()
    particle_probs_list = particle_probs_list.data.cpu().numpy()
    pred_state = pred_state.data.cpu().numpy()
    sta = sta.data.cpu().numpy()

    num_particles = particle_list.shape[1]
    head_scale = 1.5
    quiv_kwargs = {'scale_units': 'xy', 'scale': 1. / 40., 'width': 0.003, 'headlength': 5 * head_scale,
                   'headwidth': 3 * head_scale, 'headaxislength': 4.5 * head_scale}
    marker_kwargs = {'markersize': 4.5, 'markerfacecolor': 'None', 'markeredgewidth': 0.5}

    color_list = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3],
              'ff': color_list[4], 'odom': color_list[4]}

    num_steps = 20

    for s in range(1):

        plt.figure("example {}".format(s), figsize=[12, 5.15])
        plt.gca().clear()

        for i in range(num_steps):
            ax = plt.subplot(4, 5, i + 1, frameon=False)
            plt.gca().clear()

            plot_maze(task, margin=5, linewidth=0.5)

            if i < num_steps - 1:
                ax.quiver(particle_list[s, i, :, 0], particle_list[s, i, :, 1],
                          np.cos(particle_list[s, i, :, 2]), np.sin(particle_list[s, i, :, 2]),
                          particle_probs_list[s, i, :], cmap='viridis_r', clim=[.0, 0.2 / num_particles],
                          alpha=1.0,
                          **quiv_kwargs
                          )

                current_state = sta[s, i, :]
                plt.quiver(current_state[0], current_state[1], np.cos(current_state[2]),
                           np.sin(current_state[2]), color="red", **quiv_kwargs)

                plt.plot(current_state[0], current_state[1], 'or', **marker_kwargs)
            else:

                ax.plot(sta[s, :num_steps, 0], sta[s, :num_steps, 1], '-', linewidth=0.6, color='red')
                ax.plot(pred_state[s, :num_steps, 0], pred_state[s, :num_steps, 1], '-', linewidth=0.6,
                        color=colors['pf_ind_e2e'])

                ax.plot(sta[s, :1, 0], sta[s, :1, 1], '.', linewidth=0.6, color='red', markersize=3)
                ax.plot(pred_state[s, :1, 0], pred_state[s, :1, 1], '.', linewidth=0.6, markersize=3,
                        color=colors['pf_ind_e2e'])

            plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.001, hspace=0.1)
            plt.gca().set_aspect('equal')
            plt.xticks([])
            plt.yticks([])

        if save_image:
            plt.gca().axis('off')
            plt.gca().set_aspect('equal')
            plt.savefig('{}/particle_filter_example_{}.png'.format(outdir, batch), transparent=True, dpi=600, frameon=False, facecolor='w',
                        pad_inches=0.01)
