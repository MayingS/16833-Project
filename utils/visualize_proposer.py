import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from .visualization_utils import *


def plot_proposer(task, vis_outdir, particles, sta):
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

    num_particles = particles.shape[0]

    fig = plt.figure('Porposer')
    plt.gca().clear()
    plot_maze(task, margin=5, linewidth=0.5)

    # idxs = [0, 10, 50, 80, 130, 200, 300, 420, 540, 600]

    for i in range(num_particles):
        plt.quiver([particles[i,0]], [particles[i,1]], np.cos([particles[i,2]]),
                   np.sin([particles[i,2]]), color='black',
                   **quiv_kwargs_1)
        plt.plot(particles[i, 0], particles[i, 1], 'o', color='black', linewidth=3, **marker_kwargs)

    plt.plot(sta[0], sta[1], 'o', color='red', linewidth=3, **marker_kwargs)
    plt.quiver(sta[0], sta[1], np.cos(sta[2]),
                   np.sin(sta[2]), color='red',
                   **quiv_kwargs_1)
    
    plt.gca().axis('off')
    plt.gca().set_aspect('equal')
    plt.savefig(vis_outdir+'.png', transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)