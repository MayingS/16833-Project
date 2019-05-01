import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot_maze(maze='nav01', margin=1, means=None, stds=None, figure_name=None, show=False, pause=False, ax=None, linewidth=1.0):
    if ax is None:
        ax = plt.gca()
    if figure_name is not None:
        plt.figure(figure_name)

    if 'nav01' in maze:
        walls = np.array([
            # horizontal
            [[0, 500], [1000, 500]],
            [[400, 400], [500, 400]],
            [[600, 400], [700, 400]],
            [[800, 400], [1000, 400]],
            [[200, 300], [400, 300]],
            [[100, 200], [200, 200]],
            [[400, 200], [700, 200]],
            [[200, 100], [300, 100]],
            [[600, 100], [900, 100]],
            [[0, 0], [1000, 0]],
            # vertical
            [[0, 0], [0, 500]],
            [[100, 100], [100, 200]],
            [[100, 300], [100, 500]],
            [[200, 200], [200, 400]],
            [[200, 0], [200, 100]],
            [[300, 100], [300, 200]],
            [[300, 400], [300, 500]],
            [[400, 100], [400, 400]],
            [[500, 0], [500, 200]],
            [[600, 100], [600, 200]],
            [[700, 200], [700, 300]],
            [[800, 200], [800, 400]],
            [[900, 100], [900, 300]],
            [[1000, 0], [1000, 500]],
        ])
        rooms = [
            # [[400, 200], 300, 200]
            ]
        ax.set_xlim([-margin, 1000+margin])
        ax.set_ylim([-margin, 500+margin])

    if means is not None:
        walls -= means['pose'][:, :, :2]
    if stds is not None:
        walls /= stds['pose'][:, :, :2]
    # color = (0.8, 0.8, 0.8)
    color = (0, 0, 0)

    ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color=color, linewidth=linewidth)
    for room in rooms:
        ax.add_patch(Rectangle(*room, facecolor=(0.85, 0.85, 0.85), linewidth=0))
    ax.set_aspect('equal')