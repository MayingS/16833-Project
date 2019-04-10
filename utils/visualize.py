import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from modeling.DPF import *


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
