"""
Methods for preprocessing data and initalising weights

"""

import numpy as np


# ===================
# DATA PREPROCESSING FUNCTIONS
# ===================

# FIXME: WIll divide by zero if variance of input data is zero
def z_scale(data, per_channel=False):
    """ Zero mean and unit variance of the data
    Returns the calculated means and variance for each channel

    data        (numpy array): All data, samples(N) x channels(C) x rows(H) x columns(W)
    per_channel (boolean):     If True will calculate separate means and std for each image channel
    """

    means = list()
    stds = list()

    if per_channel:
        for channel in range(data.shape[1]):
            means.append(np.mean(data[:,channel]))
            stds.append(np.std(data[:,channel]))

            data[:,channel] = (data[:,channel] - means[channel]) / stds[channel]

    else:
        means.append(np.mean(data))
        stds.append(np.std(data))

        data = (data - means[0]) / stds[0]

    return data, means, stds


# ===================
# WEIGHT INITALISTAION FUNCTIONS
# ===================

def xavier_initalisation(num_inputs, num_outputs):
    """ Generates weight matrix using xavier initalisation
    Xavier initalisation aims to create a weights matrix with a variance of 1

    num_inputs  (int): number of inputs to the layer
    num_outputs (int): number of outputs from the layer
    """

    return np.random.normal(0, 1, size=(num_inputs, num_outputs)) / np.sqrt(num_inputs)


def kaiming_initalisation():
    """ """
