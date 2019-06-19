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
    per_channel (boolean):     If True will calculate separate means and variance for each image channel
    """

    means = list()
    variances = list()

    if per_channel:
        for channel in range(data.shape[1]):
            means.append(np.mean(data[:,channel]))
            variances.append(np.var(data[:,channel]))

            data[:,channel] = (data[:,channel] - means[channel]) / variances[channel]

    else:
        means.append(np.mean(data))
        variances.append(np.var(data))

        data = (data - means[0]) / variances[0]

    return data, means, variances


# ===================
# WEIGHT INITALISTAION FUNCTIONS
# ===================

def guassian_initalisation(num_inputs, num_output, relus=False):
    """ Generates a weight matrix initalised from a normal distribution with mean 0 and std 0.01

    num_inputs  (int):     Number of inputs to the layer
    num_outputs (int):     Number of outputs from the layer
    relus:      (boolean): Has no affect, just included for completness
    """

    return np.random.normal(0, 0.01, size=(num_inputs, num_outputs))


def xavier_initalisation(num_inputs, num_outputs, relus=False):
    """ Generates weight matrix using xavier initalisation
    Xavier initalisation aims to create a weights matrix with a variance of 1
    Since ReLu kills half the actiavtions at each layer we account for this by dividing again by 2

    num_inputs  (int):     Number of inputs to the layer
    num_outputs (int):     Number of outputs from the layer
    relus:      (boolean): Set to True if using ReLu activation functions
    """

    relu_offset = 2.0 if relus else 1.0
    return np.random.normal(0, 1, size=(num_inputs, num_outputs)) / np.sqrt(num_inputs / relu_offset)


def kaiming_initalisation():
    """ """
