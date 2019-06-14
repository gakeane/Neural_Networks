"""
This module implements the loss functions typically used by Neural Networks
All loss functions have the option of adding l1 and l2 regularistaion

1. Mean Square Error loss       (forward pass is quadratic, backward pass is a V)
2. Mean Absolute Error loss     (forward pass is a V, backward pass is -1 when a < y and 1 when a > y)
3. Huber loss                   (Combination of MSe and MAE)
4. Softmax loss
5. Hinge Loss
6. KL Divergnece                TODO

reference on some of the loss functions
https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

reference for the derivative of the softmax
https://deepnotes.io/softmax-crossentropy
"""

import matplotlib
matplotlib.use('qt4agg')

import numpy as np
import matplotlib.pyplot as plt


def mse_loss(activations, labels, l1=None, l2=None):
    """
    Mean square error, also known as L2 loss, good loss function for regression
    Derivative is linear, unlike MAE which has a constant derivative
    We should also use MSE if the outliers are important, I.e their not just corrupted data

    activations (numpy array):
    labels      (numpy array):
    """

    # calculate forward pass
    mse = 0.5 * np.mean(np.square(labels - activations))

    # calculate backward pass
    da = activations - labels

    # return the results
    return mse, da


# TODO: Consider what would happen using a loss function which is the magnitude of the error vector
# TODO: 0.5 * np.mean(np.linalg.norm(labels - activations), axis=1)
def mae_loss(activations, labels, l1=None, l2=None):
    """
    Mean absolute error, also known as L1 loss, good loss function for regression
    MAE is more robust to ouliers than MSE, i.e will tend to ignore them more

    activations (numpy array):
    labels      (numpy array):
    """

    # calculate forward pass
    mae = 0.5 * np.mean(np.abs(labels - activations))

    # calculate backward pass
    da = (activations >= labels).astype(activations.dtype) - (activations < labels).astype(activations.dtype)

    # return the results
    return mae, da


# TODO: It's possible to train delta, to do this we should make this loss function a class
def huber_loss(activations, labels, delta=10, l1=None, l2=None):
    """
    Also called smoothed mean absolute error, quadratic when error is small and linear when error is large
    Tuned with a parameter delta, MAE when delta=0 and MSE when delta=INF

    activations (numpy array):
    labels      (numpy array):
    delta       (float):       Smaller value makes loss more resistent to outliers
    """

    # calculate forward pass
    error = labels - activations
    loss = np.where(np.abs(error) < delta,                                      # check if absolute error is less than delta
                    np.mean(0.5 * (np.square(error))),                          # use MSE if error is less than delta
                    np.mean(delta * np.abs(error) - 0.5 * np.square(delta)))    # use MAE if error is greater than delta

    huber = np.mean(loss)

    # calculate backward pass
    da = np.where(np.abs(error) < delta,
                  activations - labels,
                  delta * ((activations >= labels).astype(activations.dtype) - (activations < labels).astype(activations.dtype)))

    # return the results
    return huber, da


# NOTE: It's possible to use this implementation of softmax for multiclass labels (i.e. for two lables are correct)
# NOTE: Shouldn't use softmax for multiclass, keep using cross-entropy and swap softmax for sigmoid
# FIXME: Comments are misleading now that i've changed this to a multiclass implementation
def softmax_loss(activations, labels, l1=None, l2=None):
    """
    The softmax function converts all activations to a value between 0 and 1 and the sum of activations for a sample is 1
    The largest activation will have the highest probability
    We calculate the loss as the log of the probability for the correct class
    The log function makes small numbers large, so if the probability for activation of the correct class is low the loss will be large

    activations (numpy array):
    labels      (numpy array):
    """

    # calculate forward pass

    # calculate the softmax for each activation (sum of all softmax for a sample should be 1)
    # we shift all the activations by the maximum actiavtion, this makes them negative so the exp function won't overflow
    # The shift has no affect on the final result since we normalise the values anyway
    exps = np.exp(activations - np.max(activations, axis=1, keepdims=True))
    softmax = exps / np.sum(exps, axis=1, keepdims=True)

    # the loss is the log of the softmax for the correct activation
    # (ideally this will be the smalest softmax value)
    # cross_entropy = -np.log(softmax[range(m), y_index])
    cross_entropy = np.sum(-y * np.log(softmax), axis=1)               # get the sum of log likelihood of the softmax for all activations
    loss = np.mean(cross_entropy)                                      # get the average of the loss functions for all samples

    # calculate backward pass
    da = softmax - y                                                   # the derivative of the loss is just the softmax - y

    # return the results
    return loss, da


def hinge_loss(activations, labels, l1=None, l2=None):
    """
    When implementing hinge loss the incorrect class labels need to be minus 1
    The activations also need to output in the range -INF to INF

    activations (numpy array):
    labels      (numpy array):
    """

    # calculate forward pass
    hinge = np.maximum(0.0, 1 - (activations * labels))

    # calculate backward pass

    # return the results
    return hinge
