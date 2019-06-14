

import numpy as np

class sigmoid_layer:
    """ Implements a sigmoid layer (prone to saturation and non-zero mean)"""

    def __init__(self):
        """ Initalise the variables which will store the results for the backward pass """

        self.y = None       # store the output for faster computation of the backward pass

    def forward_pass(self, x):
        """ Computes the forward pass of a sigmoid layer

        x (numpy array): output of the previous layer's forward pass
        """

        x = np.clip(x, -700, 700)                           # clip input to prevent over flow
        self.y = np.array(1.0 / (1.0 + np.exp(-x)));

        return y

    def backward_pass(self, da):
        """ computes the backward pass of the sigmoid layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = self.y * (1 - self.y)
        return da * dx                   # multiply local gradient by downstream gradient


class tanh_layer:
    """ Implements a tanh layer (prone to saturation) """

    def __init__(self):
        """ Initalise the variables which will store the results for the backward pass """

        self.y = None       # store the output for faster computation of the backward pass

    def forward_pass(self, x):
        """ compute the forward pass of a tanh layer

        x (numpy array): output of the previous layer's forward pass
        """

        self.y = np.tanh(x)
        return

    def backward_pass(self, da):
        """ computes the backward pass of the tanh layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = 1 - (self.y ** 2)
        return da * dx                       # multiply local gradient by downstream gradient

class relu_layer:
    """ Implements a rectified linear unit layer (dead for negative inputs) """

    def __init__(self):
        """ Initalise the variables which will store the results for the backward pass """

        self.x = None       # store the output for computation of the backward pass

    def forward_pass(self, x):
        """ compute the forward pass of a relu layer

        x (numpy array): output of the previous layer's forward pass
        """

        self.x = x
        return np.maximum(x)

    def backward_pass(self, da):
        """ computes the backward pass of the relu layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = (self.x > 0).astype(self.x.dtype)
        return da * dx


class linear_layer:
    """ Implements a linear layer (used to scale data) """

    def __init__(self, scale=1):
        """ Initalise the variables which will store the results for the backward pass

        scale (float): The amount the linear layer will scale the data by
        """

        self.scale = scale
        self.shape = None       # store the output for faster computation of the backward pass

    def forward_pass(self, x):
        """ compute the forward pass of a linear layer

        x (numpy array): output of the previous layer's forward pass
        """

        self.shape = x.shape
        return x * scale                 # multiply local gradient by downstream gradient

    def backward_pass(self, da):
        """ Computes the backward pass of the linear layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = scale * np.ones(self.shape)
        return da * dx
