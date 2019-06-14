

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



class fully_connected_layer:
    """ Implements a fully connected layer, assumes flat inputs """

    # FIXME: Implement xaiver weight initalisation, use with deep networks
    def __init__(self, num_inputs, num_outputs, l1=None, l2=None):
        """ initalise the weights and the biases

        num_inputs  (int): Number of inputs to the layer (each neuron) for each training/test sample
        num_outputs (int): Number of neurons in the layer (number of outputs)
        l1:         (int): Lambda for L1 regularisation (encourages sparse weight matrix)
        l2:         (int): Lambda for L2 regularisation (weight decay, encourages small weights)
        """

        # initalise weights and biases
        self.biases = np.zeros((1, num_outputs))                                          # initalise biases to zero
        self.weights = np.random.normal(0.0, 0.01, size=(num_inputs, num_outputs))        # initalise weights to normal distribution with std=0.01

        self.l1 = l1
        self.l2 = l2

        # store input for computing backward pass
        self.x = None

    def forward_pass(self, x):
        """ Compute the forward pass of the fully connected layer

        x (numpy array): output of the previous layer's forward pass, this can be a batch of inputs
        """

        self.x = x
        return np.dot(x, self.weights) + self.biases

    def backward_pass(self, da, learning_rate):
        """
        Computes the backward pass of the fully connected layer and returns the gradients for the next upstream layer
        also updates the weights and the biases

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        # calculate the local gradients
        db = np.sum(da, axis=0)
        dw = np.dot(self.x.T, da)
        dx = np.dot(da, self.weights.T)

        # calculate regularisation terms
        l1_reg = self.l1 * np.sign(self.weights) if self.l1 else 0.0        # lasso regularisation
        l2_reg = self.l2 * self.weights if self.l2 else 0.0                 # ridge regularisation / weight decay

        # update the weights and biases
        self.biases -= learning_rate * db
        self.weights -= learning_rate * (dw + l1_reg + l2_reg)

        # return the gradients to the next upstream layer
        return da * dx
