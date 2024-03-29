

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

        return self.y

    def backward_pass(self, da):
        """ computes the backward pass of the sigmoid layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = self.y * (1 - self.y)
        return da * dx                   # multiply local gradient by downstream gradient

    def get_regularisation(self):
        """ Sigmoid layer has no weights so regularisation is zero """

        return 0.0


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
        return self.y

    def backward_pass(self, da):
        """ computes the backward pass of the tanh layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = 1 - (self.y ** 2)
        return da * dx                       # multiply local gradient by downstream gradient

    def get_regularisation(self):
        """ tanh layer has no weights so regularisation is zero """

        return 0.0

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
        return np.maximum(0.0, x)

    def backward_pass(self, da):
        """ computes the backward pass of the relu layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = (self.x > 0).astype(self.x.dtype)
        return da * dx

    def get_regularisation(self):
        """ ReLu layer has no weights so regularisation is zero """

        return 0.0


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
        return x * self.scale                 # multiply local gradient by downstream gradient

    def backward_pass(self, da):
        """ Computes the backward pass of the linear layer and returns the gradients for the next upstream layer

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        dx = self.scale * np.ones(self.shape)
        return da * dx

    def get_regularisation(self):
        """ Linear layer has no weights so regularisation is zero """

        return 0.0



# TODO: method that will return the amount of loss due to reqularisation (might want to implement for all layers)
class fully_connected_layer:
    """ Implements a fully connected layer, assumes flat inputs """

    # FIXME: Implement xaiver weight initalisation, use with deep networks
    def __init__(self, num_inputs, num_outputs, optimiser, optimiser_parameters=None, weight_initaliser=None, l1=None, l2=None):
        """ initalise the weights and the biases

        num_inputs        (int):   Number of inputs to the layer (each neuron) for each training/test sample
        num_outputs       (int):   Number of neurons in the layer (number of outputs)
        l1:               (float): Lambda for L1 regularisation (encourages sparse weight matrix)
        l2:               (float): Lambda for L2 regularisation (weight decay, encourages small weights)
        optimiser:        (class): Class implemention
        parameters:       (dict):  Dictionary containing the parameters to initalsie the optimiser
        weight_initaliser (func):  Function used to initalise weights for the layer, default is xavier_initalisation
        """

        # initalise weights and biases
        self.biases = np.zeros((1, num_outputs))                                          # initalise biases to zero
        self.weights = weight_initaliser(num_inputs, num_outputs)                         # initalise weights using weight initalisation function

        # initalise regularisation variables
        self.l1 = l1
        self.l2 = l2

        # create optimiser instances for the weights and the biases
        self.weight_optimiser = optimiser(optimiser_parameters)
        self.bias_optimiser = optimiser(optimiser_parameters)

        # store input for computing backward pass
        self.x = None

    def forward_pass(self, x):
        """ Compute the forward pass of the fully connected layer

        x (numpy array): output of the previous layer's forward pass, this can be a batch of inputs
        """

        self.x = x
        return np.dot(x, self.weights) + self.biases

    def backward_pass(self, da):
        """
        Computes the backward pass of the fully connected layer and returns the gradients for the next upstream layer
        also updates the weights and the biases

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        N = da.shape[0]                             # divide updates by sample size so they are invariant to sample size

        # calculate the local gradients
        db = (1.0/N) * np.sum(da, axis=0)
        dw = (1.0/N) * np.dot(self.x.T, da)
        dx = np.dot(da, self.weights.T)

        # calculate regularisation terms
        if self.l1:
            l1_reg = self.l1 * np.sign(self.weights) if self.l1 else 0.0        # lasso regularisation
            dw += l1_reg

        if self.l2:
            l2_reg = self.l2 * self.weights if self.l2 else 0.0                 # ridge regularisation / weight decay
            dw += l2_reg

        # update the weights and biases based on direction given by optimser
        self.biases -= self.bias_optimiser.update(db)
        self.weights -= self.weight_optimiser.update(dw)

        # return the gradients to the next upstream layer
        return dx

    def get_regularisation(self):
        """ Returns the amount of loss due to L1 and L2 regularistion for the fully connected layer """

        l1 = self.l1 * np.sum(np.abs(self.weights)) if self.l1 else 0.0
        l2 = (self.l2 * 0.5) * np.sum(np.power(self.weights, 2)) if self.l2 else 0.0

        return l1 + l2


# TODO: pass in optimiser so we don't default to SGD
class batch_normalisation_layer:
    """ Implements a batch normalisation layer

    The inputs to the layer are scaled and mean shifted to have zero mean and a standard deviation of 1
    The beta and gamma variables are learned by the network
    This will also work in convolutional networks, the input will be flattened and the reshaped to make the maths easier

    N is the number of samples (batch size)
    D is the number of inputs per sample (activations of previos layer)
    """

    def __init__(self, input_dimensions):
        """ Initalise gamma and beta to 1s and 0s

        input_dimensions: (1D array) list of the dimensions of the input i.e (channels, height, width)
        """

        # initalise beta to zero and gamma to one so that they have no affect
        self.beta = np.zeros((1, np.product(input_dimensions)))  # dimension (1, D)
        self.gamma = np.ones((1, np.product(input_dimensions)))  # dimension (1, D)
        self.epsilon = 0.000001

        # cache for the forward pass to use in the backward pass
        self.x_norm = None    # the normalised data (dimension (N, D))
        self.mu = None        # Not actually used for the backward pass (kept for debugging)
        self.var = None       # store the value of the variance (dimension (1, D))
        self.x_mu = None      # store the value of x - mu (dimension (N, D))
        self.var_inv = None   # store the value of 1 / sqrt(var + epsilon) (dimension(1, D))

    def forward_pass(self, x):
        """ Computes the forward pass of the batch normalisation layer

        x (numpy array): Assumes a batch of inputs (minimum 2D array) where the first dimension is each sample
        """

        # flatten the inputs so we have an array of N x D
        x_flat = np.reshape(x, (x.shape[0], np.product(x.shape[1:])))   # dimension (N, D)

        # calculate the mean and variance of each input across the entire batch
        self.mu = np.mean(x_flat, axis=0)   # dimension (D)
        self.var = np.var(x_flat, axis=0)   # dimension (D)

        # Normalise the data across the batch by subtracting the mean and scaling by the std
        self.x_mu = x_flat - self.mu
        self.var_inv = np.sqrt(self.var + self.epsilon)
        self.x_norm = (self.x_mu) / self.var_inv

        # use learned parameters to determine how much we want to normalise the data
        # if gamma equals the std and beta equals the mean then this layer has no affect
        output = self.gamma * self.x_norm + self.beta

        # reshape the output so it matches the input
        return np.reshape(output, x.shape)

    # FIXME: I will need to draw out the computation graph for this or I'll forget where these calculations come from
    def backward_pass(self, da, learning_rate):
        """ Computes the backward pass of the batch normalisation layer and updates gamma and beta

        da (numpy array): gradients of the activations passed up from the next downstream layer
        """

        # reshape the gradients to make the maths easier
        N = da.shape[0]                                                 # number of samples in batch
        da_flat = np.reshape(da, (N, np.product(da.shape[1:])))         # each sample flattened to a single dimension

        # calculate the derivatives of gamma and beta
        dbeta = (1.0/N) * np.sum(da_flat, axis=0)                                 # sum across N since same beta is applied to all samples
        dgamma = (1.0/N) * np.sum(self.x_norm * da_flat, axis=0)                  # sum across N since same gamma is applied to all samples

        # calculate the normalised data gradients
        dx_norm = self.x_norm * da_flat                                 # don't sum since each x_norm element is used only once

        # calculate the variance gradients
        dvar = -0.5 * np.sum(dx_norm * self.x_mu, axis=0) * np.power(self.var + self.epsilon, -1.5)

        # calculate the mean gradients
        dmu1 = np.sum(dx_norm * -self.var_inv, axis=0)
        dmu2 = dvar * (1/N) * np.sum(-2 * self.x_mu)
        dmu = dmu1 + dmu2

        # calculate the input gradients (used 3 times in calculations of gradient, mean and norms)
        dx1 = dx_norm * self.var_inv    # x_norm portion
        dx2 = dmu * (1/N)               # mean portion
        dx3 = dvar * (2/N) * self.x_mu  # variance portion
        dx = dx1 + dx2 + dx3

        # Reshape the data to the correct shape
        dx = np.reshape(dx, gradients.shape)

        # Update beta and gamma
        self.beta -= learning_rate * dbeta
        self.gamma -= learning_rate * dgamma

        return dx


class max_pooling_layer:
    pass

class convolutional_layer:
    pass
