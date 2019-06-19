"""
This modules implements the optimisation algorihms used by the neural network layers to updates
learning parameters (i.e. weights and biases)

"""


class SGD(object):
    """ Vanilla stochastic gradient descent

    Updates weights in the direction of greatest gradient based on learning rate
    """

    def __init__(self, parameters):
        """ Initalise the learning rate for the SGD algorithm

        parameters: (dict): dictionary of parameters used to initalise the SGD algorithm
            - learning_rate (float): The rate that parameters will be updated at
        """

        if 'learning_rate' not in parameters.keys():
            raise KeyError("learning_rate not specified for SGD optimiser")

        self.learning_rate = parameters['learning_rate']

    def update(self, gradients):
        """ Calculates how much the learning parameters should be changed by

        gradients     (numpy array): The gradients of the parameters to be updated
        """

        return self.learning_rate * gradients


class SGDWithMomentum(object):
    """ Stochastic gradient descent with moment

    Maintains a velocity vector in the direction of descent
    This velocity vector is update based on the current gradients
    The parameters are then updated in the direction of this velocity
    More resiliant to noise and saddle points then vanilla SGD
    """

    def __init__(self, parameters):
        """ Initalise the velocity vector, learning rate and rho

        parameters: (dict): dictionary of parameters used to initalise the SGD with momentum algorithm
            - learning_rate (float): The rate that parameters will be updated at
            - rho           (float): Rate of decay for SGD with momentum (prevents continuous parameter updates, 0.9)
        """

        if 'learning_rate' not in parameters.keys():
            raise KeyError("learning_rate not specified for SGD with momentum optimiser")

        if 'rho' not in parameters.keys():
            raise KeyError("rho not specified for SGD with momentum optimiser")

        self.learning_rate = parameters['learning_rate']
        self.rho = parameters['rho']
        self.velocity = 0.0

    def update(self, gradients):
        """ Calculates how much the learning parameters should be changed by

        gradients     (numpy array): The gradients of the parameters to be updated
        """

        self.velocity = rho * self.velocity + gradients
        return self.learning_rate * self.velocity


class RMSProp(object):
    """ Implements the RMSProp algorithm, which is an improvement on the Adagrad algorithm

    Keeps a running sum of the square gradients terms, new gradients are divided by this
    The running sum is multipled by a decay constant to prevent the step size from reducing to zero
    Dividing by the sum of the square gradients brings the ratio between dimensions to unity
    """

    def __init__(self, parameters):
        """ Initalise gradients sum, learning rate and delta

        parameters: (dict): dictionary of parameters used to initalise the RMSProp algorithm
            - learning_rate (float): The rate that parameters will be updated at
            - delta         (float): Rate of decay for gradients sum (prevents vanishing step size, 0.9)
        """

        if 'learning_rate' not in parameters.keys():
            raise KeyError("learning_rate not specified for RMSProp optimiser")

        if 'delta' not in parameters.keys():
            raise KeyError("delta not specified for RMSProp optimiser")

        self.learning_rate = parameters['learning_rate']
        self.delta = parameters['delta']
        self.gradient_sum = 0.0

    def update(self, gradients):
        """ Calculates how much the learning parameters should be changed by

        gradients     (numpy array): The gradients of the parameters to be updated
        """

        self.gradient_sum = self.delta * self.gradient_sum + ((1 - delta) * np.square(gradients))
        return (self.learning_rate * gradients) / (np.sqrt(self.gradient_sum) + 1e-7)


class Adam(object):
    """ ADAM optimisation algorithm

    """

    def __init__(self, parameters):
        """ Initalises the first and second moments, the learning rate and beta1 and beta2

        parameters: (dict): dictionary of parameters used to initalise the ADAM algorithm
            - learning_rate (float): The rate that parameters will be updated at (1e-3)
            - beta1         (float): Rate of decay for the first moment (velocity 0.9)
            - beta2         (float): Rate of decay for the second moment (gradient sum 0.999)
        """

        if 'learning_rate' not in parameters.keys():
            raise KeyError("learning_rate not specified for RMSProp optimiser")

        if 'beta1' not in parameters.keys():
            raise KeyError("beta1 not specified for Adam optimiser")

        if 'beta2' not in parameters.keys():
            raise KeyError("beta2 not specified for Adam optimiser")

        self.learning_rate = parameters['learning_rate']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']

        self.first_moment = 0.0
        self.second_moment = 0.0
        self.t = 0.0                        # itteration counter

    def update(self, gradients):
        """

        gradients     (numpy array): The gradients of the parameters to be updated
        """

        self.first_moment = self.beta1 * self.first_moment + (1 - self.beta1) * gradients                   # velocity builds slower than SGD case
        self.second_moment = self.beta2 * self.second_moment + ((1 - self.beta2) * np.square(gradients))    # Same as RMSProp

        # divide by a very small number so that the second moment will be large, this will prevent
        # the step size for the first few itterations from being to big
        unbiased_first = first_moment / (1 - np.power(self.beta1, self.t))
        unbiased_second = first_moment / (1 - np.power(self.beta1, self.t))

        # increasee the itteration counter
        self.t += 1

        return (self.learning_rate * self.first_moment) / (np.sqrt(self.second_moment) + 1e-7)
