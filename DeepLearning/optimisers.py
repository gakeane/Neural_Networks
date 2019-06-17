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
        """ Calculates how much the learning parameters should be chnaged by

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
            - rho           (float): Rate of decay for SGD with momentum (prevents continuous parameter updates)
        """

        if 'learning_rate' not in parameters.keys():
            raise KeyError("learning_rate not specified for SGD with momentum optimiser")

        if 'rho' not in parameters.keys():
            raise KeyError("rho not specified for SGD with momentum optimiser")

        self.learning_rate = parameters['learning_rate']
        self.rho = parameters['rho']
        self.velocity = 0.0

    def update(self, gradients):
        """ Calculates how much the learning parameters should be chnaged by

        gradients     (numpy array): The gradients of the parameters to be updated
        """

        self.velocity = rho * self.velocity + gradients
        return self.learning * self.velocity


class Adam(object):
    """ ADAM optimisation algorithm

    """

    def __init__(self, parameters):
        """

        parameters: (dict): dictionary of parameters used to initalise the ADAM algorithm
        """

    def update(self, gradients):
        """

        gradients     (numpy array): The gradients of the parameters to be updated
        """
