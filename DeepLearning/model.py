"""
Implements a high level model class

1. Stores all the layers for the network
2. Performs forward pass of network
3. Calculates total network loss
4. Performs backward pass of network
5. Prints statistics on networks performance

We first create a model and then we add layers to it
"""

import numpy as np

class Model(object):
    """ Class used to represent an entire neural network """

    def __init__(self, layers, loss_function, l1=None, l2=None):
        """

        layers        (list):     list of layer objects, network will be in the order the layers are in
        loss_function (function): Function used to compute the loss of the network
        l1:           (float):    Lambda for L1 regularisation (encourages sparse weight matrix)
        l2:           (float):    Lambda for L2 regularisation (weight decay, encourages small weights)
        """

        self.layers = layers
        self.loss_function = loss_function
        self.l1 = l1
        self.l2 = l2

    def forward_pass(self, activations):
        """ Runs the forward pass of the network and returns the activations

        activations (numpy array): The input data to the neural network
        """

        for layer in self.layers:
            activations = layer.forward_pass(activations)       # pass in the activations (output) of the previous layer

        return activations

    def backward_pass(self, gradients):
        """ Runs the backward pass of the network and returns the upstream gradients
        Updates to the learning parameters are handled by the layer instances

        gradients (numpy array): The gradients of the loss function w.r.t the actiavtions of the final layer
        """

        for layer in reversed(self.layers):
            gradients = layer.backward_pass(gradients)

        return gradients

    # TODO: Need to modify this so it will work with the L1 and L2 regularisation
    # TODO: Some loss functions (huber) take extra parameters, will need to modify this so they can be passed in
    def calculate_loss(self, activations, labels):
        """ Calculate the loss of the network and the gradients w.r.t the loss function

        activations (numpy array): The activations from the final layer
        labels      (numpy array): The correct labels for the output of the final layer (should be one hot encoded)
        """

        # add code here for calculating the regularisation term
        regularisation = 0

        loss, gradients = self.loss_function(activations, labels)
        return loss + regularisation, gradients

    # FIXME: At the moment assumes single class classification
    def evaluate(self, data, labels):
        """ Evaluates the performance of the network by passing all the input data through the forward pass

        data       (numpy array): Data to be passed into the neural network. Should have samples as the first dimension
        labels     (numpy array): Labels for the data being passed into the network. Should be one hot encoded
        """

        activations = self.forward_pass(data)
        loss, _ = self.calculate_loss(activations, labels)

        correct = np.sum(np.argmax(activations, axis=1) == np.argmax(labels, axis=1))       # number of correctly classifed samples
        acc = (float(correct) / float(data.shape[0])) * 100.0                               # accuracy as a precentage

        return loss, acc

    def train_network(self, data, labels, epochs, batch_size, verbose=False, val_data=None, val_labels=None):
        """ Trains the network by performing the forward and backward pass on the network a number of times

        data       (numpy array): Numpy array containing the training data. Samples should be first dimension of the array
        labels     (numpy array): Labels for the training data. One hot encoded and samples should be the first dimension of the array
        epochs     (int):         Number of times to cycle through the entire training dataset
        batch_size (int):         Number of samples in each batch used to approximate gradients of the training data
        verbose    (boolean):     If true will calculate the precentage accuary of the trainig and validation data after each epoch
        val_data   (numpy array): Numpy array containing the validation input data
        val_labels (numpy array): Labels for the validation data. Should be one hot encoded
        """

        # initalise lists to store the training and validation loss for each epoch
        train_loss = list()
        train_acc = list()
        validation_loss = list()
        validation_acc = list()

        for epoch in range(epochs):

            # TODO: create the get_batches function
            batches = get_batches(data, labels, batch_size)

            # perform the forward and backward pass for each batch
            for batch_data, batch_labels in batches:

                activations = self.forward_pass(batch_data)
                loss, gradients = self.calculate_loss(activations, batch_labels)
                final_gradients = self.backward_pass(gradients)

            # calculate the training and validation accuracy after each epoch
            if verbose:

                if val_data is None or val_labels is None:
                    raise ValueError("Can't Evaluate network performance as validation data not provided")

                t_loss, t_accuracy = self.evaluate(data, labels)
                v_loss, v_accuracy = self.evaluate(val_data, val_labels)
                print("Epoch %.4i: train_loss: %.4f, train_acc: %.4f %%, val_loss: %.4f, val_acc: %.4f %%" %
                      (epoch, t_loss, t_accuracy, v_loss, v_accuracy))

                train_loss.append(t_loss)
                train_acc.append(t_accuracy)
                validation_loss.append(v_loss)
                validation_acc.append(v_accuracy)

        return train_loss, train_acc, validation_loss, validation_acc
