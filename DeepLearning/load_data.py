
import numpy as np

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100


MNIST_DATA_SIZE = 60000
MNIST_CLASSES = 10

CIFAR10_DATA_SIZE = 50000
CIFAR10_CLASSES = 10

CIFAR100_DATA_SIZE = 50000
CIFAR100_CLASSES = 100

def vectorise_labels(labels, num_classes):
    """
    given a list of labels converts this to a list of vectors
    where the length of the vector is the number of classes and
    all elements are zero except the elments at the index of the correct class
    """

    label_vectors = np.zeros((labels.size, num_classes))
    for ind, label in enumerate(labels):
        label_vectors[ind][int(label)] = 1.0

    return label_vectors


def reshape_data(data, number_samples, number_classes, flatten_data):
    """ Reshapes the data loaded by keras into a usable form

    data           (numpy array): The data loaded by keras
    number_samples (int):         Number of images to load
    number_classes (int):         Number of classes the data represents
    flatten_data   (boolean):     Set to True to flatten the input data to a single dimension
    """

    (trainX, trainY), (testX, testY) = data

    # truncate the training data (useful when debugging)
    if trainX.shape[0] > number_samples:
        trainX = trainX[:number_samples]
        trainY = trainY[:number_samples]

    # vectorise the input data
    if flatten_data:
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))

    # vectorise the input output data
    testY = vectorise_labels(testY, number_classes)
    trainY = vectorise_labels(trainY, number_classes)

    return trainX, trainY, testX, testY


def load_mnist_data(number_samples=MNIST_DATA_SIZE, flatten_data=False):
    """
    Loads the mnist data using the default keras method
    While this method is very quick there is additional processing required
    to vectorize the labels and reshape the images into vectors

    number_samples (int):     Number of images to load
    flatten_data   (boolean): Set to True to flatten the input data to a single dimension
    """

    # load the mnist data using the default keras load method
    data = mnist.load_data(path='mnist.npz')
    return reshape_data(data, number_samples, MNIST_CLASSES, flatten_data)


def load_cifar10_data(number_samples=CIFAR10_DATA_SIZE, flatten_data=False):
    """
    Loads the cifar10 dataset using the default keras method
    performs additional processing to present data in appropriate shape

    number_samples (int):     Number of images to load
    flatten_data   (boolean): Set to True to flatten the input data to a single dimension
    """

    data = cifar10.load_data()
    return reshape_data(data, number_samples, CIFAR10_CLASSES, flatten_data)


def load_cifar100_data(number_samples=CIFAR100_DATA_SIZE, flatten_data=False):
    """
    Loads the cifar100 dataset using the default keras method
    performs additional processing to present data in appropriate shape

    number_samples (int):     Number of images to load
    flatten_data   (boolean): Set to True to flatten the input data to a single dimension
    """


    data = cifar100.load_data()
    return reshape_data(data, number_samples, CIFAR100_CLASSES, flatten_data)
