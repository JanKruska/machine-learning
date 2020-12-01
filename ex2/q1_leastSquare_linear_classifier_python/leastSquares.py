import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)

    data_mean = np.mean(data)
    label_mean = np.mean(label)

    num = den = 0

    for i in range(len(data)):
        num += (data[i] - data_mean) * (label[i] - label_mean)
        den += (data[i] - data_mean)**2

    weight = num/den
    bias = label_mean - (weight * data_mean)

    return weight, bias
