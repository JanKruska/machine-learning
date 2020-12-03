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
    data = np.hstack((np.ones((data.shape[0], 1)),data))
    data_mean = np.mean(data,axis=0)
    label_mean = np.mean(label)

    num = np.dot(np.transpose(data-data_mean[np.newaxis,:]),label - label_mean)
    den = np.diagonal(np.dot((data-data_mean[np.newaxis,:]).T,data-data_mean[np.newaxis,:]))

    weight = num[1:]/den[1:]
    bias = label_mean - np.dot(weight,data_mean[1:])

    return weight, bias
