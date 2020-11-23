import numpy as np
from getLogLikelihood import getLogLikelihood, gaussian, gaussian_mix
from multiprocessing import Pool
import itertools

#Serialized EStep
def _EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    n = X.shape[0]
    k = len(weights)
    gamma = np.empty([n, k])
    for i in range(n):
        for j in range(k):
            gamma[i,j] = (weights[j]*gaussian(X[i,:],means[j],covariances[:,:,j]))/gaussian_mix(X[i,:],means,weights,covariances)

    return [getLogLikelihood(means, weights, covariances, X), gamma]


#===============Parallel Estep=====================

PERSISTENT_DATA = {}

def fgamma(i, j):
    weights, X, means, covariances = get_persistent_data()
    return (weights[j] * gaussian(X[i, :], means[j], covariances[:, :, j])) / gaussian_mix(X[i, :], means, weights,
                                                                                           covariances)
def init_persistent_data(weights,X,means,covariances):
    PERSISTENT_DATA['weights'] = weights
    PERSISTENT_DATA['X'] = X
    PERSISTENT_DATA['means'] = means
    PERSISTENT_DATA['covariances'] = covariances

def get_persistent_data():
    return PERSISTENT_DATA['weights'],PERSISTENT_DATA['X'],PERSISTENT_DATA['means'],PERSISTENT_DATA['covariances']

def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    n = X.shape[0]
    k = len(weights)
    gamma = np.empty([n,k])
    init_persistent_data(weights,X,means,covariances)

    with Pool() as pl:
        result = pl.starmap(fgamma,itertools.product(range(n),range(k)))

    gamma = np.reshape(result,[n,k])

    return [getLogLikelihood(means,weights,covariances,X), gamma]

