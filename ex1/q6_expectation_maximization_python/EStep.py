import numpy as np
from getLogLikelihood import getLogLikelihood, gaussian, gaussian_mix


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

    for i in range(n):
        for j in range(k):
            gamma[i,j] = (weights[j]*gaussian(X[i,:],means[j],covariances[:,:,j]))/gaussian_mix(X[i,:],means,weights,covariances)

    return [getLogLikelihood(means,weights,covariances,X), gamma]
