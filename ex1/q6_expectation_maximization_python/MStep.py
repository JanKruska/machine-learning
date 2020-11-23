import numpy as np
from getLogLikelihood import getLogLikelihood
from regularize_cov import regularize_cov

def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    n_k = np.sum(gamma,0)
    covariances = np.empty([X.shape[1], X.shape[1], gamma.shape[1]])
    means = np.array([np.dot(gamma[:,k],X)/n_k[k] for k in range(gamma.shape[1])])
    for k in range(gamma.shape[1]):
        # elementwise product of (DxN)(N) => DxN dot-product NxD => DxD
        covariances[:, :, k] = np.dot(np.transpose(X - means[k]) * gamma[:, k], X - means[k]) / n_k[k]
        # Regularize to avoid numerical anomalies
        #covariances[:, :, k] = regularize_cov(covariances[:, :, k], 0.0001)
    weights = n_k/X.shape[0]
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood
