import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    return np.sum([np.log(gauss_mix(x, means, weights, covariances)) for x in X])

def gauss_mix(x, means, weights, covariances):
    return np.sum([weights[k]*gaussian(x,means[k],covariances[:,:,k]) for k in range(len(weights))])

def gaussian(x,mu,sigma):
    d = sigma.shape[0]
    factor = 1 / np.sqrt((np.power(2 * np.pi, d) * np.linalg.det(sigma)))
    return factor * np.exp((-1 / 2 * np.dot((x-mu),np.dot(np.linalg.inv(sigma),(x-mu)))))

