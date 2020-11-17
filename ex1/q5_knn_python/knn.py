import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    pos = np.arange(-5, 5.0, 0.1)
    n = samples.shape[0]
    val = [k/(2*n*(np.linalg.norm(np.sort(np.abs(samples-i))[k-1]))) for i in pos]
    # Compute the number of samples created
    return np.stack([pos,val]).transpose()
