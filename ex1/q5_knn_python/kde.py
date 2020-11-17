import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    pos = np.arange(-5, 5.0, 0.1)
    val = np.random.normal(0, 1, 100)
    # Compute the number of samples created
    return np.stack([pos,val]).transpose()
