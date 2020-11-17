import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    pos = np.arange(-5, 5.0, 0.1)
    factor = 1 / (np.power(2 * np.pi, 1 / 2) * h)
    val = [factor * np.mean(np.exp((-1/(2*h**2)*(samples-i)**2))) for i in pos]
    # Compute the number of samples created
    return np.stack([pos,val]).transpose()
