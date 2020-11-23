import numpy as np
from matplotlib import pyplot as plt
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    n_weights, n_means, n_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)
    s_weights, s_means, s_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    #print(img.shape[0])
    #print(n_loglikelihood/s_loglikelihood)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            n_loglikelihood = getLogLikelihood(n_weights, n_means, n_covariances, img[i][j])
            s_loglikelihood = getLogLikelihood(s_weights, s_means, s_covariances, img[i][j])
            #print(s_loglikelihood / n_loglikelihood)
            if s_loglikelihood/n_loglikelihood > theta:
                img[i][j] = 255
            else:
                img[i][j] = 0
    result = img
    return result
