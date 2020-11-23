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

    n_loglikelihood = np.empty(img.shape)
    s_loglikelihood = np.empty(img.shape)
    prec,recall,f_1 = quality(ndata,sdata,theta,n_weights,n_means,n_covariances,s_weights,s_means,s_covariances)
    print('Training data Precision:\t{}\t Recall: {} \t F_1: {}'.format(prec,recall,f_1))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            n_loglikelihood[i][j] = getLogLikelihood(n_weights, n_means, n_covariances, img[i][j])
            s_loglikelihood[i][j] = getLogLikelihood(s_weights, s_means, s_covariances, img[i][j])
            #print(s_loglikelihood / n_loglikelihood)
            #if s_loglikelihood/n_loglikelihood > theta:
            #    img[i][j] = 255
            #
            #else:
            #    img[i][j] = 0
    idx = s_loglikelihood/n_loglikelihood > theta
    img[idx] = 255
    img[np.logical_not(idx)] = 0
    result = img
    return result

def quality(ndata,sdata,theta,n_weights, n_means, n_covariances,s_weights, s_means, s_covariances):
    classifications = classify(np.vstack([sdata, ndata]), theta, n_weights, n_means, n_covariances, s_weights, s_means, s_covariances)
    skin = classifications[0:sdata.shape[0],1]
    noSkin = classifications[sdata.shape[0]+1:-1, 1]

    num_true_pos = len(np.where(skin == 1)[0])
    num_true_neg = len(np.where(noSkin == 0)[0])

    precision = (num_true_pos)/(num_true_pos+(ndata.shape[0]-num_true_neg))
    recall = (num_true_pos)/(sdata.shape[0])
    f_1 = 2*(precision*recall)/(precision+recall)

    return precision,recall,f_1

def classify(data,theta,n_weights, n_means, n_covariances,s_weights, s_means, s_covariances):
    n_loglikelihood = np.empty(data.shape)
    s_loglikelihood = np.empty(data.shape)
    for i in range(data.shape[0]):
            n_loglikelihood[i] = getLogLikelihood(n_weights, n_means, n_covariances, data[i])
            s_loglikelihood[i] = getLogLikelihood(s_weights, s_means, s_covariances, data[i])
            # print(s_loglikelihood / n_loglikelihood)
            # if s_loglikelihood/n_loglikelihood > theta:
            #    img[i][j] = 255
            #
            # else:
            #    img[i][j] = 0
    idx = s_loglikelihood / n_loglikelihood > theta
    data[idx] = 1
    data[np.logical_not(idx)] = 0
    return data