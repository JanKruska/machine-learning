import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
from cvxopt import matrix, solvers


def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)

    n = X.shape[0]
    K = np.matmul(X, X.T)
    P = matrix(np.matmul(t[:,np.newaxis], t[np.newaxis,:]) * K)
    q = matrix(np.ones((n, 1)) * -1)
    A = matrix(t.reshape(1, -1))
    b = matrix(np.zeros(1))

    G = matrix(np.vstack([np.eye(n) * -1,np.eye(n)]))
    h = matrix(np.hstack([np.zeros(n),C*np.ones(n)]))
    

    solution = solvers.qp(P, q, G, h, A, b)
    alpha = np.squeeze(solution['x'])

    sv = alpha > 1e-4
    w = np.dot(alpha*t,X)
    b = np.mean(t[sv] - np.sum(np.dot(X[sv],X[sv].T) * alpha[sv] * t[sv], axis=1))
    result = (X @ w) + b
    slack = (result < 1)&(result>-1)
    return alpha, sv, w, b, result, slack
