import numpy as np
from kern import kern
from cvxopt import matrix, solvers


def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                        (num_samples x dim)
    # t        : labeling                           (num_samples x 1)
    # C        : penalty factor the slack variables (scalar)
    # p        : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv       : support vectors (boolean)          (1 x num_samples)
    # b        : bias of the classifier             (scalar)
    # slack    : points inside the margin (boolean) (1 x num_samples)

    n = X.shape[0]
    K = kern(X.T,X.T,2)
    #Why would you program the kernel in such a weird way if dataset is (num_samples x dim)
    #You would expect it to give (num_samples x num_samples) weird double transposes
    P = matrix(np.matmul(t[:, np.newaxis], t[np.newaxis, :]) * K)
    q = matrix(np.ones((n, 1)) * -1)
    A = matrix(t.reshape(1, -1))
    b = matrix(np.zeros(1))

    G = matrix(np.vstack([np.eye(n) * -1, np.eye(n)]))
    h = matrix(np.hstack([np.zeros(n), C * np.ones(n)]))

    solution = solvers.qp(P, q, G, h, A, b)
    alpha = np.squeeze(solution['x'])

    sv = alpha > 1e-4
    w = np.dot(alpha * t, X)
    b = np.mean(t[sv] - np.sum(np.dot(X[sv], X[sv].T) * alpha[sv] * t[sv], axis=1))
    result = (X @ w) + b
    slack = (result < 1) & (result > -1)
    return alpha, sv, b, result, slack
