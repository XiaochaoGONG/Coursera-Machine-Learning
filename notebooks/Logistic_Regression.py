import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y):
    m = y.shape[0]
    z = X.dot(theta)
    h = sigmoid(z)

    cost = y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))
    cost = - cost / float(m)

    grad = X.T.dot(h - y) / float(m)
    return cost, grad
