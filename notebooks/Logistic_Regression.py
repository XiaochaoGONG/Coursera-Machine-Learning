import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y):
    m = y.shape[0]
    z = X.dot(theta.reshape(-1, 1))
    h = sigmoid(z)

    cost = y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))
    cost = - cost / float(m)
    return cost

def gradientFunction(theta, X, y):
    m = y.shape[0]
    z = X.dot(theta.reshape(-1, 1))
    h = sigmoid(z)

    grad = X.T.dot(h - y) / float(m)
    return grad.flatten()

def predict(theta, X):
    Z = X.dot(theta)
    r = [1 if z > 0 else 0 for z in Z]
    r = np.c_[r]
    return r
