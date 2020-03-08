import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    numerator = 1 - np.exp(-2 * x)
    denominator = 1 + np.exp(-2 * x)
    return numerator / denominator


def relu(x):
    return max(0, x)


def leaky_relu(x, alpha=0.01):
    if x < 0:
        return alpha * x
    else:
        return x


def elu(x, alpha=0.01):
    if x < 0:
        return alpha * (np.exp(x) - 1)
    else:
        return x


def swish(x, beta):
    return 2 * x * sigmoid(beta * x)
