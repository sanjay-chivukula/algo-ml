import numpy as np


def activation_selector(activation_function_name: str):
    activation_function_dict = {
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu,
        'leaky_relu': leaky_relu,
        'elu': elu,
        'swish': swish,
    }
    if activation_function_name not in activation_function_dict.keys():
        raise KeyError
    return activation_function_dict[activation_function_name]


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
