import numpy as np

def initialize_parameters(layer_dims):
    np.random.seed(42)  # for reproducibility
    parameters = {}
    L = len(layer_dims)  # number of layers

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)  # avoid overflow
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    activation_cache = Z
    return A, activation_cache

def relu(Z):
    A = np.maximum(0, Z)
    activation_cache = Z
    return A, activation_cache

def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    else:
        raise ValueError("Unsupported activation function: choose 'relu' or 'softmax'")

    cache = (linear_cache, activation_cache)
    return A, cache
