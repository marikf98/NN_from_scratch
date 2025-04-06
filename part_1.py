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


def l_model_forward(X, parameters, use_batchnorm):
    caches = []
    L = len(parameters) // 2
    A = X # input layer A[0] = X
    for i in range(1, L):
        A_prev = A
        W = parameters[f"W{i}"]
        b = parameters[f"b{i}"]
        A, cache = linear_activation_forward(A_prev, W, b, activation="relu")
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)

    W = parameters[f"W{L}"]
    b = parameters[f"b{L}"]
    AL, cache = linear_activation_forward(A, W, b, activation="softmax")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-15, 1 - 1e-15) # avoid log(0)
    summation = np.sum(np.multiply(Y, np.log(AL)))
    cost = - (1/m) * summation
    return cost

def apply_batchnorm(A):
    m = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    epsilon = 1e-8
    A_tag = (A - m) / np.sqrt(var + epsilon)
    return A_tag

