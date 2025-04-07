import numpy as np


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)

    else:
        raise ValueError("Unsupported activation function. Use 'relu' or 'softmax'.")

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def relu_backward(dA, activation_cache):

    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(dA, activation_cache):
    return dA

def l_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # number of layers

    # Step 1: Backprop the output layer once with softmax
    dAL = AL - Y  # derivative of loss softmax
    current_cache = caches[L - 1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, activation="softmax")

    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # Step 2: Loop through hidden layers (from L-1 to 1), using relu
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_curr = grads["dA" + str(l + 1)]  # from previous layer
        dA_prev, dW, db = linear_activation_backward(dA_curr, current_cache, activation="relu")

        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters

