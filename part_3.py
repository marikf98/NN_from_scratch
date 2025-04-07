import numpy as np

from part_1 import initialize_parameters, l_model_forward, compute_cost, softmax
from part_2 import update_parameters, l_model_backward


# added use_batchnorm param to turn on and off the flag, need to ask in forum if its okay
def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size=64, use_batchnorm=False ):

    parameters = initialize_parameters(layers_dims)
    costs = []

    m = X.shape[1]  # number of examples

    for i in range(num_iterations):
        for j in range(0, m, batch_size):
            # Create mini-batch
            X_batch = X[:, j:j+batch_size]
            Y_batch = Y[:, j:j+batch_size]

            # Forward pass
            AL, caches = l_model_forward(X_batch, parameters, use_batchnorm=use_batchnorm)

            # Compute cost
            cost = compute_cost(AL, Y_batch)

            # Backward pass
            grads = l_model_backward(AL, Y_batch, caches)

            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

        # Save cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost:.4f}")

    return parameters, costs

def predict(X, Y, parameters):

    AL, _ = l_model_forward(X, parameters, use_batchnorm=False)

    # Softmax normalization
    probs, _ = softmax(AL)

    # Predictions: class with highest probability
    predictions = np.argmax(probs, axis=0)
    labels = np.argmax(Y, axis=0)

    # Compute accuracy
    accuracy = np.mean(predictions == labels) * 100

    return accuracy
