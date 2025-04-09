import part_1, part_2, part_3
import time

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

def get_dataset():
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    # scale image values to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Reshape from 2d to 1d
    X_train = X_train.reshape(-1, 28 * 28).T
    X_val = X_val.reshape(-1, 28 * 28).T
    X_test = X_test.reshape(-1, 28 * 28).T

    # create a one hot encoded matrix and transpose it to the matching shape of the softmax output
    Y_train = to_categorical(y_train).T  # shape: (10, num_train_examples)
    Y_val = to_categorical(y_val).T
    Y_test = to_categorical(y_test).T

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def train_model_section_4(X_train, Y_train, X_val, Y_val, X_test, Y_test, layers, learning_rate, batch_size, epochs):
    max_val_accuracy = 0
    no_improve_count = 0
    num_steps_without_improvement = 100
    step_counter = 0
    small_improvement = 1e-4
    parameters = part_1.initialize_parameters(layers)
    steps_in_epoch = X_train.shape[1] // batch_size
    costs = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for step in range(steps_in_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch = X_train[:, start:end]
            Y_batch = Y_train[:, start:end]

            # Forward propagation
            AL, caches = part_1.l_model_forward(X_batch, parameters, use_batchnorm=False)

            # Compute cost
            cost = part_1.compute_cost(AL, Y_batch)


            # Backward propagation
            grads = part_2.l_model_backward(AL, Y_batch, caches)

            # Update parameters
            parameters = part_2.update_parameters(parameters, grads, learning_rate)

            step_counter += 1

            if step_counter % 100 == 0:
                print(f"Cost after iteration {step_counter}: {cost:.4f}")
                costs.append(cost)

            if step_counter % 10 == 0: # every 10 mini batches evaluate the model on the validation set
                val_accuracy = part_3.predict(X_val, Y_val, parameters)

                if val_accuracy > max_val_accuracy + small_improvement:
                    max_val_accuracy = val_accuracy
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # print(f"Step {step_counter} | Cost: {cost:.4f} | Val Acc: {val_accuracy:.2f}%")
                if no_improve_count > num_steps_without_improvement:
                    print("Early stop no improvement on the validation")
                    break
        if no_improve_count > num_steps_without_improvement:
            break
    print("Training finished.")
    train_accuracy = part_3.predict(X_train, Y_train, parameters)
    val_accuracy = part_3.predict(X_val, Y_val, parameters)
    test_accuracy = part_3.predict(X_test, Y_test, parameters)

    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")

def section_4():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_dataset()
    # Initialize parameters first layer is the input with size 784 (flattened 28x28)
    layers = [784, 20, 7, 5, 10]
    learning_rate = 0.009
    batch_size = 64
    epochs = 500
    train_model_section_4(X_train, Y_train, X_val, Y_val, X_test, Y_test, layers, learning_rate, batch_size, epochs)
# Press the green button in the gutter to run the script.
def train_model_section_5(X_train, Y_train, X_val, Y_val, X_test, Y_test, layers, learning_rate, batch_size, epochs):
    max_val_accuracy = 0
    no_improve_count = 0
    num_steps_without_improvement = 100
    step_counter = 0
    small_improvement = 1e-4
    parameters = part_1.initialize_parameters(layers)
    steps_in_epoch = X_train.shape[1] // batch_size
    costs = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for step in range(steps_in_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch = X_train[:, start:end]
            Y_batch = Y_train[:, start:end]

            # Forward propagation
            AL, caches = part_1.l_model_forward(X_batch, parameters, use_batchnorm=True)

            # Compute cost
            cost = part_1.compute_cost(AL, Y_batch)


            # Backward propagation
            grads = part_2.l_model_backward(AL, Y_batch, caches)

            # Update parameters
            parameters = part_2.update_parameters(parameters, grads, learning_rate)

            step_counter += 1

            if step_counter % 100 == 0:
                print(f"Cost after iteration {step_counter}: {cost:.4f}")
                costs.append(cost)

            if step_counter % 10 == 0: # every 10 mini batches evaluate the model on the validation set
                val_accuracy = part_3.predict(X_val, Y_val, parameters, use_batchnorm=True)

                if val_accuracy > max_val_accuracy + small_improvement:
                    max_val_accuracy = val_accuracy
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # print(f"Step {step_counter} | Cost: {cost:.4f} | Val Acc: {val_accuracy:.2f}%")
                if no_improve_count > num_steps_without_improvement:
                    print("Early stop no improvement on the validation")
                    break
        if no_improve_count > num_steps_without_improvement:
            break
    print("Training finished.")
    train_accuracy = part_3.predict(X_train, Y_train, parameters, use_batchnorm=True)
    val_accuracy = part_3.predict(X_val, Y_val, parameters, use_batchnorm=True)
    test_accuracy = part_3.predict(X_test, Y_test, parameters, use_batchnorm=True)

    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
def section_5():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_dataset()
    # Initialize parameters first layer is the input with size 784 (flattened 28x28)
    layers = [784, 20, 7, 5, 10]
    learning_rate = 0.009
    batch_size = 64
    epochs = 500
    train_model_section_5(X_train, Y_train, X_val, Y_val, X_test, Y_test, layers, learning_rate, batch_size, epochs)


def main():
    print("Section 4 - Without batchnorm")
    print('\n')
    start_time_4 = time.time()
    section_4()
    elapsed_time_4 = time.time() - start_time_4
    print("Elapsed time for section 4: {:.2f} seconds".format(elapsed_time_4))
    print("\nSection 5 - With batchnorm")
    print('\n')
    start_time_5 = time.time()
    section_5()
    elapsed_time_5 = time.time() - start_time_5
    print("Elapsed time for section 5: {:.2f} seconds".format(elapsed_time_5))
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
