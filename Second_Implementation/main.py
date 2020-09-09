from Second_Implementation.util.utilities import *
from Second_Implementation.util.cost_functions import *
from Second_Implementation.Layers.LinearLayer import LinearLayer
from Second_Implementation.Layers.SigmoidLayer import SigmoidLayer
from Second_Implementation.Layers.ReLULayer import ReLULayer

# import MNIST dataset
from MNIST_for_Numpy import mnist
# import tensorflow for normalization
import tensorflow as tf

if __name__ == '__main__':

    # ---------- Setting up the data ----------

    # -------> MNIST data
    mnist.init()
    X_train, Y_train, X_test, Y_test = mnist.load()

    # reducing size of dataset by a half
    train_size = int(X_train.shape[0])
    test_size = int(X_test.shape[0])

    X_train = X_train[:train_size].T
    X_test = X_test[:test_size].T
    # reshape Y data to fit the network
    Y_train = np.reshape(Y_train[:train_size].T, (1, train_size))
    Y_test = np.reshape(Y_test[:test_size].T, (1, test_size))

    # # normalizing data
    # X_train = tf.keras.utils.normalize(X_train, axis=1)
    # X_test = tf.keras.utils.normalize(X_test, axis=1)

    # -------> XOR gate data
    #
    # inputs
    # X = np.array([
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1]
    # ])
    #
    # # expected outputs
    # Y = np.array([
    #     [0],
    #     [1],
    #     [1],
    #     [0]
    # ])
    #
    # X_train = X.T
    # Y_train = Y.T

    # ---------- Defining hyper-parameters ----------

    # define training constants
    learning_rate = 1
    number_of_epochs = 25

    # set seed value so that the weight initialization results are reproducible
    np.random.seed(48)

    # ---------- Network architecture ----------

    # Layer 1: hidden layer that takes in training data with 128 neurons
    Z1 = LinearLayer(input_shape=X_train.shape, n_out=128)
    A1 = SigmoidLayer(Z1.Z.shape)

    # Layer 2: hidden layer that takes in previous layer's data with 64 neurons
    Z2 = LinearLayer(input_shape=A1.A.shape, n_out=64)
    A2 = SigmoidLayer(Z2.Z.shape)

    # Layer 3: output layer that take is values from hidden layer
    Z3 = LinearLayer(input_shape=A2.A.shape, n_out=10)
    A3 = SigmoidLayer(Z3.Z.shape)

    # This will store all the costs after a certain number of epochs
    costs = []

    # Start training
    for epoch in range(number_of_epochs):

        # ---------- Forward pass ----------

        # perform linear calculations
        Z1.forward(X_train)
        # perform activation on results
        A1.forward(Z1.Z)

        # perform linear calculations
        Z2.forward(A1.A)
        # perform activation on results
        A2.forward(Z2.Z)

        # perform linear calculations
        Z3.forward(A2.A)
        # perform activation on results
        A3.forward(Z3.Z)

        # ---------- Compute cost ----------
        # A3.A => final outputs => predictions
        Y_train_one_hot = tf.keras.utils.to_categorical(Y_train)
        cost, dA3 = compute_mse_cost(Y=Y_train, Y_hat=A3.A)

        # print and store Costs every 100 iterations.
        if (epoch % 2) == 0:
            print("Cost at epoch#{}: {}".format(epoch, cost))
            costs.append(cost)

        # ---------- Backward pass ----------
        A3.backward(dA3)
        Z3.backward(A3.dZ)

        A2.backward(Z3.dA_prev)
        Z2.backward(A2.dZ)

        A1.backward(Z2.dA_prev)
        Z1.backward(A1.dZ)

        # ---------- Update weights and biases ----------
        Z3.update_params(learning_rate=learning_rate)
        Z2.update_params(learning_rate=learning_rate)
        Z1.update_params(learning_rate=learning_rate)

    # see the output predictions and discard the non-binary results
    predicted_outputs, _, accuracy = predict_reg(X=X_test, Y=Y_test,
                                                 Zs=[Z1, Z2, Z3],
                                                 As=[A1, A2, A3])

    print("The predicted outputs:\n {}".format(predicted_outputs))
    print("The accuracy of the model is: {}%".format(accuracy))

    plot_learning_curve(costs=costs, learning_rate=learning_rate,
                        total_epochs=number_of_epochs)

    # plot_decision_boundary(lambda x:predict_dec(Zs=[Z1, Z2], As=[A1, A2],
    # X=x.T), X_train.T, Y_train.T)
