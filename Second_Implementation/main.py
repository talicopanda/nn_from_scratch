import numpy as np
from Second_Implementation.util.utilities import *
from Second_Implementation.util.cost_functions import *
from Second_Implementation.Layers.LinearLayer import LinearLayer
from Second_Implementation.Layers.SigmoidLayer import SigmoidLayer


if __name__ == '__main__':

    # ---------- Setting up the data ----------

    # XOR gate data

    # inputs
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # expected outputs
    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    X_train = X.T
    Y_train = Y.T

    # ---------- Defining hyper-parameters ----------

    # define training constants
    learning_rate = 1
    number_of_epochs = 5000

    # set seed value so that the weight initialization results are reproduceable
    np.random.seed(48)

    # ---------- Network architecture ----------

    # Layer 1: hidden layer that takes in training data with 3 neurons
    Z1 = LinearLayer(input_shape=X_train.shape, n_out=3, ini_type='xavier')
    A1 = SigmoidLayer(Z1.Z.shape)

    # Layer 2: output layer that take is values from hidden layer
    Z2 = LinearLayer(input_shape=A1.A.shape, n_out=1, ini_type='xavier')
    A2 = SigmoidLayer(Z2.Z.shape)

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

        # ---------- Compute cost ----------
        # A2.A => final outputs => predictions
        cost, dA2 = compute_mse_cost(Y=Y_train, Y_hat=A2.A)

        # print and store Costs every 100 iterations.
        if (epoch % 100) == 0:
            print("Cost at epoch#{}: {}".format(epoch, cost))
            costs.append(cost)

        # ---------- Backward pass ----------
        A2.backward(dA2)
        Z2.backward(A2.dZ)

        A1.backward(Z2.dA_prev)
        Z1.backward(A1.dZ)

        # ---------- Update weights and biases ----------
        Z2.update_params(learning_rate=learning_rate)
        Z1.update_params(learning_rate=learning_rate)

    # see the output predictions and discard the non-binary results
    predicted_outputs, _, accuracy = predict(X=X_train, Y=Y_train, Zs=[Z1, Z2], As=[A1, A2])

    print("The predicted outputs:\n {}".format(predicted_outputs))
    print("The accuracy of the model is: {}%".format(accuracy))

    plot_learning_curve(costs=costs, learning_rate=learning_rate, total_epochs=number_of_epochs)

    # plot_decision_boundary(lambda x:predict_dec(Zs=[Z1, Z2], As=[A1, A2],
    # X=x.T), X_train.T, Y_train.T)
