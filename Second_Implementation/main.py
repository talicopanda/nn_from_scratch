import numpy as np
from Second_Implementation.util.utilities import *
from Second_Implementation.util.cost_functions import *
from Second_Implementation.Layers.LinearLayer import LinearLayer
from Second_Implementation.Layers.SigmoidLayer import SigmoidLayer


if __name__ == '__main__':
    # This is our XOR gate data

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    X_train = X.T
    Y_train = Y.T

    # define training constants
    learning_rate = 1
    number_of_epochs = 5000

    np.random.seed(48) # set seed value so that the results are reproduceable
    # (weights will now be initailzaed to the same pseudo-random numbers, each time)


    # Our network architecture has the shape:
    # (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid] -->(output)

    #------ LAYER-1 ----- define hidden layer that takes in training data
    Z1 = LinearLayer(input_shape=X_train.shape, n_out=3, ini_type='xavier')
    A1 = SigmoidLayer(Z1.Z.shape)

    #------ LAYER-2 ----- define output layer that take is values from hidden layer
    Z2= LinearLayer(input_shape=A1.A.shape, n_out=1, ini_type='xavier')
    A2= SigmoidLayer(Z2.Z.shape)

    costs = [] # initially empty list, this will store all the costs after a certian number of epochs

    # Start training
    for epoch in range(number_of_epochs):

        # ------------------------- forward-prop -------------------------
        Z1.forward(X_train)
        A1.forward(Z1.Z)

        Z2.forward(A1.A)
        A2.forward(Z2.Z)

        # ---------------------- Compute Cost ----------------------------
        cost, dA2 = compute_mse_cost(Y=Y_train, Y_hat=A2.A)

        # print and store Costs every 100 iterations.
        if (epoch % 100) == 0:
            #print("Cost at epoch#" + str(epoch) + ": " + str(cost))
            print("Cost at epoch#{}: {}".format(epoch, cost))
            costs.append(cost)

        # ------------------------- back-prop ----------------------------
        A2.backward(dA2)
        Z2.backward(A2.dZ)

        A1.backward(Z2.dA_prev)
        Z1.backward(A1.dZ)

        # ----------------------- Update weights and bias ----------------
        Z2.update_params(learning_rate=learning_rate)
        Z1.update_params(learning_rate=learning_rate)

    # See what the final weights and bias are training
    # print(Z2.params)
    # print(Z2.params)

    # see the ouptput predictions
    predicted_outputs, _, accuracy = predict(X=X_train, Y=Y_train, Zs=[Z1, Z2], As=[A1, A2])

    print("The predicted outputs:\n {}".format(predicted_outputs))
    print("The accuracy of the model is: {}%".format(accuracy))

    plot_learning_curve(costs=costs, learning_rate=learning_rate, total_epochs=number_of_epochs)


    plot_decision_boundary(lambda x:predict_dec(Zs=[Z1, Z2], As=[A1, A2], X=x.T), X_train.T, Y_train.T)
