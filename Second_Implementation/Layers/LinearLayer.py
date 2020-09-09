import numpy as np
from typing import Dict

# import function to initialize weights and biases
from Second_Implementation.util.paramInitializer import initialize_parameters


class LinearLayer:
    """
        This Class implements a linear layer for the neural network
        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters
                      Options are: "plain" (small random gaussian numbers),
                                   "xavier" and "he"

        Methods:
            forward(A_prev)
            backward(upstream_grad)
            update_params(learning_rate)
    """

    # Class variables
    m: int  # number of examples in training data
    params: Dict  # stores weights and bias in a dictionary
    Z:  np.array  # Z output of linear layer

    A_prev: np.array  # activations from the previous layer

    dW: np.array  # derivative of Cost w.r.t W
    db: np.array  # derivative of Cost w.r.t b, sum across rows
    dA_prev: np.array  # derivative of Cost w.r.t A_prev

    def __init__(self, input_shape, n_out, ini_type="plain"):
        self.m = input_shape[1]

        # initialize weights and bias with type <ini_type>
        self.params = initialize_parameters(input_shape[0], n_out, ini_type)

        # create space for resultant Z output
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))

    def forward(self, A_prev):
        """
        Performs the forwards propagation using activations from previous layer
        Args:
            A_prev:  Activations/Input Data coming into the layer from previous
                     layer
        """

        self.A_prev = A_prev  # store the Activations/Training Data coming in

        # compute the linear function
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']

    def backward(self, upstream_grad):
        """
        Performs the back propagation using upstream gradients. Multiplies
        upstream gradient with local gradient to get the derivative of Cost
        Args:
            upstream_grad: gradient coming in from the upper layer
        """

        # Derivatives to update parameters:

        # derivative of Cost w.r.t W
        # dLinear/dW = self.A_prev.T
        self.dW = np.dot(upstream_grad, self.A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        # dLinear/dB = 1
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)

        # Derivative to keep backpropagading down

        # derivative of Cost w.r.t A_prev (if
        # dLinear/dA_prev = self.params['W'].T
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learning_rate=0.1):
        """
        Performs the gradient descent update
        Args:
            learning_rate: learning rate hyper-param for gradient descent
        """

        self.params['W'] -= learning_rate * self.dW  # update weights
        self.params['b'] -= learning_rate * self.db  # update bias(es)
