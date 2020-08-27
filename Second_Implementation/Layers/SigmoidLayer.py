import numpy as np

class SigmoidLayer:
    """
    This Class implements a Sigmoid activation layers
    Args:
        shape: shape of input to the layer
    Methods:
        forward(Z)
        backward(upstream_grad)
    """

    A: np.array  # output of linear layer
    dZ: np.array  # derivative of Cost w.r.t the previous layer

    def __init__(self, shape):
        # create space for the resultant activations
        self.A = np.zeros(shape)

    def forward(self, Z):
        """
        Performs the forwards propagation step through the activation function
        Args:
            Z: input from previous (linear) layer
        """
        self.A = 1 / (1 + np.exp(-Z))  # compute sigmoid activations

    def backward(self, upstream_grad):
        """
        Performs the back propagation step through the activation function
        Local gradient: derivative of sigmoid = A*(1-A)
        Args:
            upstream_grad: gradient coming into this layer from the layer above
        """
        # Multiplies upstream gradient with local gradient to get the derivative
        # of Cost
        self.dZ = upstream_grad * self.A*(1-self.A)
