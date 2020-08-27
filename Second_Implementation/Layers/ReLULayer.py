import numpy as np

class ReLULayer:
    """
    This Class implements a Rectified Linear Unit (ReLU) activation layer
    Args:
        shape: shape of input to the layer
    Methods:
        forward(Z)
        backward(upstream_grad)
    """

    A: np.array  # output of ReLU layer
    dZ: np.array  # derivative of Cost w.r.t the previous layer (upper stream)
    dRelu: np.array  # derivative of Relu w.r.t previous layer (local stream)

    def __init__(self, shape):
        # create space for the resultant activations
        self.A = np.zeros(shape)

    def forward(self, Z):
        """
        Performs the forwards propagation step through the activation function
        Args:
            Z: input from previous (linear) layer
        """
        self.A = np.max(0, Z)  # compute ReLU activations

    def backward(self, upstream_grad):
        """
        Performs the back propagation step through the activation function
        Multiplies upstream gradient with local gradient to get the derivative
        of Cost
        Args:
            upstream_grad: gradient coming into this layer from the layer above
        """
        # Local Gradient:
        # Relu Gradient = 0 where input values were <= 0 and 1 otherwise
        self.dRelu = self.A.copy()
        self.dRelu[self.A > 0] = 1
        self.dZ = upstream_grad * self.dRelu
