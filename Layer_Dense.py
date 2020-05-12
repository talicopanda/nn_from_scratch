import numpy as np
np.random.seed(0)


class Layer_Dense:
    def __init__(self, inputs, neurons):
        # randomly set the weights in an array of shape (<inputs>, <neurons>)
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        # np.random.randn() produces a Gaussian distribution with a mean of 0
        # and a variance of 1

        # set biases to 0
        self.biases = np.zeros(shape=(1, neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

        # Remember input values for backprop
        self.inputs = inputs

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        # inputs.T = partial derivatives of weights
        # self.dweights = dvalue*dweights (chain rule)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # dbiases = dvalue*dbias (chain rule) but dbias = 1
        # so dbiases = dvalue and we are summing up all samples

        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


''' "Derivation" of the above

inputs = [[1.0, 2.0, 3.0, 2.5], # first neuron
          [2.0, 5.0, -1.0, 2.0], # second neuron
          [-1.5, 2.7, 3.3, -0.8]] # third neuron

# layer 1
# their respective connection's weights
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# layer 2
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

# numpy automatically converts <inputs> and <biases> into arrays internally

Note: <inputs> are first on the dot product so the resulting output matrix 
keeps each row for the neurons and each column for the weights 
(matching with the inputs matrix)

layer1_outputs = np.dot(inputs, np.array(weights).T) + np.array(biases)
# inputs to layer 2 are the outputs from layer 1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + np.array(biases2)

Note: Notice that the dot product could be easily replaced with the following:
(they accomplish the same, but notice the difference in the number of lines)

# layer_outputs = []  # Output of current layer
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0  # Output of given neuron
#     for input, weight in zip(inputs, neuron_weights):
#         neuron_output += input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

'''
