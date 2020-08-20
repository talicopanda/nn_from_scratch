import numpy as np


class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # axis=None adds up the whole array
        # axis=0 adds up the columns
        # axis=1 adds up the rows
        # keepdims=True keeps the dimensions from the array

        ''' If we subtract the maximum value from a list of input values, the 
        inputs will always be non-positive and thus the exponential will be in 
        (-infty, 1], since e^-infty = 0 and e^0 = 1 '''

        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

        ''' Similarly:
        # values from output earlier when described what a neural network is
        layer_outputs = [4.8, 1.21, 2.385]
        
        # e - mathematical constant, we use E here to match a common coding
        # style where constants are uppercased
        E = 2.71828182846  # you can also use math.e
        
        # For each value in a vector, calculate the exponential value
        exp_values = []
        for output in layer_outputs:
            exp_values.append(E ** output)  # ** - power operator in Python
        print('exponentiated values:')
        print(exp_values)
        
        # Now normalize values
        norm_base = sum(exp_values)  # We sum all values
        norm_values = []
        for value in exp_values:
            norm_values.append(value / norm_base)'''


