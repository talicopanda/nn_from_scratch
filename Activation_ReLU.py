import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

        ''' Notice that it is equivalent to:
        output = []
        for i in inputs:
            if i > 0:
                output.append(i)
            else:
                output.append(0)
        '''

        # Remember input values for backprop
        self.inputs = inputs

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative
        self.dvalues[self.inputs <= 0] = 0
