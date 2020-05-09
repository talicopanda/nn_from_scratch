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
