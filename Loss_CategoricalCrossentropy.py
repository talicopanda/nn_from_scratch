import numpy as np


class Loss_CategoricalCrossentropy:

    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Probabilities for target values
        y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Overall loss
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss

''' Derivation
import math

softmax_output = [0.7, 0.1, 0.2]  # example output from the output layer of the neural network.
target_output = [1, 0, 0]  # ground truth

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +  # zero
         math.log(softmax_output[2]) * target_output[2])  # zero
# We don't actually need to calculate cases of index 1 and 2 (when one-hot target = 0)
# also target_output[0] = 1 so the loss can just be:

loss = -(math.log(softmax_output[0]))
'''

'''
When doing a batch of inputs we can do:
softmax_outputs = np.array([[0.7, 0.1, 0.2],  
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1] # indexes of the right predictions

for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])
    
or similarly:

print(softmax_outputs[[0,1,2], class_targets])

Then, applying the loss function:

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log)
'''
