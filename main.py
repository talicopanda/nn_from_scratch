import numpy as np

from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
from Layer_Dense import Layer_Dense
from Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy

np.random.seed(0)

def create_data(points, classes):
    """ This function allows us to create a dataset with as many classes as we
    want. The function has parameters to choose the number of classes and the
    number of points/observations per class in the resulting non-linear dataset
    """
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


if __name__ == '__main__':
    # Create dataset
    X, y = create_data(100, 3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs

    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()

    # Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs

    # Create Softmax activation (to be used with Dense layer):
    activation2 = Activation_Softmax()

    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()

    # Make a forward pass of our training data thru this layer
    dense1.forward(X)

    # Make a forward pass thru activation function - we take output of previous layer here
    activation1.forward(dense1.output)

    # Make a forward pass thru second Dense layer - it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Make a forward pass thru activation function - we take output of previous layer here
    activation2.forward(dense2.output)

    # Let's see output of few first samples:
    print(activation2.output[:5])

    # Calculate loss from output of activation2 (softmax activation)
    loss = loss_function.forward(activation2.output, y)

    # Let's print loss value
    print('loss:', loss)

    # Calculate accuracy from output of activation2 and targets
    predictions = np.argmax(activation2.output, axis=1)  # find the index of
    # the "predicted" value (i.e. max activation)
    accuracy = np.mean(predictions == y)  # averages the number of correct predictions

    # Print accuracy
    print('acc:', accuracy)


