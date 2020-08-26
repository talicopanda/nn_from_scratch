import numpy as np

def initialize_parameters(n_in, n_out, ini_type='plain'):
    """
    Helper function to initialize some form of random weights and Zero biases
    Args:
        n_in: size of input layer
        n_out: size of output/number of neurons
        ini_type: set initialization type for weights
    Returns:
        params: a dictionary containing W and b
    """

    # initialize empty dictionary of neural net parameters W and b
    params = dict()
    # the <params> dictionary contains 2 numpy arrays found by the keywords
    # "W" and "b" representing the weights and biases respectively

    # Full explanation of the different weight initialization techniques:
    # https://cs231n.github.io/neural-networks-2/#init

    if ini_type == 'plain':
        # set weights 'W' to small random gaussian
        params['W'] = np.random.randn(n_out, n_in) * 0.01
    elif ini_type == 'xavier':
        # same as above but normalized
        # set variance of W to 1/n
        params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))
    elif ini_type == 'he':
        # Good initialization when ReLU used in hidden layers
        # As exaplined in this paper: https://arxiv.org/abs/1502.01852
        params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)  # set variance of W to 2/n

    # set bias 'b' to zeros
    params['b'] = np.zeros((n_out, 1))

    return params
