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

    params = dict()  # initialize empty dictionary of neural net parameters W and b

    if ini_type == 'plain':
        params['W'] = np.random.randn(n_out, n_in) *0.01  # set weights 'W' to small random gaussian
    elif ini_type == 'xavier':
        params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
    elif ini_type == 'he':
        # Good when ReLU used in hidden layers
        # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # Kaiming He et al. (https://arxiv.org/abs/1502.01852)
        # http: // cs231n.github.io / neural - networks - 2 /  # init
        params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)  # set variance of W to 2/n

    params['b'] = np.zeros((n_out, 1))    # set bias 'b' to zeros

    return params

def compute_cost(Y, Y_hat):
    """
    This function computes and returns the Cost and its derivative.
    The is function uses the Squared Error Cost function -> (1/2m)*sum(Y - Y_hat)^.2
    Args:
        Y: labels of data
        Y_hat: Predictions(activations) from a last layer, the output layer
    Returns:
        cost: The Squared Error Cost result
        dY_hat: gradient of Cost w.r.t the Y_hat
    """
    m = Y.shape[1]

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar

    dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

    return cost, dY_hat

# ------------- Different Cost Functions -----------------

def compute_bce_cost(Y, P_hat):
    """
    This function computes Binary Cross-Entropy(bce) Cost and returns the Cost and its
    derivative.
    This function uses the following Binary Cross-Entropy Cost defined as:
    => (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    Args:
        Y: labels of data
        P_hat: Estimated output probabilities from the last layer, the output layer
    Returns:
        cost: The Binary Cross-Entropy Cost result
        dP_hat: gradient of Cost w.r.t P_hat
    """
    m = Y.shape[1]  # m -> number of examples in the batch

    cost = (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar (e.g. this turns [[17]] into 17)

    dP_hat = (1/m) * (-(Y/P_hat) + ((1-Y)/(1-P_hat)))

    return cost, dP_hat

def compute_stable_bce_cost(Y, Z):
    """
    This function computes the "Stable" Binary Cross-Entropy(stable_bce) Cost and returns the Cost and its
    derivative w.r.t Z_last(the last linear node) .
    The Stable Binary Cross-Entropy Cost is defined as:
    => (1/m) * np.sum(max(Z,0) - ZY + log(1+exp(-|Z|)))
    Args:
        Y: labels of data
        Z: Values from the last linear node
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    m = Y.shape[1]

    cost = (1/m) * np.sum(np.maximum(Z, 0) - Z*Y + np.log(1+ np.exp(- np.abs(Z))))
    dZ_last = (1/m) * ((1/(1+np.exp(- Z))) - Y)  # from Z computes the Sigmoid so P_hat - Y, where P_hat = sigma(Z)

    return cost, dZ_last

def compute_keras_like_bce_cost(Y, P_hat, from_logits=False):
    """
    This function computes the Binary Cross-Entropy(stable_bce) Cost function the way Keras
    implements it. Accepting either probabilities(P_hat) from the sigmoid neuron or values direct
    from the linear node(Z)
    Args:
        Y: labels of data
        P_hat: Probabilities from sigmoid function
        from_logits: flag to check if logits are being provided or not(Default: False)
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    if from_logits:
        # assume that P_hat contains logits and not probabilities
        return compute_stable_bce_cost(Y, Z=P_hat)

    else:
        # Assume P_hat contains probabilities. So make logits out of them

        # First clip probabilities to stable range
        EPSILON = 1e-07
        P_MAX = 1- EPSILON  # 0.9999999

        P_hat = np.clip(P_hat, a_min=EPSILON, a_max=P_MAX)

        # Second, Convert stable probabilities to logits(Z)
        Z = np.log(P_hat/(1-P_hat))

        # now call compute_stable_bce_cost
        return compute_stable_bce_cost(Y, Z)
