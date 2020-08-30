import numpy as np

"""
This file implementations of :
    - Mean Squared Error Cost function
    - Binary Cross_entropy Cost function
        * compute_binary_cost(Y, P_hat) -> "unstable"
        * compute_stable_bce_cost(Y, Z) -> "stable" 
        * computer_keras_like_bce_cost(Y, P_hat, from_logits=False) -> stable
"""

def compute_mse_cost(Y, Y_hat):
    """
    Computes Mean Squared Error(mse) Cost and returns the Cost and
    its derivative.
    Squared Error Cost = (1/2m)*sum(Y - Y_hat)^.2
    Args:
        Y: labels of data
        Y_hat: Predictions(activations) from a last layer, the output layer
    Returns:
        cost: The Squared Error Cost result
        dY_hat: gradient of Cost w.r.t Y_hat
    """
    # number of examples in the batch
    m = Y.shape[1]

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    # remove extra dimensions to give just a scalar
    # (e.g. np.squeeze([[17]]) returns 17)
    cost = np.squeeze(cost)

    # derivative of the squared error cost function
    dY_hat = -1 / m * (Y - Y_hat)

    return cost, dY_hat


# ----------- Additional Cost Functions -----------


def compute_bce_cost(Y, P_hat):
    """
    Computes Binary Cross-Entropy(bce) Cost and returns the Cost and its
    derivative.
    BCE Cost = (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    Args:
        Y: labels of data
        P_hat: Estimated output probabilities from the last layer, the output layer
    Returns:
        cost: The Binary Cross-Entropy Cost result
        dP_hat: gradient of Cost w.r.t P_hat
    """
    # number of examples in the batch
    m = Y.shape[1]

    cost = (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    # remove extra dimensions
    cost = np.squeeze(cost)

    dP_hat = (1/m) * (-(Y/P_hat) + ((1-Y)/(1-P_hat)))

    return cost, dP_hat


def compute_stable_bce_cost(Y, Z):
    """
    Computes the "Stable" Binary Cross-Entropy(stable_bce) Cost and returns the \
    Cost and its derivative w.r.t Z_last(the last linear node)
    Stable BCE Cost = (1/m) * np.sum(max(Z,0) - ZY + log(1+exp(-|Z|)))
    Args:
        Y: labels of data
        Z: Values from the last linear node
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    m = Y.shape[1]

    cost = (1/m) * np.sum(np.maximum(Z, 0) - Z*Y + np.log(1+ np.exp(- np.abs(Z))))
    # from Z computes the Sigmoid so P_hat - Y, where P_hat = sigma(Z)
    dZ_last = (1/m) * ((1/(1+np.exp(- Z))) - Y)

    return cost, dZ_last

# TODO: Research more about this cost implementation
def compute_keras_like_bce_cost(Y, P_hat, from_logits=False):
    """
    Computes the Binary Cross-Entropy(stable_bce) Cost function the way Keras
    implements it. Accepting either probabilities(P_hat) from the sigmoid neuron
    or values direct from the linear node(Z)
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
