from builtins import range
import numpy as np



def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    layernorm = bn_param.get("layernorm", 0)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":

        #1: Calculate mean value
        mean = x.mean(axis = 0) # (D,)

        #2: Subtract mean vector of every training example
        diff_from_mean = x - mean # (N,D)

        #3: Calculate variance
        dev_from_mean_sq = diff_from_mean ** 2 # (N,D)
        var = 1./N * np.sum(dev_from_mean_sq, axis = 0) # (D,)

        #4: Calculate standard deviation
        stddev = np.sqrt(var + eps) # (D,)
        inverted_stddev = 1./stddev # (D,)

        #5: Apply normalization
        x_norm = diff_from_mean * inverted_stddev # also called z or x_hat (N,D)

        ##6: Apply Gamma and Beta
        scaled_x = gamma * x_norm # (N,D)
        out = scaled_x + beta # (N,D)

        #############IDK Zone##################
        # cache values for backward pass
        cache = {'mean': mean, 'stddev': stddev, 'var': var, 'gamma': gamma, 
                 'beta': beta, 'eps': eps, 'x_norm': x_norm, 'diff_from_mean': diff_from_mean,
                 'inverted_stddev': inverted_stddev, 'x': x}

        # since we transpose dout and make it (D,N) during backprop for layernorm
        cache['axis'] = 1 if layernorm else 0

        # also keep an exponentially decaying running weighted mean of the mean and 
        # variance of each feature, to normalize data at test-time
        if not layernorm:
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var
        #############IDK Zone##################


    elif mode == "test":
        # normalize the incoming data based on the standard deviation (sqrt(variance))
        z = (x - running_mean)/np.sqrt(running_var + eps)
        # scale and shift
        out = gamma * z + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):

    dx, dgamma, dbeta = None, None, None
    ###########################################################################

    # extract all relevant params
    beta, gamma, x_norm, var, eps, stddev, diff_from_mean, inverted_stddev, x, mean, axis = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['diff_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean'], cache['axis']

    # get the num of training examples and dimensionality of the input (num of features)
    N, D = dout.shape # can also use x.shape

    # (9)
    dbeta = np.sum(dout, axis=axis)
    dscaled_x = dout 

    # (8)
    dgamma = np.sum(x_norm * dscaled_x, axis=axis)
    dx_norm = gamma * dscaled_x

    # (7)
    dinverted_stddev = np.sum(diff_from_mean * dx_norm, axis=0)
    ddev_from_mean = inverted_stddev * dx_norm

    # (6)
    dstddev = -1/(stddev**2) * dinverted_stddev

    # (5)
    dvar = (0.5) * 1/np.sqrt(var + eps) * dstddev

    # (4)
    ddev_from_mean_sq = 1/N * np.ones((N,D)) * dvar # variance of mean is 1/N

    # (3)
    ddev_from_mean += 2 * diff_from_mean * ddev_from_mean_sq

    # (2)
    dx = 1 * ddev_from_mean
    dmean = -1 * np.sum(ddev_from_mean, axis=0)

    # (1)
    dx += 1./N * np.ones((N,D)) * dmean


    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    
    # extract all relevant params
    beta, gamma, x_norm, var, eps, stddev, diff_from_mean, inverted_stddev, mean, x, axis = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['diff_from_mean'], cache['inverted_stddev'], cache['mean'], \
    cache['x'], cache['axis']

    # get the num of training examples and dimensionality of the input (num of features)
    N = dout.shape[0] # can also use x.shape

    # (9)
    dbeta = np.sum(dout, axis=axis)
    dscaled_x = dout 

    # (8)
    dgamma = np.sum((x - mean) * (var + eps)**(-1. / 2.) * dout, axis=axis)

    dmean = 1/N * np.sum(dout, axis=0)
    dvar = 2/N * np.sum(diff_from_mean * dout, axis=0)
    dstddev = dvar/(2 * stddev)
    dx = gamma*((dout - dmean)*stddev - dstddev*(diff_from_mean))/stddev**2
    ###########################################################################

    return dx, dgamma, dbeta
