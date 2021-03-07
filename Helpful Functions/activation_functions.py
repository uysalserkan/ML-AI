"""
  Serkan UYSAL
  uysalserkan08@gmail.com
  -- 2021
"""

def sigmoid(x):
    """
    Sigmoid is equivalent to a 2-element Softmax, where the second element is assumed to be zero. 
    The sigmoid function always returns a value between 0 and 1.
    """
    return 1 / (1 + exp(-x))

def relu(x, miv_val = 0):
    """
    Applies the rectified linear unit activation function.

    Modifying default parameters allows you to use non-zero thresholds, change the max value of the activation, 
    and to use a non-zero multiple of the input for values below the threshold.
    """
    return max(min_val, x)

def softmax(x):
    """
    Requirements: `import tensorflow as tf`
    Softmax converts a real vector to a vector of categorical probabilities.

    The elements of the output vector are in range (0, 1) and sum to 1.
    """
    return exp(x) / tf.reduce_sum(exp(x))

def tanh(x):
    """
    Hyperbolic tangent activation function.

    tanh(x) = sinh(x)/cosh(x)
    """
    return ((exp(x) - exp(-x)) / (exp(x) + exp(-x))) # also we can return numpy.tanh(x)

def selu(x, alpha=1.67326324, scale=1.05070098):
    """
    Basically, the SELU activation function multiplies scale (> 1) with the output of the tf.keras.activations.elu function 
    to ensure a slope larger than one for positive inputs.
    """
    if x > 0:
        return x * scale
    return x * alpha * scale

def elu(x, alpha=1.0):
    """
    ELUs have negative values which pushes the mean of the activations closer to zero. 
    Mean activations that are closer to zero enable faster learning as they bring the gradient closer to the natural gradient. 
    ELUs saturate to a negative value when the argument gets smaller. Saturation means a small derivative which decreases 
    the variation and the information that is propagated to the next layer.
    """
    if x > 0:
        return x
    return (exp(x) - 1) * alpha

def gelu(x, approximate=False):
    """
    Requirement: import numpy as np
    Gaussian error linear unit (GELU) computes `x * P(X <= x`), where `P(X) ~ N(0, 1)`. 
    The (GELU) nonlinearity weights inputs by their value, rather than gates inputs by their sign as in ReLU.
    """
    if approximate:
        return x * (1 + np.tanh(sqrt(2 / np.pi) * (x + 0.044715 * np.pow(x, 3)))) * 0.5
    return x * (1 + np.erf(x / sqrt(2))) * 0.5

def softsign(x):
    return x / (abs(x) + 1)

def softplus(x):
    return log(exp(x) + 1)

