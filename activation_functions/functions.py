import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0