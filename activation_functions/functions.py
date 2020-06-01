import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)