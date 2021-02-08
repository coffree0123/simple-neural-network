import numpy as np

# Activation function and its derivative
def relu(x):
    return np.maximum(0, x)
 
def relu_prime(x):
    s = x
    s[x <= 0] = 0
    s[x > 0] = 1
    return s

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))