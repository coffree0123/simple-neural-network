import numpy as np

import activations
from layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.weights = None
        self.bias = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(self.inputs)

        return self.output

    def backward(self, next_delta):
        # Compute dE/X for a given next_delta=dE/Y.
        # There is no gradient in activation layer.
        delta = next_delta * self.activation_prime(self.inputs)
        return delta

    def update(self, lr=0.001):
        # Activation_layer do not need to update weight.
        return

    def clear_grad(self):
        self.weights = None
        self.bias = None


class ReLU(ActivationLayer):
    def __init__(self):
        activation = activations.relu
        activation_prime = activations.relu_prime
        super().__init__(activation, activation_prime)


class Tanh(ActivationLayer):
    def __init__(self):
        activation = activations.tanh
        activation_prime = activations.tanh_prime
        super().__init__(activation, activation_prime)
