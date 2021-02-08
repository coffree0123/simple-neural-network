import numpy as np

from layer import Layer


class Linear(Layer):
    def __init__(self, input_size, output_size, weights=None, bias=None):
        self.weights = weights if weights is not None \
            else np.random.rand(input_size, output_size)
        self.bias = bias if bias is not None else np.random.rand(
            1, output_size)
        self.weights_gradient = np.zeros((input_size, output_size))
        self.bias_gradient = np.zeros((1, output_size))
        delta = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        output = np.dot(self.inputs, self.weights) + self.bias

        return output

    def backward(self, next_delta):
        # Computes dE/dX, dE/dW, dE/dB for a given next_delta=dE/dY.
        batch_size = len(self.inputs)
        delta = np.dot(next_delta, self.weights.T)
        self.weights_gradient += np.dot(self.inputs.T, next_delta) / batch_size
        self.bias_gradient += next_delta.sum(axis=0) / batch_size

        return delta

    def update(self, lr=0.001):
        # Use SGD to update weights.
        self.weights -= lr * self.weights_gradient
        self.bias -= lr * self.bias_gradient
        # Clear gradient after updating weights.
        self.clear_grad()
        return

    def clear_grad(self):
        self.weights_gradient = np.zeros_like(self.weights_gradient)
        self.bias_gradient = np.zeros_like(self.bias_gradient)
