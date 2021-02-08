import numpy as np

from fc_layer import *
from activation_layer import *


class Module:
    def __init__(self):
        self.layers = []
        # Here I only implement MSE.
        self.loss = lambda y_true, y_pred: np.mean(
            np.power(y_true - y_pred, 2))
        self.loss_prime = lambda y_true, y_pred: -2 * (y_true - y_pred)

    def __call__(self, inputs):
        # This is used to implement y = Network(x)
        return self.forward(inputs)

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, batch_size=32, iterations=100, lr=0.001, eval_steps=100):
        # Use SGD to update network parameters.
        self.lr = lr
        num_batch = len(x) // batch_size

        for epoch in range(iterations):
            cur_loss = 0
            for batch in range(num_batch):
                # Here implement backpropagation algorithm.
                inputs = x[batch_size * (batch): batch_size * (batch+1), :]
                y_true = y[batch_size * (batch): batch_size * (batch+1), :]
                y_pred = self.forward(inputs)
                self.backward(y_true, y_pred)
                cur_loss += self.loss(y_true, y_pred)

            cur_loss = cur_loss / num_batch

            if (epoch + 1) % eval_steps == 0:
                print("epoch:{}/{}, loss:{}".format(epoch+1, iterations, cur_loss))

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, y_true, y_pred):
        # Calculate delta and backward the delta.
        next_delta = self.loss_prime(y_true, y_pred)
        for layer in reversed(self.layers):
            next_delta = layer.backward(next_delta)
            # Update weights using SGD.
            layer.update(self.lr)

    def save_model(self, pathname="model.npy"):
        # Here I just save the whole model.
        np.save(pathname, self.layers)

    def load_model(self, pathname="model.npy"):
        # If you load a previous model, it will replace the original structure of your network.
        self.layers = np.load(pathname, allow_pickle=True)

    def clear_grad(self):
        for layer in self.layers:
            layer.clear_grad()


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)
