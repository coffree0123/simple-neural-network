import numpy as np
from abc import ABC, abstractclassmethod

class Layer(ABC):
    # Define abstract layer for all kind of layers.
    @abstractclassmethod
    def __init__(self):
        pass

    def forward(self):
        # Computes the ouput Y for an input X.
        pass

    def backward(self):
        # Computes dE/dX, dE/dW, dE/dB for a given next_delta=dE/dY.
        pass

    def update(self):
        # Update weights.
        pass

    def clear_grad(self):
        # Clear gradient in a layer.
        pass
