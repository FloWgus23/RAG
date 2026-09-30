from smolml.core.ml_array import zeros
import smolml.utils.activation as activation
import smolml.utils.initializers as initializers

"""
//////////////
/// LAYERS ///
//////////////
"""

class DenseLayer:
    """
    Creates a standard Dense (linear) layer for a Neural Network.
    This layer performs the operation: output = activation(input @ weights + biases)
    """
    def __init__(self, input_size: int, output_size: int, activation_function: callable = activation.linear, 
                 weight_initializer: initializers.WeightInitializer = initializers.XavierUniform) -> None:
        """
        Initializes layer parameters: weights using the specified initializer and biases with zeros.
        Default activation is linear and default weight initialization is Xavier Uniform.
        """
        self.weights = weight_initializer.initialize(input_size, output_size)
        self.biases = zeros(1, output_size)  # Initialize biases with zeros
        self.activation_function = activation_function

    def forward(self, input_data):
        """
        Performs the forward pass: applies linear transformation followed by activation function.
        """
        z = input_data @ self.weights + self.biases  # Linear transformation
        return self.activation_function(z)  # Apply activation function

    def update(self, optimizer, layer_idx):
        """Update parameters using the provided optimizer"""
        self.weights, self.biases = optimizer.update(self, layer_idx, param_names=("weights", "biases"))