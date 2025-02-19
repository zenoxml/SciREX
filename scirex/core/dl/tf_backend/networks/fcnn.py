# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org
"""
    File: fcnn.py
    Description: Implements a fully connected neural network using native TensorFlow operations.

    Authors: Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 03/02/2025: Initial version
"""
from typing import List, Optional
import tensorflow as tf
from scirex.core.dl.tf_backend.layers.dense import DenseLayer


class FullyConnectedNetwork:
    """
    A fully connected neural network implementation using native TensorFlow operations.

    Attributes:
        architecture (List[int]): List defining the number of neurons in each layer
        layers (List[DenseLayer]): List of dense layers in the network
        activation (str): Activation function for hidden layers
        output_activation (str): Activation function for output layer
    """

    def __init__(
        self,
        architecture: List[int],
        hidden_activation: str = "relu",
        output_activation: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """
        Initialize the neural network.

        Args:
            architecture: List of integers defining the number of neurons in each layer
                        (including input and output layers)
            hidden_activation: Activation function for hidden layers
            output_activation: Activation function for output layer

        Example:
            >>> # Create a network with architecture [2, 30, 30, 1]
            >>> net = FullyConnectedNetwork([2, 30, 30, 1])
        """
        if len(architecture) < 2:
            raise ValueError("Architecture must have at least input and output layers")

        self.architecture = architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dtype = dtype
        self.layers = []

        self._build_network()

    def _build_network(self):
        """Construct the network by creating and connecting dense layers."""
        # Create hidden layers
        for i in range(len(self.architecture) - 2):
            layer = DenseLayer(
                input_dim=self.architecture[i],
                units=self.architecture[i + 1],
                activation=self.hidden_activation,
                dtype=self.dtype,
            )
            self.layers.append(layer)

        # Create output layer
        output_layer = DenseLayer(
            input_dim=self.architecture[-2],
            units=self.architecture[-1],
            activation=self.output_activation,
            dtype=self.dtype,
        )
        self.layers.append(output_layer)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the network.

        Args:
            inputs: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Check input dimensions
        if inputs.shape[-1] != self.architecture[0]:
            raise ValueError(
                f"Expected input dimension {self.architecture[0]}, "
                f"got {inputs.shape[-1]}"
            )

        # Forward pass through each layer
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_weights(self) -> List[tf.Tensor]:
        """Get all trainable weights in the network."""
        weights = []
        for layer in self.layers:
            weights.extend([layer.weights, layer.bias])
        return weights

    @property
    def parameter_count(self) -> int:
        """Calculate total number of trainable parameters in the network."""
        return sum(layer.parameter_count for layer in self.layers)

    def reset_parameters(self):
        """Reset all layer parameters to their initial values."""
        for layer in self.layers:
            layer.reset_parameters()

    def get_config(self) -> dict:
        """Get network configuration."""
        return {
            "architecture": self.architecture,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
        }

    @classmethod
    def from_config(cls, config: dict) -> "FullyConnectedNetwork":
        """Create network from configuration dictionary."""
        return cls(**config)

    def summary(self) -> str:
        """Generate a string summary of the network architecture."""
        summary = ["\nNetwork Architecture Using Tensorflow v2 backend:"]
        summary.append("-" * 50)
        summary.append(f"Input dimension: {self.architecture[0]}")

        for i, (in_dim, out_dim) in enumerate(
            zip(self.architecture[:-1], self.architecture[1:])
        ):
            layer_type = "Output" if i == len(self.architecture) - 2 else "Hidden"
            activation = (
                self.output_activation
                if i == len(self.architecture) - 2
                else self.hidden_activation
            )
            params = in_dim * out_dim + out_dim  # weights + biases
            summary.append(
                f"{layer_type} Layer {i + 1}: {in_dim} → {out_dim} "
                f"(Activation: {activation}, Parameters: {params})"
            )

        summary.append("-" * 50)
        summary.append(f"Total parameters: {self.parameter_count}")
        return "\n".join(summary)


if __name__ == "__main__":
    nn_architecture = [2, 30, 30, 1]
    net = FullyConnectedNetwork(nn_architecture)
    print(net.summary())
