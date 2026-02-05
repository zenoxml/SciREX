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
    File: dense.py
    Description: This module creates dense layers in the TensorFlow backend.

    Authors:
        - Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 03/02/2025: Initial version

"""

from typing import Optional, Union, Callable
import tensorflow as tf
from tensorflow.keras import layers
from scirex.core.dl.tf_backend.layers.base import TensorflowLayer
from scirex.core.dl.tf_backend.activations import *
from scirex.core.dl.tf_backend.datautils import *
from scirex.core.dl.tf_backend.mathutils import *


class DenseLayer(TensorflowLayer):
    """Factory class for creating native TensorFlow dense layers.

    Implements dense (fully connected) layers using native TensorFlow operations.

    Example:
        >>> # Create a dense layer with 32 units and ReLU activation
        >>> dense1 = DenseLayer.create_layer(
        ...     input_dim=64,
        ...     units=32,
        ...     activation='relu'
        ... )
        >>>
        >>> # Use the layer in forward pass
        >>> output = dense1(input_tensor)
    """

    def __init__(
        self,
        input_dim: int,
        units: int,
        activation: Optional[Union[str, Callable]] = None,
        dtype: tf.DType = tf.float32,
    ):
        """Initialize the dense layer.

        Args:
            input_dim: Dimension of the input features
            units: Number of output units
            activation: Activation function or name of activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.dtype = dtype

        self.activation = self._get_activation_fn(activation)

        # Initialize weights and biases
        self.weights = None
        self.bias = None
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier/Glorot initialization."""
        initializer = tf.initializers.GlorotUniform()
        self.weights = tf.Variable(
            initializer(shape=[self.input_dim, self.units]),
            trainable=True,
            dtype=self.dtype,
            name="dense_weights",
        )
        self.bias = tf.Variable(
            tf.zeros([self.units]), trainable=True, dtype=self.dtype, name="dense_bias"
        )

    def _get_activation_fn(
        self, activation: Optional[Union[str, Callable]]
    ) -> Optional[Callable]:
        """Convert activation function name to callable if necessary."""
        if activation is None:
            return None
        if callable(activation):
            return activation

        activation_map = {
            "relu": relu,
            "sigmoid": sigmoid,
            "tanh": tanh,
        }

        if activation.lower() not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation}")

        return activation_map[activation.lower()]

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, units]
        """
        # Check input shape
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {inputs.shape[-1]}"
            )

        # Linear transformation
        outputs = tf.matmul(inputs, self.weights) + self.bias

        # Apply activation if specified
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    @property
    def parameter_count(self) -> int:
        """Calculate total number of trainable parameters in the layer."""
        return self.weights.shape[0] * self.weights.shape[1] + self.units

    def get_config(self) -> dict:
        """Get layer configuration."""
        return {
            "input_dim": self.input_dim,
            "units": self.units,
            "activation": self.activation.__name__ if self.activation else None,
        }

    @classmethod
    def from_config(cls, config: dict) -> "DenseLayer":
        """Create layer from configuration dictionary."""
        return cls(**config)

    def reset_parameters(self):
        """Reset layer parameters to their initial values."""
        self._initialize_parameters()

    def create_layer(self, **kwargs):
        return DenseLayer(**kwargs)

    def configure_layer(self, **kwargs):
        return DenseLayer(**kwargs)
