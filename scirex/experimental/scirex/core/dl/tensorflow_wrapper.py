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

"""TensorFlow layer factory implementations.

This module provides factory methods for creating TensorFlow/Keras layers with
consistent configuration options. Currently supports Dense and Conv2D layers.

Key classes:
    - TensorflowDense:  for Dense layers
    - TensorflowConv2D:  for Conv2D layers

Example:
    >>> dense_layer = TensorflowDense.create_layer(units=64, activation='relu')
    >>> conv_layer = TensorflowConv2D.create_layer(filters=32, kernel_size=3)
"""

from typing import Optional, Union, Tuple
import tensorflow as tf
from tensorflow.keras import layers


class TensorflowDense:
    """Factory class for creating Keras Dense layers.

    Provides a static method for creating and configuring Keras Dense layers
    with consistent parameters.

    Example:
        >>> # Create a dense layer with 32 units and ReLU activation
        >>> dense1 = TensorflowDense.create_layer(
        ...     units=32,
        ...     activation='relu'
        ... )
        >>>
        >>> # Create an output layer with 1 unit and no activation
        >>> dense2 = TensorflowDense.create_layer(
        ...     units=1,
        ...     activation=None
        ... )
    """

    @staticmethod
    def create_layer(
        units: int,
        activation: Optional[Union[str, callable]] = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
    ) -> tf.keras.layers.Dense:
        """Create and return a Keras Dense layer.

        Args:
            units: Number of output units
            activation: Activation function to use
            kernel_initializer: Initializer for kernel weights
            bias_initializer: Initializer for bias vector
            dtype: Data type for layer computations

        Returns:
            tf.keras.layers.Dense: The configured Dense layer
        """
        return layers.Dense(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            dtype=dtype,
        )


class TensorflowConv2D:
    """Factory class for creating Keras Conv2D layers.

    Provides a static method for creating and configuring Keras Conv2D layers
    with consistent parameters.

    Example:
        >>> # Create a conv layer with 64 filters and 3x3 kernel
        >>> conv1 = TensorflowConv2D.create_layer(
        ...     filters=64,
        ...     kernel_size=3,
        ...     activation='relu'
        ... )
        >>>
        >>> # Create a conv layer with 32 filters and 5x5 kernel
        >>> conv2 = TensorflowConv2D.create_layer(
        ...     filters=32,
        ...     kernel_size=(5, 5),
        ...     activation='relu'
        ... )
    """

    @staticmethod
    def create_layer(
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        activation: Optional[Union[str, callable]] = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
    ) -> tf.keras.layers.Conv2D:
        """Create and return a Keras Conv2D layer.

        Args:
            filters: Number of output filters
            kernel_size: Size of convolution kernel
            activation: Activation function to use
            kernel_initializer: Initializer for kernel weights
            bias_initializer: Initializer for bias vector
            dtype: Data type for layer computations

        Returns:
            tf.keras.layers.Conv2D: The configured Conv2D layer
        """
        return layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            dtype=dtype,
        )


# Example usage
if __name__ == "__main__":
    # Create dense layers
    hidden_layer = TensorflowDense.create_layer(
        units=64, activation="relu", dtype=tf.float32
    )

    output_layer = TensorflowDense.create_layer(
        units=10, activation="softmax", dtype=tf.float32
    )

    # Create conv layers
    conv_layer1 = TensorflowConv2D.create_layer(
        filters=32, kernel_size=3, activation="relu"
    )

    conv_layer2 = TensorflowConv2D.create_layer(
        filters=64, kernel_size=(5, 5), activation="relu"
    )

    # Example model construction
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = conv_layer1(inputs)
    x = conv_layer2(x)
    x = tf.keras.layers.Flatten()(x)
    x = hidden_layer(x)
    outputs = output_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
