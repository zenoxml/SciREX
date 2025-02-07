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
# http://www.apache.org/licenses/LICENSE-2.0
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
Example Script: models.py

This module provides base classes and implementations for different neural network
architectures. It includes a factory pattern for model creation and supports
multiple model architectures through a common interface.

The module includes:
- Base model interface
- Simple CNN implementation
- Factory method for model instantiation

Attributes:
    ModelType (Enum): Available model architectures

Authors:
 - Nithyashree R (nithyashreer@iisc.ac.in)
"""


from enum import Enum
import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras
from typing import Tuple, Union


class ModelType(Enum):
    """
    Enumeration of supported model architectures.

    Attributes:
        SIMPLE_CNN (str): Simple convolutional neural network architecture
    """
    SIMPLE_CNN = "simple_cnn"
    # Add more model types here as needed


class BaseModel:
    """
    Base class for all model architectures.

    This class serves as an abstract base class that defines the interface
    for all model implementations. Each model type should inherit from this
    class and implement the build method.

    Attributes:
        input_shape (Tuple[int, ...]): Shape of the input tensor
        num_classes (int): Number of output classes
    """
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int):
        """
        Initialize the base model.

        :param input_shape: Shape of the input tensor
        :type input_shape: Tuple[int, ...]
        :param num_classes: Number of output classes
        :type num_classes: int
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self) -> tf.keras.Model:
        """
        Build and return the model.

        This method should be implemented by all subclasses to define
        their specific model architecture.

        :return: Compiled Keras model
        :rtype: tf.keras.Model
        :raises NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement build()")


class SimpleCNN(BaseModel):
    """
    Simple CNN model architecture implementation.

    This class implements a basic convolutional neural network architecture
    suitable for image classification tasks. The architecture includes:
    - Input reshaping to handle image data
    - Convolutional layer with ReLU activation
    - Max pooling layer
    - Dense output layer

    Attributes:
        input_shape (Tuple[int, ...]): Shape of the input tensor
        num_classes (int): Number of output classes
    """
    def build(self) -> tf.keras.Model:
        """
        Build and compile the Simple CNN model.

        Creates a sequential model with:
        - Input layer matching the specified input shape
        - Reshape layer to prepare for convolution
        - Conv2D layer with 12 filters and ReLU activation
        - MaxPooling2D layer
        - Flatten layer to prepare for dense output
        - Dense output layer

        :return: Compiled CNN model
        :rtype: tf.keras.Model
        """
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=self.input_shape),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(self.num_classes),
        ])

        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model


def get_model(model_type: ModelType, input_shape: Tuple[int, ...], 
             num_classes: int) -> tf.keras.Model:
    """
    Factory function to get the specified model.

    This function creates and returns a compiled model of the specified type.
    It uses a factory pattern to map model types to their implementations.

    :param model_type: Type of model to create
    :type model_type: ModelType
    :param input_shape: Shape of the input tensor
    :type input_shape: Tuple[int, ...]
    :param num_classes: Number of output classes
    :type num_classes: int
    :return: Compiled model of the specified type
    :rtype: tf.keras.Model
    :raises ValueError: If the specified model type is not supported
    """
    model_map = {
        ModelType.SIMPLE_CNN: SimpleCNN,
    }
    
    model_class = model_map.get(model_type)
    if model_class is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model_class(input_shape, num_classes).build()