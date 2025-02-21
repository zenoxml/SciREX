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
Example Script: model_cnn.py

A TensorFlow implementation of a Convolutional Neural Network (CNN) that provides:
- ModelType enum defining the CNN variant (SIMPLE_CNN)
- BaseModel class that handles 2D/3D input shape conversion
- SimpleCNN implementation with two conv-pool blocks (1->4->8 channels)
- Factory function (get_model) for model instantiation


Attributes:
    ModelType (Enum): Available model architectures
    
Authors:
 - Nithyashree R (nithyashreer@iisc.ac.in).
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


class BaseModel:
    """
    Base class for all model architectures.

    This class serves as an abstract base class that defines the interface
    for all model implementations. It automatically handles channel dimension
    for 2D input shapes.

    Attributes:
        input_shape (Tuple[int, ...]): Shape of the input tensor with channels
        num_classes (int): Number of output classes
    """
    def __init__(self, input_shape: Union[Tuple[int, int], Tuple[int, int, int]], 
                num_classes: int):
        """
        Initialize the base model with automatic channel handling.

        :param input_shape: Shape of the input tensor (height, width) or (height, width, channels)
        :type input_shape: Union[Tuple[int, int], Tuple[int, int, int]]
        :param num_classes: Number of output classes
        :type num_classes: int
        """
        if len(input_shape) == 2:
            self.input_shape = (*input_shape, 1)  # Add channel dimension
        else:
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
    CNN implementation for image classification.

    This class implements a convolutional neural network with two
    convolutional blocks followed by a dense classification layer.

    Architecture details:
        Block 1:
            - Conv2D: 1->4 channels, 4x4 kernel
            - MaxPool2D: 2x2 pooling
            - ReLU activation
        Block 2:
            - Conv2D: 4->8 channels, 4x4 kernel
            - MaxPool2D: 2x2 pooling
            - ReLU activation
        Classification:
            - Flatten
            - Dense layer to num_classes
            - LogSoftmax activation

    Attributes:
        input_shape (Tuple[int, int, int]): Shape of the input tensor with channels
        num_classes (int): Number of output classes
    """
    def build(self) -> tf.keras.Model:
        """
        Build and compile the CNN model.

        Creates a sequential model with two convolutional blocks followed by
        a dense classification layer. Each conv block includes convolution,
        max pooling, and ReLU activation.

        :return: Compiled CNN model
        :rtype: tf.keras.Model
        """
        model = keras.Sequential([
            # First conv block: 1->4 channels
            keras.layers.Conv2D(4, kernel_size=4, input_shape=self.input_shape),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Activation("relu"),
            # Second conv block: 4->8 channels
            keras.layers.Conv2D(8, kernel_size=4),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Activation("relu"),
            # Flatten and dense layer
            keras.layers.Flatten(),
            keras.layers.Dense(self.num_classes),
            keras.layers.Activation("log_softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model


def get_model(model_type: ModelType, input_shape: Union[Tuple[int, int], Tuple[int, int, int]], 
             num_classes: int) -> tf.keras.Model:
    """
    Factory function to get the specified model.

    This function creates and returns a compiled model of the specified type.
    It uses a factory pattern to map model types to their implementations.

    :param model_type: Type of model to create
    :type model_type: ModelType
    :param input_shape: Shape of the input tensor (height, width) or (height, width, channels)
    :type input_shape: Union[Tuple[int, int], Tuple[int, int, int]]
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