# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab,
# Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# SciREX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with SciREX. If not, see <https://www.gnu.org/licenses/>.
#
# For any clarifications or special considerations,
# please contact <scirex@zenteiq.ai>

# Author: Nithyashree R

# This code defines a Python class QuantizationAwareTraining to enable quantization-aware training (QAT) and post-training quantization for TensorFlow models. 
# It includes methods for model evaluation, TensorFlow Lite conversion, and size measurement of both quantized and post-quantized models.

import tempfile
import os
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

class QuantizationAwareTraining:
    """
    A reusable class for performing quantization-aware training on any dataset.

    Attributes:
        model (tf.keras.Model): The base model architecture.
        q_aware_model (tf.keras.Model): The quantization-aware trained model.
    """

    def __init__(self, input_shape, num_classes):
        """
        Initializes the model architecture.

        :param input_shape: Shape of input data
        :type input_shape: tuple
        :param num_classes: Number of output classes.
        :type num_classes: int
        """
        self.model = self._build_model(input_shape, num_classes)
        self.q_aware_model = None

    @staticmethod
    def _build_model(input_shape, num_classes):
        """
        Builds the base model architecture.

        :param input_shape: Shape of input data.
        :type input_shape: tuple
        :param num_classes: Number of output classes.
        :type num_classes: int
        :return: A Keras model.
        :rtype: tf.keras.Model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Reshape(target_shape=input_shape + (1,)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)
        ])
        return model

    def train(self, train_data, train_labels, epochs=1, validation_split=0.1):
        """
        Trains the base model.

        :param train_data: Training dataset.
        :type train_data: np.ndarray
        :param train_labels: Training labels.
        :type train_labels: np.ndarray
        :param epochs: Number of training epochs.
        :type epochs: int
        :param validation_split: Fraction of training data to be used for validation.
        :type validation_split: float
        """
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.fit(train_data, train_labels, epochs=epochs, validation_split=validation_split)

    def apply_quantization_aware_training(self):
        """
        Applies quantization-aware training to the base model.
        """
        quantize_model = tfmot.quantization.keras.quantize_model
        self.q_aware_model = quantize_model(self.model)

        self.q_aware_model.compile(optimizer='adam',
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                   metrics=['accuracy'])

    def train_q_aware_model(self, train_data, train_labels, batch_size=500, epochs=1, validation_split=0.1):
        """
        Trains the quantization-aware model.

        :param train_data: Training dataset.
        :type train_data: np.ndarray
        :param train_labels: Training labels.
        :type train_labels: np.ndarray
        :param batch_size: Batch size for training.
        :type batch_size: int
        :param epochs: Number of training epochs.
        :type epochs: int
        :param validation_split: Fraction of training data to be used for validation.
        :type validation_split: float
        """
        self.q_aware_model.fit(train_data, train_labels,
                               batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def evaluate(self, test_data, test_labels):
        """
        Evaluates both the base model and the quantized model.

        :param test_data: Test dataset.
        :type test_data: np.ndarray
        :param test_labels: Test labels.
        :type test_labels: np.ndarray
        :return: Accuracy of base model and quantized model.
        :rtype: tuple
        """
        baseline_accuracy = self.model.evaluate(test_data, test_labels, verbose=0)[1]
        q_aware_accuracy = self.q_aware_model.evaluate(test_data, test_labels, verbose=0)[1]
        return baseline_accuracy, q_aware_accuracy

    def convert_to_tflite(self):
        """
        Converts the quantization-aware model to TensorFlow Lite format.

        :return: Quantized TFLite model.
        :rtype: bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    def post_quantization(self):
        """
        Applies post-training quantization to the base model.

        :return: Post-quantized TFLite model.
        :rtype: bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    @staticmethod
    def save_model(model_content, filename):
        """
        Saves the TFLite model to a file.

        :param model_content: The TFLite model content.
        :type model_content: bytes
        :param filename: File name to save the model.
        :type filename: str
        """
        with open(filename, 'wb') as f:
            f.write(model_content)

    @staticmethod
    def measure_model_size(filepath):
        """
        Measures the size of a model file.

        :param filepath: Path to the model file.
        :type filepath: str
        :return: Size of the model in megabytes.
        :rtype: float
        """
        return os.path.getsize(filepath) / float(2 ** 20)


