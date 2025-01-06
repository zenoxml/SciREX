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
   Example Script: quantization.py
   
   This script implements model optimization using TensorFlow Model Optimization Toolkit's quantization
   capabilities on neural network architectures. It demonstrates an implementation for applying
   and evaluating both quantization-aware training and post-training quantization.
   
   This example includes:
       - Implementation of QuantizationAwareTraining class
       - Building and training baseline CNN models
       - Applying quantization-aware training with TensorFlow
       - Training and evaluation workflows for both models
       - Converting models to TFLite format with optimization
       - Implementing post-training quantization for model compression

   Authors:
       - Nithyashree R (nithyashreer@iisc.ac.in)

   Version Info:
       - 06/01/2024: Initial version
"""

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

    def __init__(
        self, input_shape, num_classes, filters=12, kernel_size=(3, 3), pool_size=(2, 2)
    ):
        """
        Initializes the model architecture.

        :param input_shape: Shape of input data.
        :param num_classes: Number of output classes.
        :param filters: Number of filters for the Conv2D layer.
        :param kernel_size: Kernel size for the Conv2D layer.
        :param pool_size: Pool size for the MaxPooling2D layer.
        """
        self.model = self._build_model(
            input_shape, num_classes, filters, kernel_size, pool_size
        )
        self.q_aware_model = None

    @staticmethod
    def _build_model(input_shape, num_classes, filters, kernel_size, pool_size):
        """
        Builds the base model architecture.

        :param input_shape: Shape of input data.
        :param num_classes: Number of output classes.
        :param filters: Number of filters for the Conv2D layer.
        :param kernel_size: Kernel size for the Conv2D layer.
        :param pool_size: Pool size for the MaxPooling2D layer.
        :return: A Keras model.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Reshape(target_shape=input_shape + (1,)),
                tf.keras.layers.Conv2D(
                    filters=filters, kernel_size=kernel_size, activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=pool_size),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        return model

    def train(self, train_data, train_labels, epochs=10, validation_split=0.1):
        """
        Trains the base model.

        :param train_data: Training dataset.
        :param train_labels: Training labels.
        :param epochs: Number of training epochs.
        :param validation_split: Fraction of training data for validation.
        """
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            ],
        )

    def apply_quantization_aware_training(self):
        """
        Applies quantization-aware training to the base model.
        """
        quantize_model = tfmot.quantization.keras.quantize_model
        self.q_aware_model = quantize_model(self.model)

        self.q_aware_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train_q_aware_model(
        self, train_data, train_labels, batch_size=500, epochs=10, validation_split=0.1
    ):
        """
        Trains the quantization-aware model.

        :param train_data: Training dataset.
        :param train_labels: Training labels.
        :param batch_size: Batch size for training.
        :param epochs: Number of training epochs.
        :param validation_split: Fraction of training data for validation.
        """
        if self.q_aware_model is None:
            raise ValueError(
                "Quantization-aware model is not initialized. Call `apply_quantization_aware_training` first."
            )

        self.q_aware_model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            ],
        )

    def evaluate(self, test_data, test_labels):
        """
        Evaluates both the base model and the quantized model.

        :param test_data: Test dataset.
        :param test_labels: Test labels.
        :return: Accuracy of base model and quantized model.
        """
        baseline_accuracy = self.model.evaluate(test_data, test_labels, verbose=0)[1]
        q_aware_accuracy = self.q_aware_model.evaluate(
            test_data, test_labels, verbose=0
        )[1]
        return baseline_accuracy, q_aware_accuracy

    def convert_to_tflite(self):
        """
        Converts the quantization-aware model to TensorFlow Lite format.

        :return: Quantized TFLite model.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    def post_quantization(self):
        """
        Applies post-training quantization to the base model.

        :return: Post-quantized TFLite model.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    @staticmethod
    def save_model(model_content, filename):
        """
        Saves the TFLite model to a file.

        :param model_content: The TFLite model content.
        :param filename: File name to save the model.
        """
        with open(filename, "wb") as file:
            file.write(model_content)

    @staticmethod
    def measure_model_size(filepath):
        """
        Measures the size of a model file.

        :param filepath: Path to the model file.
        :return: Size of the model in megabytes.
        """
        return os.path.getsize(filepath) / float(2**20)
