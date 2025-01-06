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
Example Script: pruning.py

    This script implements model compression using TensorFlow Model Optimization Toolkit's pruning capabilities 
    on neural network architectures. It demonstrates a reusable implementation for applying and evaluating 
    model pruning with polynomial decay schedule.

    This example includes:
        - Implementation of a reusable ModelPruning class for model compression
        - Building and training baseline CNN models
        - Applying progressive pruning from 50% to 80% sparsity
        - Training and evaluation workflows for both baseline and pruned models
        
        
     Authors: 
     - Nithyashree R (nithyashreer@iisc.ac.in)

    Version Info:
        - 06/01/2024: Initial version
        
"""

import tempfile
import os
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras


class ModelPruning:
    """
    A reusable class for performing pruning on any model and dataset.

    Attributes:
        model (tf.keras.Model): The base model architecture that will undergo pruning.
        pruned_model (tf.keras.Model): The model after pruning.
        baseline_model_accuracy (float): Accuracy of the baseline model evaluated on test data.
        pruned_model_accuracy (float): Accuracy of the pruned model evaluated on test data.
    """

    def __init__(
        self,
        input_shape=(28, 28),
        num_classes=10,
        epochs=10,
        batch_size=35,  # Changed default batch_size to get ~1688 steps
        validation_split=0.1,
    ):
        """
        Initializes the pruning process for a model.

        :param input_shape: Shape of the input data.
        :type input_shape: tuple
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param epochs: Number of epochs to train the pruned model. Default is 10.
        :type epochs: int
        :param batch_size: Size of the training batch. Default is 35.
        :type batch_size: int
        :param validation_split: Fraction of training data to be used for validation. Default is 0.1.
        :type validation_split: float
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = self._build_model()
        self.pruned_model = None
        self.baseline_model_accuracy = None
        self.pruned_model_accuracy = None

    def _build_model(self):
        """
        Builds the base model architecture.

        :return: A compiled Keras model.
        :rtype: tf.keras.Model
        """
        model = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=self.input_shape),
                keras.layers.Reshape(target_shape=(28, 28, 1)),
                keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(self.num_classes),
            ]
        )
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def train_baseline_model(self, train_images, train_labels):
        """
        Trains the baseline model without pruning.

        :param train_images: Training data features.
        :param train_labels: Training data labels.
        """
        self.model.fit(
            train_images,
            train_labels,
            batch_size=self.batch_size,  # Added batch_size parameter
            epochs=self.epochs,
            validation_split=self.validation_split,
        )

    def evaluate_baseline(self, test_images, test_labels):
        """
        Evaluates the baseline model.

        :param test_images: Test data features.
        :param test_labels: Test data labels.
        :return: Accuracy of the baseline model.
        :rtype: float
        """
        _, self.baseline_model_accuracy = self.model.evaluate(
            test_images, test_labels, verbose=0
        )
        return self.baseline_model_accuracy

    def save_baseline_model(self):
        """
        Saves the baseline model to a temporary file using the .keras format.

        :return: Path to the saved model file.
        :rtype: str
        """
        keras_file = tempfile.mktemp(".keras")
        self.model.save(keras_file, save_format="keras")
        return keras_file

    def apply_pruning(self):
        """
        Applies pruning to the base model.

        :return: A pruned model.
        :rtype: tf.keras.Model
        """
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        batch_size = self.batch_size
        epochs = self.epochs
        validation_split = self.validation_split

        # Ensure the model is built before accessing input_shape
        self.model.build((None, *self.input_shape))
        num_images = 60000 * (
            1 - validation_split
        )  # Fixed number of training images for MNIST

        end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.50,
                final_sparsity=0.80,
                begin_step=0,
                end_step=end_step,
            )
        }

        self.pruned_model = prune_low_magnitude(self.model, **pruning_params)
        self.pruned_model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        return self.pruned_model

    def train_pruned_model(self, train_images, train_labels):
        """
        Trains the pruned model.

        :param train_images: Training data features.
        :param train_labels: Training data labels.
        """
        logdir = tempfile.mkdtemp()

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        self.pruned_model.fit(
            train_images,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
        )

    def evaluate_pruned_model(self, test_images, test_labels):
        """
        Evaluates the pruned model.

        :param test_images: Test data features.
        :param test_labels: Test data labels.
        :return: Accuracy of the pruned model.
        :rtype: float
        """
        _, self.pruned_model_accuracy = self.pruned_model.evaluate(
            test_images, test_labels, verbose=0
        )
        return self.pruned_model_accuracy
