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

# This class performs pruning on a given model using TensorFlow Model Optimization, trains it, and evaluates both the baseline and pruned models.
# It also exports the pruned model in Keras and TFLite formats and provides methods for calculating the gzipped model size.


import tempfile
import os
import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as tfmot

class ModelPruning:
    """
    A reusable class for performing pruning on any model and dataset.

    Attributes:
        model (tf.keras.Model): The base model architecture.
        pruned_model (tf.keras.Model): The pruned model after pruning operation.
        baseline_model_accuracy (float): Accuracy of the baseline model on the test set.
        pruned_model_accuracy (float): Accuracy of the pruned model on the test set.
    """

    def __init__(self, model, train_data, test_data, epochs=2, batch_size=128, validation_split=0.1):
        """
        Initializes the pruning process.

        :param model: The base model architecture to be pruned.
        :param train_data: Tuple of (train_images, train_labels) for training.
        :param test_data: Tuple of (test_images, test_labels) for testing.
        :param epochs: Number of epochs to train the pruned model. Default is 2.
        :param batch_size: Size of the training batch. Default is 128.
        :param validation_split: Fraction of training data to be used for validation. Default is 0.1.
        """
        self.model = model
        self.train_images, self.train_labels = train_data
        self.test_images, self.test_labels = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.pruned_model = None
        self.baseline_model_accuracy = None
        self.pruned_model_accuracy = None

    def _compute_end_step(self):
        """
        Compute the number of steps to finish pruning after the specified number of epochs.

        :return: The end step for pruning.
        """
        num_images = self.train_images.shape[0] * (1 - self.validation_split)
        return np.ceil(num_images / self.batch_size).astype(np.int32) * self.epochs

    def _apply_pruning(self):
        """
        Applies pruning to the model using PolynomialDecay schedule.

        :return: A pruned model.
        """
        end_step = self._compute_end_step()

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.50,
                final_sparsity=0.80,
                begin_step=0,
                end_step=end_step
            )
        }

        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(self.model, **pruning_params)
        pruned_model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return pruned_model

    def train_and_evaluate(self):
        """
        Train and evaluate the baseline model and pruned model.

        :return: None
        """
        # Train baseline model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.fit(self.train_images, self.train_labels, epochs=self.epochs, validation_split=self.validation_split)
        
        # Evaluate baseline model
        _, self.baseline_model_accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        print('Baseline test accuracy:', self.baseline_model_accuracy)

        # Apply pruning and train pruned model
        self.pruned_model = self._apply_pruning()
        
        # Callbacks for pruning updates
        logdir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]
        
        # Train pruned model
        self.pruned_model.fit(self.train_images, self.train_labels,
                              batch_size=self.batch_size, epochs=self.epochs,
                              validation_split=self.validation_split, callbacks=callbacks)
        
        # Evaluate pruned model
        _, self.pruned_model_accuracy = self.pruned_model.evaluate(self.test_images, self.test_labels, verbose=0)
        print('Pruned test accuracy:', self.pruned_model_accuracy)

        # Export pruned model
        self._export_pruned_model()

    def _export_pruned_model(self):
        """
        Strips pruning and saves the pruned model in both Keras and TFLite formats.
        """
        model_for_export = tfmot.sparsity.keras.strip_pruning(self.pruned_model)

        # Save pruned model to Keras format
        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        print('Saved pruned Keras model to:', pruned_keras_file)

        # Convert pruned model to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        pruned_tflite_model = converter.convert()

        _, pruned_tflite_file = tempfile.mkstemp('.tflite')
        with open(pruned_tflite_file, 'wb') as f:
            f.write(pruned_tflite_model)

        print('Saved pruned TFLite model to:', pruned_tflite_file)

    @staticmethod
    def get_gzipped_model_size(file):
        """
        Returns the size of a gzipped model file.

        :param file: Path to the model file.
        :return: Size of the gzipped model file in bytes.
        """
        import zipfile
        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file)

        return os.path.getsize(zipped_file)
