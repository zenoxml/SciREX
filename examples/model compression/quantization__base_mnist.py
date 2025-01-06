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

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os


# Define the QuantizationAwareTraining class as provided above
class QuantizationAwareTraining:
    """
    A reusable class for performing quantization-aware training on any dataset.
    """

    def __init__(self, input_shape, num_classes):
        self.model = self._build_model(input_shape, num_classes)
        self.q_aware_model = None

    @staticmethod
    def _build_model(input_shape, num_classes):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Reshape(target_shape=input_shape + (1,)),
                tf.keras.layers.Conv2D(
                    filters=12, kernel_size=(3, 3), activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        return model

    def train(self, train_data, train_labels, epochs=1, validation_split=0.1):
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.model.fit(
            train_data, train_labels, epochs=epochs, validation_split=validation_split
        )

    def apply_quantization_aware_training(self):
        quantize_model = tfmot.quantization.keras.quantize_model
        self.q_aware_model = quantize_model(self.model)
        self.q_aware_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train_q_aware_model(
        self, train_data, train_labels, batch_size=500, epochs=1, validation_split=0.1
    ):
        self.q_aware_model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
        )

    def evaluate(self, test_data, test_labels):
        baseline_accuracy = self.model.evaluate(test_data, test_labels, verbose=0)[1]
        q_aware_accuracy = self.q_aware_model.evaluate(
            test_data, test_labels, verbose=0
        )[1]
        return baseline_accuracy, q_aware_accuracy

    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    def post_quantization(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    @staticmethod
    def save_model(model_content, filename):
        with open(filename, "wb") as f:
            f.write(model_content)

    @staticmethod
    def measure_model_size(filepath):
        return os.path.getsize(filepath) / float(2**20)


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize QuantizationAwareTraining class for the MNIST dataset
qat = QuantizationAwareTraining(input_shape=(28, 28), num_classes=10)

# Train the base model
qat.train(train_images, train_labels, epochs=1)

# Apply Quantization-Aware Training
qat.apply_quantization_aware_training()

# Train the quantization-aware model
qat.train_q_aware_model(train_images, train_labels, epochs=1)

# Evaluate both the baseline model and the quantized model
baseline_accuracy, q_aware_accuracy = qat.evaluate(test_images, test_labels)

print("Baseline model accuracy:", baseline_accuracy)
print("Quantization-aware model accuracy:", q_aware_accuracy)

# Convert the quantization-aware model to TFLite format
quantized_tflite_model = qat.convert_to_tflite()

# Save the quantized TFLite model
tflite_model_filename = "quantized_mnist_model.tflite"
qat.save_model(quantized_tflite_model, tflite_model_filename)

# Measure and print the size of the quantized TFLite model
model_size = qat.measure_model_size(tflite_model_filename)
print(f"Quantized model size: {model_size:.2f} MB")

# Optionally, you can also apply post-training quantization
post_quantized_tflite_model = qat.post_quantization()

# Save the post-quantized model
post_quantized_filename = "post_quantized_mnist_model.tflite"
qat.save_model(post_quantized_tflite_model, post_quantized_filename)

# Measure and print the size of the post-quantized TFLite model
post_quantized_model_size = qat.measure_model_size(post_quantized_filename)
print(f"Post-quantized model size: {post_quantized_model_size:.2f} MB")
