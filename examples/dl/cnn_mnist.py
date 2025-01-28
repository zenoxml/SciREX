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
    Example Script: cnn-mnist.py

    This script demonstrates how to use the neural network implementation from
    the SciREX library to perform classification on the MNIST dataset.

    This example includes:
        - Loading the MNIST dataset using tensorflow.keras.datasets
        - Training Convolutional Neural Networks
        - Evaluating and visualizing the results

    Key Features:
        - Uses cross-entropy loss for training
        - Implements accuracy metric for evaluation
        - Includes model checkpointing
        - Provides training history visualization

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 04/01/2024: Initial version
        - 06/01/2024: Update imports

"""
from os import path
import jax
import jax.numpy as jnp
import optax
from tensorflow.keras.datasets import mnist

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from scirex.core.dl import Model, Network
from scirex.core.dl.nn.loss import cross_entropy_loss
from scirex.core.dl.nn.metrics import accuracy
import scirex.core.dl.nn as nn

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)


class CNN(Network):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
    - Conv2D: 1->4 channels, 4x4 kernel
    - MaxPool2D: 2x2 pooling
    - ReLU activation
    - Conv2D: 4->8 channels, 4x4 kernel
    - MaxPool2D: 2x2 pooling
    - ReLU activation
    - Flatten
    - Dense: 8*4*4->10 units
    - LogSoftmax activation
    """

    layers: list

    def __init__(self):
        """Initialize the CNN architecture with predefined layers."""
        self.layers = [
            nn.Conv2d(1, 4, kernel_size=4, key=key1),  # First conv layer: 1->4 channels
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions
            nn.relu,  # Activation function
            nn.Conv2d(
                4, 8, kernel_size=4, key=key1
            ),  # Second conv layer: 4->8 channels
            nn.MaxPool2d(2, 2),  # Further reduce dimensions
            nn.relu,  # Activation function
            jnp.ravel,  # Flatten for dense layer
            nn.Linear(8 * 4 * 4, 10, key=key2),  # Output layer: 10 classes
            nn.log_softmax,  # For numerical stability
        ]

    def __call__(self, x):
        """
        Forward pass through the network.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        """
        Generate class predictions from model outputs.
        """
        return jnp.argmax(self(x), axis=-1)


# Training hyperparameters
batch_size = 10  # Number of samples per batch
learning_rate = 0.001  # Learning rate for Adam optimizer
num_epochs = 10  # Number of training epochs
optimizer = optax.adam(learning_rate)

# Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Take subset of data for quick demonstration
train_images = train_images[:1000].reshape(-1, 1, 28, 28) / 255.0  # Normalize to [0,1]
test_images = test_images[:1000].reshape(-1, 1, 28, 28) / 255.0
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

print("Train Images Shape: ", train_images.shape)
print("Train Labels Shape: ", train_labels.shape)

# Initialize model with CNN architecture
model = Model(CNN(), optimizer, cross_entropy_loss, [accuracy])

# Train model if no saved checkpoint exists
if not path.exists("mnist-cnn.dl"):
    history = model.fit(train_images, train_labels, num_epochs, batch_size)
    model.save_net("mnist-cnn.dl")
else:
    print("Loading the model from mnist-cnn.dl")
    model.load_net("mnist-cnn.dl")

# Evaluate model performance
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc[0]:.4f}")

# Save training history plot
model.plot_history("mnist-cnn.png")

# Confusion Matrix
pred_labels = model.predict(test_images)
# Display confusion matrix
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
disp = ConfusionMatrixDisplay(
    confusion_matrix(test_labels, pred_labels), display_labels=classes
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# classification_report
print("\n\nReport: \n", classification_report(test_labels, pred_labels))
