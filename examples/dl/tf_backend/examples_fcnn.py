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
    Example Script: example_fcnn.py

    This script demonstrates how to use the neural network implementation from
    the tensorflow backend of the SciREX library to perform classification.

    Authors:
        - Divij Ghose

    Version Info:
        - 17/01/2024: Initial version

"""

import numpy as np
import tensorflow as tf
from scirex.core.dl.tf_backend.networks.fcnn import FullyConnectedNetwork
from scirex.core.dl.tf_backend.optimizers import get_optimizer
from scirex.core.dl.tf_backend.datautils import get_batches, get_devices, datatypes
from scirex.core.dl.tf_backend.mathutils import *

datatypes = datatypes()

# Check available devices
devices = get_devices()

# Generate some sample data
np.random.seed(0)
X = np.random.rand(1000, 2).astype(np.float32)  # 1000 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(np.float32).reshape(-1, 1)  # Simple binary classification

# Split into train and test sets
train_size = int(0.8 * len(X)) # 80% of the data for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Convert data to tensors with correct dtype
X_train = tf.convert_to_tensor(X_train, dtype=datatypes["float32"])
y_train = tf.convert_to_tensor(y_train, dtype=datatypes["float32"])
X_test = tf.convert_to_tensor(X_test, dtype=datatypes["float32"])
y_test = tf.convert_to_tensor(y_test, dtype=datatypes["float32"])
# Define network architecture
architecture = [2, 32, 32, 1]  # Input size: 2, Hidden layers: 32, 32, Output size: 1

# Create the network
network = FullyConnectedNetwork(
    architecture=architecture,
    hidden_activation="relu",
    output_activation=None,
    dtype=datatypes["float32"],
)

# Print network summary
print(network.summary())

# Define training parameters
learning_rate_dict = {
    "initial_learning_rate": 0.001,
    "use_lr_scheduler": True,
    "decay_rate": 0.9,
    "decay_steps": 1000,
    "staircase": False,
}

# Get optimizer
optimizer = get_optimizer("adam", learning_rate_dict)

# Training parameters
batch_size = 32
num_epochs = 500
buffer_size = 1024

# Create batched dataset
train_dataset = get_batches(X_train, y_train, batch_size, buffer_size)


# Training loop
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # Forward pass
        logits = network(x_batch)
        # Compute loss
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_batch,
                logits=logits
            )
        )

    # Compute gradients
    gradients = tape.gradient(loss, network.get_weights())
    # Update weights
    optimizer.apply_gradients(zip(gradients, network.get_weights()))

    return loss


# Training loop
print("\nStarting training...")
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    for x_batch, y_batch in train_dataset:
        batch_loss = train_step(x_batch, y_batch)
        epoch_loss += batch_loss
        num_batches += 1

    epoch_loss /= num_batches

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluate on test set
logits = network(X_test)
y_pred = tf.sigmoid(logits)
test_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_test, y_pred))
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y_test), tf.float32))

print("\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Example prediction
sample_input = np.array([[0.7, 0.3], [0.2, 0.8]])
predictions = network(sample_input)
print("\nSample Predictions:")
print("Input:", sample_input)
print("Predictions:", predictions.numpy())
