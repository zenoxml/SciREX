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

    This script demonstrates how to use the neural network implementation from the SciREX library to perform classification on the MNIST dataset.

    This example includes:
        - Loading the MNIST dataset using tensorflow.keras.datasets
        - Splitting the data into train and test sets
        - Training Convolutional Neural Networks
        - Evaluating and visualizing the results

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 04/01/2024: Initial version

"""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tensorflow.keras.datasets import mnist

from scirex.core.dl import Model, Network
from scirex.core.dl.utils import cross_entropy_loss, accuracy


key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)


class CNN(Network):
    layers: list

    def __init__(self):
        self.layers = [
            eqx.nn.Conv2d(1, 4, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(2, 2),
            jax.nn.relu,
            eqx.nn.Conv2d(4, 8, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(2, 2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(8 * 4 * 4, 10, key=key2),
            jax.nn.log_softmax,
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        return jnp.argmax(self(x), axis=-1)


batch_size = 10
learning_rate = 0.001
num_epochs = 10
optimizer = optax.adam(learning_rate)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[:1000].reshape(-1, 1, 28, 28) / 255.0
test_images = test_images[:1000].reshape(-1, 1, 28, 28) / 255.0
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
print("Train Images Shape: ", train_images.shape)
print("Train Labels Shape: ", train_labels.shape)

model = Model(CNN(), optimizer, cross_entropy_loss, [accuracy])
history = model.fit(train_images, train_labels, num_epochs, batch_size)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc[0]:.4f}")
model.plot_history("mnist-cnn.png")
