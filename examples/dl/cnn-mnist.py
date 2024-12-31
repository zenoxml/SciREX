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

    This script demonstrates how to use the fully connected neural network implementation from the SciREX library to perform classification on the MNIST dataset.

    This example includes:
        - Loading the MNIST dataset using tensorflow.keras.datasets
        - Splitting the data into train and test sets
        - Training Convolutional Neural Networks
        - Evaluating and visualizing the results

    Dependencies:
        - jax (for automatic differentiation and jit compilation)
        - equinox (for neural network modules)
        - optax (for optimization)
        - tensorflow.keras.datasets (for dataset)
        - matplotlib (for visualization)
        - scirex.core.dl.fcnn

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 30/Dec/2024: Initial version

"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tensorflow.keras.datasets import mnist
import time
from typing import Tuple, Dict
import matplotlib.pyplot as plt

from scirex.core.dl.fcnn import (
    Model,
    evaluate,
    train,
)


def load_mnist(batch_size: int) -> Tuple[Tuple, Tuple]:
    """
    Load and preprocess MNIST dataset
    Returns:
        ((train_images, train_labels), (test_images, test_labels))
        each split into batches
    """
    # Load MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Create training data
    train_images = jnp.array(train_images)[..., None]  # Add channel dimension
    train_images = train_images.astype(jnp.float32) / 255.0
    train_labels = jnp.array(train_labels, dtype=jnp.int32)

    # Create test data
    test_images = jnp.array(test_images)[..., None]
    test_images = test_images.astype(jnp.float32) / 255.0
    test_labels = jnp.array(test_labels, dtype=jnp.int32)

    def create_batches(images, labels):
        # Calculate number of complete batches
        num_complete_batches = len(images) // batch_size

        # Reshape into batches
        batched_images = images[: num_complete_batches * batch_size].reshape(
            (num_complete_batches, batch_size, *images.shape[1:])
        )
        batched_labels = labels[: num_complete_batches * batch_size].reshape(
            (num_complete_batches, batch_size)
        )

        return batched_images, batched_labels

    return create_batches(train_images, train_labels), create_batches(
        test_images, test_labels
    )


def plot_training_history(history: Dict, model_type: str):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title(f"{model_type} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.title(f"{model_type} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("cnn-mnist.png")


if __name__ == "__main__":
    # Set random seed
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10

    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_mnist(batch_size)

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate)  # Gradient clipping
    )

    # Train CNN
    print(f"\nTraining CNN...")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training batches per epoch: {len(train_images)}")
    print(f"Test batches per epoch: {len(test_images)}")
    keys = jax.random.split(key, 4)
    layers = [
        lambda x: x.T
        eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, key=keys[0]),
        jax.nn.relu,
        eqx.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, key=keys[1]),
        jax.nn.relu,
        eqx.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, key=keys[2]),
        jax.nn.relu,
        # Flatten 
        lambda x: x.reshape(-1),
        eqx.nn.Linear(64 * 22 * 22, 10, key=keys[3]),
    ]
    trained_model, history = train(
        model=Model(layers),
        x_train=train_images,
        y_train=train_labels,
        x_test=test_images,
        y_test=test_labels,
        optim=optimizer,
        num_epochs=num_epochs,
        print_every=100,
    )

    # Final evaluation
    print("\nFinal Results:")
    print("-" * 50)

    # CNN results
    test_loss, test_acc = evaluate(trained_model, test_images, test_labels)
    print(f"CNN Test Loss: {test_loss:.4f}")
    print(f"CNN Test Accuracy: {test_acc:.4f}")

    # Plot training history
    plot_training_history(history, "CNN")
