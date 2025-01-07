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
    Example Script: vae-mnist.py

    This script demonstrates how to use the neural network implementation from
    the SciREX library to variational auto-encoder on the MNIST dataset.

    This example includes:
        - Loading the MNIST dataset using tensorflow.keras.datasets
        - Training Variational Auto-Encoder
        - Evaluating and visualizing the results

    Key features:
        - Encoder-decoder architecture
        - Latent space sampling with reparameterization trick
        - KL divergence loss for regularization
        - Model checkpointing and visualization

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 06/01/2024: Initial version

"""

from os import path
import jax
import jax.numpy as jnp
import optax
from tensorflow.keras.datasets import mnist

from scirex.core.dl import Model, Network, FCNN, nn

# Generate random keys for layer initialization
keys = jax.random.split(jax.random.PRNGKey(0), 3)

# Define encoder architecture
encoderLayers = [
    nn.Conv2d(1, 4, kernel_size=4, key=keys[0]),  # First conv layer
    nn.MaxPool2d(2, 2),  # Reduce spatial dimensions
    nn.relu,  # Activation
    nn.Conv2d(4, 8, kernel_size=4, key=keys[1]),  # Second conv layer
    nn.MaxPool2d(2, 2),  # Further reduce dimensions
    nn.relu,  # Activation
    jnp.ravel,  # Flatten for dense layer
    nn.Linear(8 * 4 * 4, 5, key=keys[2]),  # Project to latent space
    nn.log_softmax,  # For numerical stability
]

# Define decoder architecture
decoderLayers = [
    nn.Linear(4, 64, key=keys[0]),  # First dense layer
    nn.relu,  # Activation
    nn.Linear(64, 128, key=keys[1]),  # Second dense layer
    nn.relu,  # Activation
    nn.Linear(128, 784, key=keys[2]),  # Output layer (28*28=784)
    nn.sigmoid,
    lambda x: x.reshape(-1, 28, 28),  # Reshape to image dimensions
]


class VAE(Network):
    """
    Variational Autoencoder implementation.

    The VAE consists of:
    1. An encoder that maps inputs to latent space parameters
    2. A sampling layer that uses the reparameterization trick
    3. A decoder that reconstructs inputs from latent samples

    Attributes:
        encoder (FCNN): Neural network for encoding
        decoder (FCNN): Neural network for decoding
    """

    encoder: FCNN
    decoder: FCNN

    def __init__(self, encoderLayers, decoderLayers):
        """
        Initialize VAE with encoder and decoder architectures.

        Args:
            encoderLayers (list): Layer definitions for encoder
            decoderLayers (list): Layer definitions for decoder
        """
        self.encoder = FCNN(encoderLayers)
        self.decoder = FCNN(decoderLayers)

    def __call__(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (jax.Array): Input image tensor

        Returns:
            jax.Array: Reconstructed image tensor
        """
        # Encode input to get latent parameters
        x = self.encoder(x)
        # Split into mean and log standard deviation
        mean, stddev = x[:-1], jnp.exp(x[-1])
        # Sample from latent space using reparameterization trick
        z = mean + stddev * jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        # Decode latent sample
        return self.decoder(z)


def loss_fn(output, y):
    """
    Compute KL divergence loss between output and target.

    Args:
        output (jax.Array): Model output
        y (jax.Array): Target values

    Returns:
        float: Loss value
    """
    return jnp.abs(nn.kl_divergence(output.reshape(-1), y.reshape(-1)))


# Initialize model with VAE architecture
model = Model(VAE(encoderLayers, decoderLayers), optax.adam(1e-3), loss_fn)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize images
x_train = x_train.reshape(-1, 1, 28, 28) / 255.0
x_test = x_test.reshape(-1, 1, 28, 28) / 255.0

# Load or train model
if path.exists("mnist_vae.dl"):
    print("Loading model from disk")
    model.load_net("mnist_vae.dl")
else:
    # Train for 100 epochs with batch size 64
    model.fit(x_train, x_train, num_epochs=100, batch_size=64)
    model.plot_history("mnist_vae.png")
    model.save_net("mnist_vae.dl")

# Evaluate model performance
test_loss, test_acc = model.evaluate(x_test, x_test)
print(f"Test Loss: {test_loss:.4f}")
