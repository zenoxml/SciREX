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
    Module: fcnn.py

    This module implements Fully Connected Neural Networks

    It provides functionality to:
      - Train Fully Connected Neural Networks on a batched dataset
      - Evaluate model performance using classification metrics

    Classes:
        Model: Implements a Fully Connected Neural Network using Jax.

    Dependencies:
        - jax (for automatic differentiation and jit compilation)
        - equinox (for neural network modules)
        - jaxtyping (for type annotations)
        - optax (for optimization)

    Key Features:
        - Support for binary and multi-class classification using neural networks
        - Optimized training using autograd and jit compilation from jax
        - Efficient neural networks implementation using equinox modules

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 30/Dec/2024: Initial version

"""
import time
import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
import optax
from jaxtyping import Array, Float, PyTree
from typing import Dict, Any
from typing import Callable, Tuple


class Model(eqx.Module):
    """A simple neural network model"""

    layers: list

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: Array) -> Array:
        """Forward pass of the model"""
        for layer in self.layers:
            x = layer(x)
        return x


def cross_entropy_loss(model: Model, x: Array, y: Array) -> Float[Array, ""]:
    """
    Compute the cross-entropy loss

    Args:
        model: Neural network model
        x: Batched input data
        y: Batched target labels
    """
    logits = jax.vmap(model)(x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, jnp.expand_dims(y, 1), axis=1)
    return jnp.mean(nll)


@eqx.filter_jit
def compute_accuracy(model: Model, x: Array, y: Array) -> Float[Array, ""]:
    """
    Compute accuracy

    Args:
        model: Neural network model
        x: Batch of input data
        y: Batch of target labels
    """
    pred_y = jax.vmap(model)(x)
    pred_class = jnp.argmax(pred_y, axis=1)
    return jnp.mean(pred_class == y)


def evaluate(
    model: Model, x: Array, y: Array
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """
    Evaluate model on data

    Args:
        model: Neural network model
        x: Batched input data
        y: Batched target labels

    Returns:
        Tuple of (average loss, average accuracy)
    """
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(x)

    for batch_x, batch_y in zip(x, y):
        loss = cross_entropy_loss(model, batch_x, batch_y)
        acc = compute_accuracy(model, batch_x, batch_y)
        total_loss += loss
        total_acc += acc

    return total_loss / n_batches, total_acc / n_batches


@eqx.filter_jit
def make_step(
    model: Model,
    opt_state: PyTree,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformation,
) -> Tuple[Model, PyTree, Float[Array, ""]]:
    """
    Perform a single training step

    Args:
        model: Neural network model
        opt_state: Optimizer state
        x: Batch of input data
        y: Batch of target labels
        optimizer: Optax optimizer

    Returns:
        Tuple of (updated model, updated optimizer state, loss value)
    """
    loss_val, grads = eqx.filter_value_and_grad(cross_entropy_loss)(model, x, y)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def train(
    model: Model,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    optim: optax.GradientTransformation,
    num_epochs: int,
    print_every: int,
) -> Tuple[Model, Dict[str, list[Any]]]:
    """
    Train the model

    Args:
        model: Neural network model
        x_train: Training data batches (shape: [num_batches, batch_size, ...])
        y_train: Training labels batches (shape: [num_batches, batch_size])
        x_test: Test data batches
        y_test: Test labels batches
        optim: Optax optimizer
        num_epochs: Number of training epochs
        print_every: Print progress every N batches

    Returns:
        Trained model and history
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    num_batches = len(x_train)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_times": [],
    }

    for epoch in range(num_epochs):
        total_loss = 0.0
        start_time = time.time()

        # Training loop
        for batch_idx, (batch_x, batch_y) in enumerate(zip(x_train, y_train)):
            # Perform optimization step
            model, opt_state, loss_val = make_step(
                model, opt_state, batch_x, batch_y, optim
            )
            total_loss += loss_val

            # Print progress
            if (batch_idx + 1) % print_every == 0:
                avg_loss = total_loss / print_every
                test_loss, test_acc = evaluate(model, x_test, y_test)
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}"
                )
                print(
                    f"Train Loss: {loss_val:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                )
                total_loss = 0.0

        # End of epoch evaluation
        end_time = time.time()
        time_taken = end_time - start_time
        test_loss, test_acc = evaluate(model, x_test, y_test)
        train_loss, train_acc = evaluate(model, x_train, y_train)
        print(f"\nEpoch {epoch+1}/{num_epochs} completed")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        history["epoch_times"].append(time_taken)

    return model, history
