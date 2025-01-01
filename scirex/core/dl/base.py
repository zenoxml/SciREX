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
    Module: base.py

    This module implements the base class for Neural Networks

    Key Classes:
        Network: Base class for a Neural Network Architecture
        Model: Base class for a Neural Network Model

    Key Features:
        - Built on top of Jax and Equinox for efficient hardware-aware computation 
        - Optimized training using autograd and jit compilation from jax
        - Efficient neural networks implementation using equinox modules
        - Modular and extensible design for easy customization

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/01/2025: Initial version

"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Callable
from tqdm import tqdm


class Network(eqx.Module):
    """
    Base neural network class that inherits from Equinox Module.

    This class serves as a template for implementing neural network architectures.
    Subclasses should implement the __init__ and __call__ methods according to
    their specific architecture requirements.
    """

    def __init__(self):
        """Initialize the neural network architecture."""
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass of the neural network.

        Args:
            x (jax.Array): Input tensor to the network.

        Returns:
            jax.Array: Output tensor from the network.
        """
        pass

    def predict(self, x: jax.Array) -> jax.Array:
        """
        Make predictions using the network.

        Args:
            x (jax.Array): Input tensor to make predictions on.

        Returns:
            jax.Array: Predicted output from the network.
        """
        return self.__call__(x)


class Model:
    """
    High-level model class that handles training, evaluation, and prediction.

    This class implements the training loop, batch processing, and evaluation
    metrics for neural network training.

    Attributes:
        history (list): List storing training metrics for each epoch.
    """

    history = []

    def __init__(
        self,
        net: Network,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        metrics: list[Callable],
    ):
        """
        Initialize the model with network architecture and training parameters.

        Args:
            net (Network): Neural network architecture to train.
            optimizer (optax.GradientTransformation): JAX optimizer for training.
            loss_fn (Callable): Loss function for training.
            metrics (list[Callable]): List of metric functions for evaluation.
        """
        self.net = net
        self.loss_fn = loss_fn
        self._loss_fn = lambda net, x, y: loss_fn(net(x), y)
        self.optimizer = optimizer
        self.metrics = metrics

    def evaluate(self, x: jax.Array, y: jax.Array):
        """
        Evaluate the model on given data.

        Args:
            x (jax.Array): Input features for evaluation.
            y (jax.Array): Target values for evaluation.

        Returns:
            tuple: Loss value and list of metric values.
        """
        return self._evaluate(self.net, x, y)

    def fit(
        self,
        x_train: jax.Array,
        y_train: jax.Array,
        num_epochs: int = 1,
        batch_size: int = 64,
    ):
        """
        Train the model on the provided data.

        Args:
            x_train (jax.Array): Training features.
            y_train (jax.Array): Training targets.
            num_epochs (int, optional): Number of training epochs. Defaults to 1.
            batch_size (int, optional): Size of training batches. Defaults to 64.

        Returns:
            list: Training history containing metrics for each epoch.
        """
        (x_train, y_train), (x_val, y_val) = self._create_batches(
            x_train, y_train, batch_size
        )
        opt_state = self.optimizer.init(eqx.filter(self.net, eqx.is_array))
        self.history = []
        net = self.net

        for epoch in range(num_epochs):
            start_time = time.time()
            total_loss = 0.0
            for batch_x, batch_y in tqdm(
                zip(x_train, y_train), desc=f"Epoch {epoch+1}", total=len(x_train)
            ):
                avg_loss, net, opt_state = self._train_step(
                    batch_x, batch_y, net, opt_state
                )
                total_loss += avg_loss
            epoch_time = time.time() - start_time
            val_loss, val_metrics = self._evaluate(net, x_val, y_val)
            print(
                f"Epoch {epoch+1} | Loss: {total_loss/len(x_train):.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s"
            )
            self.history.append(
                {
                    "loss": total_loss / len(x_train),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "epoch_time": epoch_time,
                }
            )
        self.net = net
        return self.history

    def predict(self, x: jax.Array):
        """
        Generate predictions for given input data.

        Args:
            x (jax.Array): Input features to predict on.

        Returns:
            jax.Array: Model predictions.
        """
        return self.net.predict(x)

    def plot_history(self, metrics: list[str] = None, figsize: tuple = (12, 6)):
        """
        Plot training history metrics.

        Args:
            metrics (list[str], optional): List of metrics to plot. If None, plots loss and validation loss.
                Available metrics: 'loss', 'val_loss', and any keys in val_metrics.
            figsize (tuple, optional): Figure size (width, height). Defaults to (12, 6).

        Returns:
            tuple: Figure and axes objects.

        Raises:
            ValueError: If history is empty or requested metric is not available.
        """
        if not self.history:
            raise ValueError("No training history available. Train the model first.")

        if metrics is None:
            metrics = ["loss", "val_loss"]

        fig, ax = plt.subplots(figsize=figsize)
        epochs = range(1, len(self.history) + 1)

        for metric in metrics:
            if metric in ["loss", "val_loss"]:
                values = [epoch_data[metric] for epoch_data in self.history]
                ax.plot(epochs, values, label=metric, marker="o")
            elif metric in self.history[0]["val_metrics"]:
                metric_idx = self.metrics.index(metric)
                values = [
                    epoch_data["val_metrics"][metric_idx] for epoch_data in self.history
                ]
                ax.plot(epochs, values, label=f"val_{metric}", marker="o")
            else:
                raise ValueError(f"Metric '{metric}' not found in history.")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Training History")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        return fig, ax

    @eqx.filter_jit
    def _evaluate(self, net: Network, x: jax.Array, y: jax.Array):
        """
        Internal method for model evaluation with JIT compilation.

        Args:
            net (Network): Neural network to evaluate.
            x (jax.Array): Input features.
            y (jax.Array): Target values.

        Returns:
            tuple: Loss value and list of metric values.
        """
        output = jax.vmap(net)(x)
        pred_y = jax.vmap(net.predict)(x)
        loss = self.loss_fn(output, y)
        return loss, [f(pred_y, y) for f in self.metrics]

    @eqx.filter_jit
    def _create_batches(self, features, targets, batch_size):
        """
        Create training batches from features and targets.

        Args:
            features (jax.Array): Input features to batch.
            targets (jax.Array): Target values to batch.
            batch_size (int): Size of each batch.

        Returns:
            tuple: Tuple containing:
                - Tuple of batched features and targets
                - Tuple of validation features and targets

        Raises:
            ValueError: If batch_size is greater than number of samples.
        """
        num_complete_batches = len(features) // batch_size
        if num_complete_batches == 0:
            raise ValueError("Batch size must be greater than number of samples")
        num_complete_batches = num_complete_batches - (
            0 if num_complete_batches == 1 else 1
        )
        num_features_batched = num_complete_batches * batch_size
        batched_features = features[:num_features_batched].reshape(
            (num_complete_batches, batch_size, -1)
        )
        batched_targets = targets[:num_features_batched].reshape(
            (num_complete_batches, batch_size, -1)
        )
        validation_data = (
            features[num_features_batched:],
            targets[num_features_batched:],
        )
        return (batched_features, batched_targets), validation_data

    @eqx.filter_jit
    def _train_step(self, features, labels, net: Network, opt_state):
        """
        Perform single training step with JIT compilation.

        Args:
            features (jax.Array): Batch of input features.
            labels (jax.Array): Batch of target values.
            net (Network): Neural network to train.
            opt_state: Current optimizer state.

        Returns:
            tuple: Tuple containing:
                - Average loss for the batch
                - Updated network
                - Updated optimizer state
        """
        loss, grads = eqx.filter_value_and_grad(self._loss_fn)(net, features, labels)
        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(net, eqx.is_array)
        )
        net = eqx.apply_updates(net, updates)
        return loss / len(features), net, opt_state
