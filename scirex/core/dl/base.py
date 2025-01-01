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
        - 02/01/2025: Initial version

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
    Create your network by inheriting from this.

    This class serves as a template for implementing neural network architectures.
    Subclasses should implement the __call__ and predict methods according to
    their specific architecture requirements.

    This will make your network class a
    [dataclass](https://docs.python.org/3/library/dataclasses.html) and a
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html).

    Specify all its fields at the class level (identical to
    [dataclasses](https://docs.python.org/3/library/dataclasses.html)). This defines
    its children as a PyTree.
    """
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): Input tensor to the network.

        Returns:
            jnp.ndarray: Output tensor from the network.
        """
        pass


    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions using the network.

        Args:
            x (jnp.ndarray): Input tensor to make predictions on.

        Returns:
            jnp.ndarray: Predicted target from the network.
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
        self.optimizer = optimizer
        self.metrics = metrics


    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray):
        """
        Evaluate the model on given data.

        Args:
            x (jnp.ndarray): Input features for evaluation.
            y (jnp.ndarray): Target values for evaluation.

        Returns:
            tuple: Loss value and list of metric values.
        """
        return self._evaluate(self.net, x, y)


    def fit(
            self,
            features: jnp.ndarray,
            target: jnp.ndarray,
            num_epochs: int = 1,
            batch_size: int = 64,
    ):
        """
        Train the model on the provided data.

        Args:
            x_train (jnp.ndarray): Training features.
            y_train (jnp.ndarray): Training targets.
            num_epochs (int, optional): Number of training epochs. Defaults to 1.
            batch_size (int, optional): Size of training batches. Defaults to 64.

        Returns:
            list: Training history containing metrics for each epoch.
        """
        net = self.net
        opt_state = self.optimizer.init(eqx.filter(net, eqx.is_array))
        self.history = []
        (x_train, y_train), (x_val, y_val) = self._create_batches(
            features, target, batch_size
        )

        for epoch in range(num_epochs):
            loss, epoch_time, net, opt_state = self._epoch_step(
                x_train, y_train, net, opt_state
            )
            val_loss, val_metrics = self._evaluate(net, x_val, y_val)
            print(
                f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s"
            )
            self.history.append({
                "loss": loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "epoch_time": epoch_time,
            })

        self.net = net
        return self.history


    def predict(self, x: jnp.ndarray):
        """
        Generate predictions for given input data.

        Args:
            x (jnp.ndarray): Input features to predict on.

        Returns:
            jnp.ndarray: Model predictions.
        """
        return jax.vmap(self.net.predict)(jnp.array(x))


    @eqx.filter_jit
    def _evaluate(self, net: Network, x: jnp.ndarray, y: jnp.ndarray):
        """
        Internal method for model evaluation with JIT compilation.
        Required for evaluating the model during training with JIT.

        Args:
            net (Network): Neural network to evaluate.
            x (jnp.ndarray): Input features.
            y (jnp.ndarray): Target values.

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
            features (jnp.ndarray): Input features to batch.
            targets (jnp.ndarray): Target values to batch.
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
            (num_complete_batches, batch_size, *features.shape[1:])
        )
        batched_targets = targets[:num_features_batched].reshape(
            (num_complete_batches, batch_size, *targets.shape[1:])
        )
        validation_data = (
            features[num_features_batched:],
            targets[num_features_batched:],
        )
        return (batched_features, batched_targets), validation_data


    @eqx.filter_jit
    def _epoch_step(self, features, labels, net: Network, opt_state):
        """
        Perform single epoch with JIT compilation.

        Args:
            features (jnp.ndarray): Batch of input features.
            labels (jnp.ndarray): Batch of target values.
            net (Network): Neural network to train.
            opt_state: Current optimizer state.

        Returns:
            tuple: Tuple containing:
                - Average loss for the epoch
                - Time taken for the epoch
                - Updated network
                - Updated optimizer state
        """
        start_time = time.time()
        total_loss = 0.0
        for batch_x, batch_y in tqdm(
                zip(features, labels),
                desc="Epochs",
                total=features.shape[0],
        ):
            avg_loss, net, opt_state = self._update_step(
                batch_x,
                batch_y,
                net,
                opt_state
            )
            total_loss += avg_loss

        epoch_time = time.time() - start_time
        return total_loss / features.shape[0], epoch_time, net, opt_state


    @eqx.filter_jit
    def _update_step(self, features, labels, net: Network, opt_state):
        """
        Perform single training step with JIT compilation.

        Args:
            features (jnp.ndarray): Input features.
            labels (jnp.ndarray): Target values.
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


    def _loss_fn(self, net: Network, x: jnp.ndarray, y: jnp.ndarray):
        """
        Compute loss for the given input data.
        Required for getting gradients during training and JIT.

        Args:
            net (Network): Neural network to compute loss.
            x (jnp.ndarray): Input features for loss computation.
            y (jnp.ndarray): Target values for loss computation.

        Returns:
            jnp.ndarray: Loss value.
        """
        return jax.vmap(self.loss_fn)(jax.vmap(net)(x), y).mean()
