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
    Module: utils.py

    This module implements utility functions for Neural Networks

    Key Features:
        - Contains commonly used loss functions and metrics for neural networks

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/01/2025: Initial version

"""
import jax
import jax.numpy as jnp
import optax


def accuracy(pred_y: jax.Array, y: jax.Array) -> float:
    """
    Compute accuracy

    Args:
        x: input data
        y: target labels
    """
    return jnp.mean(pred_y == y)


def mse_loss(output: jax.Array, y: jax.Array) -> float:
    """
    Compute mean squared error loss

    Args:
        output: output of the model
        y: target values
    """
    return jnp.mean(jnp.square(output - y))


def cross_entropy_loss(output: jax.Array, y: jax.Array) -> float:
    """
    Compute the cross-entropy loss

    Args:
        output: output of the model
        y: Batched target labels
    """

    n_classes = output.shape[-1]
    loss = optax.softmax_cross_entropy(output, jax.nn.one_hot(y, n_classes)).mean()
    return loss
