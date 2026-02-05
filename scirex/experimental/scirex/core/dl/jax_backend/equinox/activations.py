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
    Module: activation.py

    This module implements activation functions for Neural Networks

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/01/2025: Initial version

"""
import jax


def relu(x: jax.Array) -> jax.Array:
    """
    Compute ReLU activation

    Args:
        x: input data
    """
    return jax.nn.relu(x)


def relu6(x: jax.Array) -> jax.Array:
    """
    Compute ReLU6 activation

    Args:
        x: input data
    """
    return jax.nn.relu6(x)


def sigmoid(x: jax.Array) -> jax.Array:
    """
    Compute sigmoid activation

    Args:
        x: input data
    """
    return jax.nn.sigmoid(x)


def softplus(x: jax.Array) -> jax.Array:
    """
    Compute softplus activation

    Args:
        x: input data
    """
    return jax.nn.softplus(x)


def sparse_plus(x: jax.Array) -> jax.Array:
    """
    Compute sparse_plus activation

    Args:
        x: input data
    """
    return jax.nn.sparse_plus(x)


def sparse_sigmoid(x: jax.Array) -> jax.Array:
    """
    Compute sparse_sigmoid activation

    Args:
        x: input data
    """
    return jax.nn.sparse_sigmoid(x)


def soft_sign(x: jax.Array) -> jax.Array:
    """
    Compute soft_sign activation

    Args:
        x: input data
    """
    return jax.nn.soft_sign(x)


def silu(x: jax.Array) -> jax.Array:
    """
    Compute silu activation

    Args:
        x: input data
    """
    return jax.nn.silu(x)


def swish(x: jax.Array) -> jax.Array:
    """
    Compute swish activation

    Args:
        x: input data
    """
    return jax.nn.swish(x)


def log_sigmoid(x: jax.Array) -> jax.Array:
    """
    Compute log_sigmoid activation

    Args:
        x: input data
    """
    return jax.nn.log_sigmoid(x)


def leaky_relu(x: jax.Array) -> jax.Array:
    """
    Compute leaky_relu activation

    Args:
        x: input data
    """
    return jax.nn.leaky_relu(x)


def hard_sigmoid(x: jax.Array) -> jax.Array:
    """
    Compute hard_sigmoid activation

    Args:
        x: input data
    """
    return jax.nn.hard_sigmoid(x)


def hard_swish(x: jax.Array) -> jax.Array:
    """
    Compute hard_swish activation

    Args:
        x: input data
    """
    return jax.nn.hard_swish(x)


def hard_tanh(x: jax.Array) -> jax.Array:
    """
    Compute hard_tanh activation

    Args:
        x: input data
    """
    return jax.nn.hard_tanh(x)


def elu(x: jax.Array) -> jax.Array:
    """
    Compute elu activation

    Args:
        x: input data
    """
    return jax.nn.elu(x)


def celu(x: jax.Array) -> jax.Array:
    """
    Compute celu activation

    Args:
        x: input data
    """
    return jax.nn.celu(x)


def selu(x: jax.Array) -> jax.Array:
    """
    Compute selu activation

    Args:
        x: input data
    """
    return jax.nn.selu(x)


def gelu(x: jax.Array) -> jax.Array:
    """
    Compute gelu activation

    Args:
        x: input data
    """
    return jax.nn.gelu(x)


def glu(x: jax.Array) -> jax.Array:
    """
    Compute glu activation

    Args:
        x: input data
    """
    return jax.nn.glu(x)


def squareplus(x: jax.Array) -> jax.Array:
    """
    Compute squareplus activation

    Args:
        x: input data
    """
    return jax.nn.squareplus(x)


def mish(x: jax.Array) -> jax.Array:
    """
    Compute mish activation

    Args:
        x: input data
    """
    return jax.nn.mish(x)
