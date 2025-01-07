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
    Module: others.py

    This module implements other helpful functions for Neural Networks

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/01/2025: Initial version
        - 06/01/2025: Rename and add other functions from jax.nn

"""
import jax


def softmax(x: jax.Array) -> jax.Array:
    """
    Compute softmax activation

    Args:
        x: input data
    """
    return jax.nn.softmax(x)


def log_softmax(x: jax.Array) -> jax.Array:
    """
    Compute log_softmax activation

    Args:
        x: input data
    """
    return jax.nn.log_softmax(x)


def standardize(x: jax.Array) -> jax.Array:
    """
    Compute standardize activation

    Args:
        x: input data
    """
    return jax.nn.standardize(x)


def one_hot(x: jax.Array) -> jax.Array:
    """
    Compute one_hot activation

    Args:
        x: input data
    """
    return jax.nn.one_hot(x)
