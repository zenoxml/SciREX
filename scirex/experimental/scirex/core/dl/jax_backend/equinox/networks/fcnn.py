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
      - Train Fully Connected Neural Networks

    Classes:
        FCNN: Implements a Fully Connected Neural Network building up on `base.py`.

    Key Features:
        - Built on top of base class getting all its functionalities
        - Efficient neural networks implementation using equinox modules

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 03/01/2025: Initial version

"""
import jax.numpy as jnp
from ..base import Network


class FCNN(Network):
    layers: list

    """
    Fully Connected Neural Network
    """

    def __init__(self, layers: list):
        """
        Constructor for Fully Connected Neural Network

        Args:
            layers: List of layers
        """
        self.layers = layers

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the Fully Connected Neural Network
        Args:
            x: Input tensor

        Returns:
            jnp.ndarray: Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
