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
    Module: mlp.py

    This module implements Multi-Layer Perceptron (MLP) neural network architecture.

    Key Classes:
        MLP: Multi-Layer Perceptron

    Key Features:
        - Built on top of base class getting all its functionalities
        - Efficient neural networks implementation using equinox modules

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 02/01/2025: Initial version

"""
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable
from .fcnn import FCNN


class MLP(FCNN):
    """
    Multi-Layer Perceptron
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = 0,
        depth: int = 0,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = lambda x: x,
        random_seed: int = 0,
    ):
        """
        Constructor for Multi-Layer Perceptron

        Args:
            in_size: Input size
            out_size: Output size
            hidden_size: Hidden size
            depth: Depth of the network
            activation: Activation function
            final_activation: Final activation function
            random_seed: Random seed
        """
        key = jax.random.PRNGKey(random_seed)
        if depth == 0:
            self.layers = [eqx.nn.Linear(in_size, out_size, key=key)]
        else:
            self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key), jax.nn.relu]
            for _ in range(depth - 1):
                self.layers += [
                    eqx.nn.Linear(hidden_size, hidden_size, key=key),
                    jax.nn.relu,
                ]
            self.layers += [eqx.nn.Linear(hidden_size, out_size, key=key)]

        self.layers += [final_activation]
