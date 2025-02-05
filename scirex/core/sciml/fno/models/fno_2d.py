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
Module: fno_2d.py

This module provides the implementation of the two-dimensional Fourier Neural Operator (FNO) model.

Classes:
    FNO2d: 2D Fourier Neural Operator

Dependencies:
    - jax: For array processing
    - equinox: For neural network layers

Key Features:
    - Lifting layer
    - FNO blocks
    - Projection layer

Authors:
    Diya Nag Chaudhury

Version Info:
    28/Jan/2025: Initial version - Diya Nag Chaudhury

References:
    None
"""

import jax
import equinox as eqx
import jax.numpy as jnp

from typing import List

from ..layers.fno_block_2d import FNOBlock2d


class FNO2d(eqx.Module):
    """Complete 2D Fourier Neural Operator

    This module combines the lifting layer, FNO blocks, and projection layer
    to create a complete 2D Fourier Neural Operator.

    Attributes:
    lifting: eqx.nn.Conv2d
    fno_blocks: list
    projection: eqx.nn.Conv2d

    Methods:
    __init__: Initializes the FNO2d object
    __call__: Calls the FNO2d object
    """

    lifting: eqx.nn.Conv2d
    fno_blocks: list
    projection: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
        width,
        activation,
        n_blocks,
        *,
        key,
    ):
        """
        Initializes the FNO2d object

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            modes1 (_type_): _description_
            modes2 (_type_): _description_
            width (_type_): _description_
            activation (_type_): _description_
            n_blocks (_type_): _description_
            key (_type_): _description_
        """

        keys = jax.random.split(key, n_blocks + 2)

        self.lifting = eqx.nn.Conv2d(in_channels, width, kernel_size=1, key=keys[0])

        self.fno_blocks = []
        for i in range(n_blocks):
            self.fno_blocks.append(
                FNOBlock2d(width, width, modes1, modes2, activation, key=keys[i + 1])
            )

        self.projection = eqx.nn.Conv2d(
            width, out_channels, kernel_size=1, key=keys[-1]
        )

    def __call__(self, x):
        """_
        Calls the FNO2d object

        Args:
        x: jnp.ndarray

        Returns:
        jnp.ndarray
        """
        x = self.lifting(x)

        for block in self.fno_blocks:
            x = block(x)

        x = self.projection(x)
        return x
