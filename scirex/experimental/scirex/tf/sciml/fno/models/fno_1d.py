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
Module: fno_1d.py

This module provides the implementation of a basic one-dimensional Fourier Neural Operator (FNO) model.

Classes:
    FNO1d: 1D Fourier Neural Operator

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
    29/Dec/2024: Initial version - Diya Nag Chaudhury
    29/Jan/2025: Minor changes: Diya Nag Chaudhury

References:
    None
"""

import jax
import equinox as eqx
import jax.numpy as jnp

from typing import List

from ..layers.fno_block_1d import FNOBlock1d


class FNO1d(eqx.Module):
    """
    A 1D Fourier Neural Operator.

    This model consists of a lifting layer, followed by a series of FNO blocks,
    and a projection layer.

    Attributes:
    lifting: eqx.nn.Conv1d
    fno_blocks: List[FNOBlock1d]
    projection: eqx.nn.Conv1d

    Methods:
    __init__: Initializes the FNO1d object
    __call__: Calls the FNO1d object
    """

    lifting: eqx.nn.Conv1d
    fno_blocks: List[FNOBlock1d]
    projection: eqx.nn.Conv1d

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        width,
        activation,
        n_blocks,
        *,
        key,
    ):
        """
        Constructor for the FNO1d class.

        Args:
        in_channels: int
        out_channels: int
        modes: int
        width: int
        activation: Callable
        n_blocks: int
        key: jax.random.PRNGKey

        Usage:
        fno = FNO1d(in_channels, out_channels, modes, width, activation, n_blocks, key)

        """
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv1d(
            in_channels,
            width,
            1,
            key=lifting_key,
        )

        self.fno_blocks = []
        for i in range(n_blocks):
            key, subkey = jax.random.split(key)
            self.fno_blocks.append(
                FNOBlock1d(
                    width,
                    width,
                    modes,
                    activation,
                    key=subkey,
                )
            )

        key, projection_key = jax.random.split(key)
        self.projection = eqx.nn.Conv1d(
            width,
            out_channels,
            1,
            key=projection_key,
        )

    def __call__(
        self,
        x,
    ):
        """
        Forward pass of the FNO1d model.

        Args:
        x: jnp.ndarray

        Returns:
        x: jnp.ndarray

        Raises:
        None

        Usage:
        x = fno(x)

        """

        x = self.lifting(x)

        for fno_block in self.fno_blocks:
            x = fno_block(x)

        x = self.projection(x)

        return x
