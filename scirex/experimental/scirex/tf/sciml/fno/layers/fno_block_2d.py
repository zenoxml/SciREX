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
Module: fno_block_2d.py

This module provides the implementation of a single block of the Fourier Neural Operator (FNO) model.

Classes:
    FNOBlock1d: Single block of the FNO model

Dependencies:
    - jax: For array processing
    - equinox: For neural network layers

Key Features:
    - Spectral convolution
    - Bypass convolution
    - Activation function

Authors:
    Diya Nag Chaudhury

Version Info:
    28/Jan/2024: Initial version - Diya Nag Chaudhury

References:
    None
"""

import jax
import equinox as eqx
import jax.numpy as jnp

from typing import Callable

from scirex.core.sciml.fno.layers.spectral_conv_2d import SpectralConv2d


class FNOBlock2d(eqx.Module):
    """2D FNO Block combining spectral and regular convolution

    This block combines a spectral convolution with a regular convolution

    Args:
        in_channels: int
        out_channels: int
        modes1: int
        modes2: int
        activation: callable
        key: jax.random

    Returns:
        callable: FNOBlock2d object

    """

    spectral_conv: SpectralConv2d
    conv: eqx.nn.Conv2d
    activation: callable

    def __init__(self, in_channels, out_channels, modes1, modes2, activation, *, key):
        """
        This method initializes the FNOBlock2d object.

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            modes1 (_type_): _description_
            modes2 (_type_): _description_
            activation (_type_): _description_
            key (_type_):

        Returns:
            None
        """
        keys = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels, out_channels, modes1, modes2, key=keys[0]
        )
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=1, key=keys[1])
        self.activation = activation

    def __call__(self, x):
        """Call method for FNOBlock2d

        Args:
            x: jax.Array

        Returns:
            jax.Array: Output of the FNOBlock2d object
        """
        return self.activation(self.spectral_conv(x) + self.conv(x))
