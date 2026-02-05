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
Module: fno_block_1d.py

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
    29/Jan/2025: Minor changes - Diya Nag Chaudhury

References:
    None
"""

import jax
import equinox as eqx
import jax.numpy as jnp

from typing import Callable

from scirex.core.sciml.fno.layers.spectral_conv_1d import SpectralConv1d


class FNOBlock1d(eqx.Module):
    """
    A single block of the FNO model.

    This block consists of a spectral convolution followed by a bypass convolution
    and an activation function.

    Attributes:
    spectral_conv: SpectralConv1d
    bypass_conv: eqx.nn.Conv1d
    activation: Callable

    Methods:
    __init__: Initializes the FNOBlock1d object
    __call__: Calls the FNOBlock1d object
    """

    spectral_conv: SpectralConv1d
    bypass_conv: eqx.nn.Conv1d
    activation: Callable

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        activation,
        *,
        key,
    ):
        """
        This method initializes the FNOBlock1d object.

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            modes (_type_): _description_
            activation (_type_): _description_
            key (_type_): _description_

        Returns:
            None
        """
        spectral_conv_key, bypass_conv_key = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(
            in_channels,
            out_channels,
            modes,
            key=spectral_conv_key,
        )
        self.bypass_conv = eqx.nn.Conv1d(
            in_channels,
            out_channels,
            1,  # Kernel size is one
            key=bypass_conv_key,
        )
        self.activation = activation

    def __call__(
        self,
        x,
    ):
        """
        This method calls the FNOBlock1d object.

        Args:
            x (jax.Array): Input array

        Returns:
            jax.Array: Output array
        """
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))
