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
Module: spectral_conv_2d.py

This module provides the implementation of a 2D spectral convolution layer.


Classes:
    SpectralConv2d: 2D spectral convolution layer

Dependencies:
    - jax: For array processing
    - equinox: For neural network layers

Key Features:
    - Complex multiplication
    - Fourier domain convolution
    - Real and imaginary weights

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


class SpectralConv2d(eqx.Module):
    """2D Spectral Convolution layer for FNO

    This layer performs a 2D convolution in the Fourier domain.

    Attributes:
        real_weights: jax.Array
        imag_weights: jax.Array
        in_channels: int
        out_channels: int
        modes1: int  # Number of Fourier modes to keep in x direction
        modes2: int  # Number of Fourier modes to keep in y direction

    Methods:
        __init__: Initializes the SpectralConv2d object
        __call__: Calls the SpectralConv2d object

    """

    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes1: int  # Number of Fourier modes to keep in x direction
    modes2: int  # Number of Fourier modes to keep in y direction

    def __init__(self, in_channels, out_channels, modes1, modes2, *, key):
        """
        This method initializes the SpectralConv2d object.

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            modes1 (_type_): _description_
            modes2 (_type_): _description_
            key (_type_): _description_

        Returns:
            None
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        real_key, imag_key = jax.random.split(key)

        self.real_weights = jax.random.uniform(
            real_key,
            (in_channels, out_channels, modes1, modes2),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights = jax.random.uniform(
            imag_key,
            (in_channels, out_channels, modes1, modes2),
            minval=-scale,
            maxval=+scale,
        )

    def __call__(self, x):
        """
        Forward pass of the 2D spectral convolution layer.

        Args:
            x (jax.Array): Input tensor

        Returns:
            x (jax.Array): Output tensor
        """
        # x: (channels, height, width)
        x_ft = jnp.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = jnp.zeros(
            (self.out_channels, x.shape[1], x_ft.shape[-1]), dtype=jnp.complex64
        )

        weights = self.real_weights + 1j * self.imag_weights
        out_ft = out_ft.at[:, : self.modes1, : self.modes2].set(
            jnp.einsum("ixy,ioxy->oxy", x_ft[:, : self.modes1, : self.modes2], weights)
        )

        # Return to physical space
        x = jnp.fft.irfft2(out_ft, s=(x.shape[1], x.shape[2]))
        return x
