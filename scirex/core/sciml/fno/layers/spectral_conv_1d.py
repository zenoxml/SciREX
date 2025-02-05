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
Module: spectral_conv_1d.py

This module provides the implementation of a 1D spectral convolution layer.

Classes:
    SpectralConv1d: 1D spectral convolution layer

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
    29/Jan/2025: Minor changes - Diya Nag Chaudhury

References:
    None
"""

import jax
import equinox as eqx
import jax.numpy as jnp


class SpectralConv1d(eqx.Module):
    """
    A 1D spectral convolution layer.

    This layer performs a 1D convolution in the Fourier domain.

    Attributes:
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes: int

    Methods:
    __init__: Initializes the SpectralConv1d object
    complex_mult1d: Performs complex multiplication in 1D
    __call__: Calls the SpectralConv1d object
    """

    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes: int

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        *,
        key,
    ):
        """
        Constructor for the SpectralConv1d class.

        Parameters:
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        modes: int
            Number of modes
        key: jax.random.PRNGKey
            Random key for initialization

        Returns:
        None

        Usage:
        spectral_conv = SpectralConv1d(
            in_channels=1,
            out_channels=1,
            modes=64,
            key=key,
        )
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)

        real_key, imag_key = jax.random.split(key)
        self.real_weights = jax.random.uniform(
            real_key,
            (in_channels, out_channels, modes),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights = jax.random.uniform(
            imag_key,
            (in_channels, out_channels, modes),
            minval=-scale,
            maxval=+scale,
        )

    def complex_mult1d(
        self,
        x_hat,
        w,
    ):
        """
        Returns the complex multiplication of x_hat and w.

        Parameters:
        x_hat: jax.Array
            Input array in the Fourier domain
        w: jax.Array
            Weights in the Fourier domain

        Returns:
        jax.Array
            Complex multiplication of x_hat and w

        Usage:
        y_hat = spectral_conv.complex_mult1d(x_hat, w)

        """
        return jnp.einsum("iM,ioM->oM", x_hat, w)

    def __call__(
        self,
        x,
    ):
        """
        Forward pass of the SpectralConv1d layer.

        Parameters:
        x: jax.Array
            Input array

        Returns:
        jax.Array
            Output array

        Usage:
        y = spectral_conv(x)
        """
        channels, spatial_points = x.shape

        # shape of x_hat is (in_channels, spatial_points//2+1)
        x_hat = jnp.fft.rfft(x)
        # shape of x_hat_under_modes is (in_channels, self.modes)
        x_hat_under_modes = x_hat[:, : self.modes]
        weights = self.real_weights + 1j * self.imag_weights
        # shape of out_hat_under_modes is (out_channels, self.modes)
        out_hat_under_modes = self.complex_mult1d(x_hat_under_modes, weights)

        # shape of out_hat is (out_channels, spatial_points//2+1)
        out_hat = jnp.zeros((self.out_channels, x_hat.shape[-1]), dtype=x_hat.dtype)
        out_hat = out_hat.at[:, : self.modes].set(out_hat_under_modes)

        out = jnp.fft.irfft(out_hat, n=spatial_points)

        return out
