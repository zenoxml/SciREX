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
    Module: convolution.py

    This module implements convolutional layers for Neural Networks

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 06/01/2025: Initial version

"""
import jax
import equinox as eqx


class Conv(eqx.nn.Conv):
    """
    Performs a convolution operation
    """


class Conv1d(eqx.nn.Conv1d):
    """
    Performs a 1D convolution operation
    """


class Conv2d(eqx.nn.Conv2d):
    """
    Performs a 2D convolution operation
    """


class Conv3d(eqx.nn.Conv3d):
    """
    Performs a 3D convolution operation
    """


class ConvTranspose(eqx.nn.ConvTranspose):
    """
    Performs a transposed convolution operation
    """


class ConvTranspose1d(eqx.nn.ConvTranspose1d):
    """
    Performs a 1D transposed convolution operation
    """


class ConvTranspose2d(eqx.nn.ConvTranspose2d):
    """
    Performs a 2D transposed convolution operation
    """


class ConvTranspose3d(eqx.nn.ConvTranspose3d):
    """
    Performs a 3D transposed convolution operation
    """
