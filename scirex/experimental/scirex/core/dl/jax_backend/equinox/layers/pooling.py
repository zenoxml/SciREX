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
    Module: pooling.py

    This module implements pooling layers for Neural Networks

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 06/01/2025: Initial version

"""
import jax
import equinox as eqx


class Pool(eqx.nn.Pool):
    """
    Performs a pooling operation
    """


class AvgPool1d(eqx.nn.AvgPool1d):
    """
    Performs a 1D average pooling operation
    """


class AvgPool2d(eqx.nn.AvgPool2d):
    """
    Performs a 2D average pooling operation
    """


class AvgPool3d(eqx.nn.AvgPool3d):
    """
    Performs a 3D average pooling operation
    """


class MaxPool1d(eqx.nn.MaxPool1d):
    """
    Performs a 1D max pooling operation
    """


class MaxPool2d(eqx.nn.MaxPool2d):
    """
    Performs a 2D max pooling operation
    """


class MaxPool3d(eqx.nn.MaxPool3d):
    """
    Performs a 3D max pooling operation
    """


class AdaptivePool(eqx.nn.AdaptivePool):
    """
    Performs an adaptive pooling operation
    """


class AdaptiveAvgPool1d(eqx.nn.AdaptiveAvgPool1d):
    """
    Performs a 1D adaptive average pooling operation
    """


class AdaptiveAvgPool2d(eqx.nn.AdaptiveAvgPool2d):
    """
    Performs a 2D adaptive average pooling operation
    """


class AdaptiveAvgPool3d(eqx.nn.AdaptiveAvgPool3d):
    """
    Performs a 3D adaptive average pooling operation
    """


class AdaptiveMaxPool1d(eqx.nn.AdaptiveMaxPool1d):
    """
    Performs a 1D adaptive max pooling operation
    """


class AdaptiveMaxPool2d(eqx.nn.AdaptiveMaxPool2d):
    """
    Performs a 2D adaptive max pooling operation
    """


class AdaptiveMaxPool3d(eqx.nn.AdaptiveMaxPool3d):
    """
    Performs a 3D adaptive max pooling operation
    """
