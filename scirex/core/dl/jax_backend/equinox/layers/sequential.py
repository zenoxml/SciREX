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
    Module: sequential.py

    This module implements sequential layers for Neural Networks

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 06/01/2025: Initial version

"""
import jax
import equinox as eqx


class Sequential(eqx.nn.Sequential):
    """
    Implements a Sequential layer, which is a stack of layers
    """


class Lambda(eqx.nn.Lambda):
    """
    Implements a Lambda layer (which wraps a callable for use with Sequential)
    """


class StatefulLayer(eqx.nn.StatefulLayer):
    """
    Implements a StatefulLayer
    """
