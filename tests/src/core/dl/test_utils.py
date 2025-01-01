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
    Module: test_utils.py

    This module contains unit tests for the `utils.py` file in the Deep Learning framework.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/01/2025: Initial version

"""
import pytest
import jax.numpy as jnp
from scirex.core.dl.utils import (
    accuracy,
    mse_loss,
    cross_entropy_loss
)

@pytest.mark.parametrize("pred_y, y, expected", [
    (jnp.array([6, 4]), jnp.array([6, 4]), 1.0),
    (jnp.array([0, 1]), jnp.array([1, 0]), 0.0),
    (jnp.array([0, 1]), jnp.array([0, 0]), 0.5),
])
def test_accuracy(pred_y, y, expected):
    assert accuracy(pred_y, y) == expected


@pytest.mark.parametrize("output, y, expected", [
    (jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0]), 1.0),
    (jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]), 0.0),
])
def test_mse_loss(output, y, expected):
    assert mse_loss(output, y) == expected


@pytest.mark.parametrize("output, y, expected", [
    (
        jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        jnp.array([0, 1]),
        pytest.approx(0.693, rel=1e-2)
    ),
    (
        jnp.array([[0.33, 0.33, 0.34], [0.3, 0.6, 0.1]]),
        jnp.array([0, 1]),
        pytest.approx(0.977, rel=1e-2)
    )
])
def test_cross_entropy_loss(output, y, expected):
    assert cross_entropy_loss(output, y) == expected
