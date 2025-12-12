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
    File: mathutils.py

    Description: JAX implementation of various mathematical utility functions

    Authors:
        - R. Shrinivass (github: Shrini14)

    Version Info:
        - 12/12/2025: Initial JAX backend version
"""

import jax.numpy as jnp


def add(x, y):
    return jnp.add(x, y)


def subtract(x, y):
    return jnp.subtract(x, y)


def multiply(x, y):
    return jnp.multiply(x, y)


def divide(x, y):
    return jnp.divide(x, y)


def square(x):
    return jnp.square(x)


def sqrt(x):
    return jnp.sqrt(x)


def exp(x):
    return jnp.exp(x)


def log(x):
    return jnp.log(x)


def sin(x):
    return jnp.sin(x)


def cos(x):
    return jnp.cos(x)


def tan(x):
    return jnp.tan(x)


def reduce_sum(x, axis=None):
    return jnp.sum(x, axis=axis)


def reduce_mean(x, axis=None):
    return jnp.mean(x, axis=axis)


def reduce_max(x, axis=None):
    return jnp.max(x, axis=axis)


def reduce_min(x, axis=None):
    return jnp.min(x, axis=axis)


def dot(x, y):
    return jnp.tensordot(x, y,axes=1)


if __name__ == "__main__":
    pass
