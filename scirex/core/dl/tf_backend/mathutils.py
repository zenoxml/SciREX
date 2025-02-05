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

    Description: This module contains the implementation of various mathematical utility functions

    Authors:
        - Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 31/01/2025: Initial version

"""

import tensorflow as tf


def add(x, y):
    return tf.add(x, y)


def subtract(x, y):
    return tf.subtract(x, y)


def multiply(x, y):
    return tf.multiply(x, y)


def divide(x, y):
    return tf.divide(x, y)


def square(x):
    return tf.square(x)


def sqrt(x):
    return tf.sqrt(x)


def exp(x):
    return tf.exp(x)


def log(x):
    return tf.log(x)


def sin(x):
    return tf.sin(x)


def cos(x):
    return tf.cos(x)


def tan(x):
    return tf.tan(x)


def reduce_sum(x, axis=None):
    return tf.reduce_sum(x, axis=axis)


def reduce_mean(x, axis=None):
    return tf.reduce_mean(x, axis=axis)


def reduce_max(x, axis=None):
    return tf.reduce_max(x, axis=axis)


def reduce_min(x, axis=None):
    return tf.reduce_min(x, axis=axis)


def dot(x, y):
    return tf.tensordot(x, y, axes=1)


if __name__ == "__main__":
    pass
