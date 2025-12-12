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
    File: activations.py

    Description: This module contains the implementation of various activation functions
                 used in deep learning, using the tensorflow backend.

    Authors:
        - Divij Ghose (divijghose@{iisc.ac.in}), R.Shrinivass (github:Shrini14)

    Version Info:
        - 31/01/2025: Initial version

"""

import tensorflow as tf


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def tanh(x):
    return tf.nn.tanh(x)


def relu6(x):
    return tf.nn.relu6(x)


def softplus(x):
    return tf.nn.softplus(x)


def sparse_plus(x):
    return tf.nn.relu(x) + 1e-6 * x


def sparse_sigmoid(x):
    return tf.nn.sigmoid(x) + 1e-6 * x


def soft_sign(x):
    return tf.nn.softsign(x)


def silu(x):
    return tf.nn.silu(x)


def swish(x):
    return tf.nn.swish(x)


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x):
    return tf.nn.leaky_relu(x)


def hard_sigmoid(x):
    return tf.keras.activations.hard_sigmoid(x)


def hard_swish(x):
    return tf.nn.hard_swish(x)


def hard_tanh(x):
    return tf.clip_by_value(x, -1.0, 1.0)


def elu(x):
    return tf.nn.elu(x)


def celu(x):
    return tf.nn.celu(x)


def selu(x):
    return tf.nn.selu(x)


def gelu(x):
    return tf.nn.gelu(x)


def glu(x):
    return x * tf.nn.sigmoid(x)


def squareplus(x):
    return (x + tf.sqrt(tf.square(x) + 4.0)) / 2.0


def mish(x):
    return x * tf.math.tanh(tf.nn.softplus(x))


if __name__ == "__main__":
    pass
