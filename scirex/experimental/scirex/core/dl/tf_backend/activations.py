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
        - Divij Ghose (divijghose@{iisc.ac.in})

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


if __name__ == "__main__":
    pass
