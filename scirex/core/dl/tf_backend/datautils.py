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
    File: datautils.py

    Description: This module contains the implementation of various data 
                 handling utility functions in tensorflow.

    Authors:
        - Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 31/01/2025: Initial version

"""
import tensorflow as tf
import numpy as np


def reshape(x, shape):
    return tf.reshape(x, shape)


def transpose(x, axes=None):
    return tf.transpose(x, perm=axes)


def is_tensor(x):
    return tf.is_tensor(x)


def convert_to_tensor(x, dtype=None):
    return tf.convert_to_tensor(x, dtype=dtype)


def cast(x, dtype):
    return tf.cast(x, dtype)


def concat(tensors, axis):
    return tf.concat(tensors, axis=axis)


def vstack(tensors):
    return tf.vstack(tensors)


def hstack(tensors):
    return tf.hstack(tensors)


def stack(tensors, axis):
    return tf.stack(tensors, axis=axis)


def convert_to_numpy(x):
    # check if x is already a numpy array
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.numpy()


def get_devices():
    devices = tf.config.list_physical_devices()
    # pretty print
    print("Available devices:\n")
    print("-" * 50)
    for device in devices:
        device_name = device.name.replace("/physical_device:", "")
        print(f"Device name: {device_name}")
        if "GPU" in device.device_type:
            details = tf.config.experimental.get_device_details(device)
            print(f"Device type: {details['device_name']}")
        print("-" * 50)
    return devices


def get_device_count():
    return len(tf.config.list_physical_devices())


def get_batches(x_train, y_train, batch_size, buffer_size=1024):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    batched_dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    return batched_dataset
