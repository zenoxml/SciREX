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
"""TensorFlow interface for TensorFlow Probability optimizers."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class LBFGSHelper:
    """
    Helper class for L-BFGS optimizer.
    """

    def __init__(self, trainable_variables, build_loss):
        """
        Constructor.

        Args:
            trainable_variables: list of tf.Variable, trainable variables.
            build_loss: callable, loss function.
        """
        self.trainable_variables = trainable_variables
        self.build_loss = build_loss

        # shapes of trainable variables
        self.trainable_variables_shape = tf.shape_n(trainable_variables)
        self.num_tensors_trainable_variables = len(self.trainable_variables_shape)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.trainable_variables_shape):
            n = np.prod(shape)
            self.indices.append(
                tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
            )
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

    @tf.function
    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        Args:
           weights_1d: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights
        self.set_flat_weights(weights_1d)
        with tf.GradientTape() as tape:
            # Calculate the loss
            loss = self.build_loss()
        # Calculate convert to 1D tf.Tensor
        grads = tape.gradient(loss, self.trainable_variables)
        grads = tf.dynamic_stitch(self.indices, grads)
        return loss, grads

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.

        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        weights = tf.dynamic_partition(
            weights_1d, self.partitions, self.num_tensors_trainable_variables
        )
        for i, (shape, param) in enumerate(
            zip(self.trainable_variables_shape, weights)
        ):
            self.trainable_variables[i].assign(tf.reshape(param, shape))

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.

        Args:
            weights: A list of tf.Tensor representing the weights.

        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        return tf.dynamic_stitch(self.indices, weights)


def lbfgs_minimize(trainable_variables, build_loss, previous_optimizer_results=None):
    """TensorFlow interface for tfp.optimizer.lbfgs_minimize.

    Args:
        trainable_variables: Trainable variables, also used as the initial position.
        build_loss: A function to build the loss function expression.
        previous_optimizer_results
    """
    func = LBFGSHelper(trainable_variables, build_loss)
    initial_position = None
    if previous_optimizer_results is None:
        initial_position = func.to_flat_weights(trainable_variables)
    results = tfp.optimizer.lbfgs_minimize(
        func,
        initial_position=initial_position,
        previous_optimizer_results=previous_optimizer_results,
        num_correction_pairs=10,
        tolerance=1e-8,
        x_tolerance=0,
        f_relative_tolerance=0,
        max_iterations=10000,
        parallel_iterations=1,
        max_line_search_iterations=50,
    )
    # The final optimized parameters are in results.position.
    # Set them back to the variables.
    func.set_flat_weights(results.position)
    return results
