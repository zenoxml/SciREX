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
    Module: loss.py

    This module implements loss functions for Neural Networks
    Currenty, it uses optax library for loss functions
    (documentation: https://optax.readthedocs.io/en/latest/api/losses.html)

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/01/2025: Initial version

"""
import jax
import jax.numpy as jnp
import optax


def mse_loss(output: jax.Array, y: jax.Array) -> float:
    """
    Compute mean squared error loss

    Args:
        output: output of the model
        y: target values
    """
    return jnp.mean(jnp.square(output - y))


def cross_entropy_loss(output: jax.Array, y: jax.Array) -> float:
    """
    Compute the cross-entropy loss

    Args:
        output: output of the model
        y: Batched target labels
    """

    n_classes = output.shape[-1]
    loss = optax.softmax_cross_entropy(output, jax.nn.one_hot(y, n_classes)).mean()
    return loss


convex_kl_divergence = optax.losses.convex_kl_divergence
cosine_distance = optax.losses.cosine_distance
cosine_similarity = optax.losses.cosine_similarity
ctc_loss = optax.losses.ctc_loss
ctc_loss_with_forward_probs = optax.losses.ctc_loss_with_forward_probs
hinge_loss = optax.losses.hinge_loss
huber_loss = optax.losses.huber_loss
kl_divergence = optax.losses.kl_divergence
kl_divergence_with_log_targets = optax.losses.kl_divergence_with_log_targets
l2_loss = optax.losses.l2_loss
log_cosh = optax.losses.log_cosh
make_fenchel_young_loss = optax.losses.make_fenchel_young_loss
multiclass_hinge_loss = optax.losses.multiclass_hinge_loss
multiclass_perceptron_loss = optax.losses.multiclass_perceptron_loss
multiclass_sparsemax_loss = optax.losses.multiclass_sparsemax_loss
ntxent = optax.losses.ntxent
perceptron_loss = optax.losses.perceptron_loss
poly_loss_cross_entropy = optax.losses.poly_loss_cross_entropy
ranking_softmax_loss = optax.losses.ranking_softmax_loss
safe_softmax_cross_entropy = optax.losses.safe_softmax_cross_entropy
sigmoid_binary_cross_entropy = optax.losses.sigmoid_binary_cross_entropy
sigmoid_focal_loss = optax.losses.sigmoid_focal_loss
smooth_labels = optax.losses.smooth_labels
softmax_cross_entropy = optax.losses.softmax_cross_entropy
softmax_cross_entropy_with_integer_labels = (
    optax.losses.softmax_cross_entropy_with_integer_labels
)
sparsemax_loss = optax.losses.sparsemax_loss
squared_error = optax.losses.squared_error
