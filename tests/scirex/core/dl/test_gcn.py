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
    Module: test_gcn.py

    This module contains unit tests for the Graph Convolution Network in the SciREX framework.

    Authors:
        - Rajarshi Dasgupta (rajarshid@iisc.ac.in)

    Version Info:
        - 10/01/2025: Initial version

"""
import pytest
import jax
import jax.numpy as jnp
import optax
from scirex.core.dl.gcn import GCN, GCNModel

key = jax.random.PRNGKey(0)
key_x, key_A, key_model = jax.random.split(key, num=3)

num_nodes = 5
input_feature_vector_size = 10
output_vector_size = 2

x = jax.random.normal(key_x, (num_nodes, input_feature_vector_size))
x = x / jnp.linalg.norm(x, axis=1, keepdims=True)  # Normalising

A = jax.random.bernoulli(key_A, p=0.4, shape=(num_nodes, num_nodes))
A = (A + jnp.transpose(A)) * 1
deg = A.sum(axis=0)


def loss_fn(output, target):
    differences = output.transpose() @ output - target
    return jnp.square(differences).sum()


def test_GCN_call():
    gcn = GCN(
        [input_feature_vector_size, 10, 10, output_vector_size],
        [jnp.tanh] * 2 + [jax.nn.sigmoid],
        key_model,
    )

    output = gcn(x, A, deg)
    assert output.shape == (num_nodes, output_vector_size)


def test_GCNModel_fit():
    gcn = GCN(
        [input_feature_vector_size, 10, 10, output_vector_size],
        [jnp.tanh] * 2 + [jax.nn.sigmoid],
        key_model,
    )

    target = jnp.eye(output_vector_size)

    model = GCNModel(gcn, loss_fn)

    output = gcn(x, A, deg)
    loss_initial = loss_fn(output, target)

    gcn = model.fit(x, A, deg, target, learning_rate=5e-2, num_iters=10)

    output = gcn(x, A, deg)
    loss_final = loss_fn(output, target)

    assert loss_final < loss_initial
