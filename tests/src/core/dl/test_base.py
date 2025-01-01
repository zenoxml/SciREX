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
    Module: test_base.py

    This module contains unit tests for the `base.py` file in the Deep Learning framework.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/01/2025: Initial version

"""
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from scirex.core.dl import Model, Network
from scirex.core.dl.utils import mse_loss


@pytest.fixture
def linear_model():
    key = jax.random.PRNGKey(0)

    class Linear(Network):
        weight: jax.Array
        bias: jax.Array

        def __init__(self):
            wkey, bkey = jax.random.split(key)
            self.weight = jax.random.normal(wkey, (10, 1))
            self.bias = jax.random.normal(bkey, (1,))

        def __call__(self, x):
            return x @ self.weight + self.bias

    return Model(Linear(), optax.sgd(1e-3), mse_loss, [mse_loss])


@pytest.fixture
def data():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 10))
    y = x @ jax.random.normal(key, (10, 1)) + jax.random.normal(key, (100, 1))
    return x, y


def test_model_predict_single(linear_model, data):
    x, y = data
    y_pred = linear_model.predict(x[0])
    assert y_pred.shape == y[0].shape


def test_model_predict_multiple(linear_model, data):
    x, y = data
    y_pred = linear_model.predict(x)
    assert y_pred.shape == y.shape


def test_model_evaluate(linear_model, data):
    x, y = data
    loss, metrics = linear_model.evaluate(x, y)
    assert loss.shape == ()
    assert len(metrics) == 1


def test_create_batches(linear_model, data):
    x, y = data
    (x_train, y_train), (x_val, y_val) = linear_model._create_batches(x, y, 9)
    assert x_train.shape == (10, 9, 10)
    assert y_train.shape == (10, 9, 1)
    assert x_val.shape == (10, 10)
    assert y_val.shape == (10, 1)


def test_model_fit(linear_model, data):
    x, y = data
    loss_initial, _ = linear_model.evaluate(x, y)
    history = linear_model.fit(x, y, num_epochs=10, batch_size=10)
    loss_final, _ = linear_model.evaluate(x, y)
    assert len(history) == 10
    assert loss_final < loss_initial
