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
    Module: test_fcnn.py

    This module contains unit tests for the Fully Connected Neural Network implementation with the SciREX framework.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 03/01/2024: Initial version

"""
import pytest
import jax
import jax.numpy as jnp
import optax
from scirex.core.dl.jax_backend.equinox import layers, activations, losses
from scirex.core.dl.jax_backend.equinox.networks import FCNN
from scirex.core.dl.jax_backend.equinox.base import Model



key = jax.random.PRNGKey(0)

layersLinear = [layers.Linear(20, 10, key=key), activations.relu, layers.Linear(10, 1, key=key)]
layersConv = [
    layers.Conv2d(1, 2, 2, key=key),
    layers.MaxPool2d(2),
    jnp.ravel,
    layers.Linear(12, 1, key=key),
]

x = jax.random.normal(key, (100, 20))
y = x @ jax.random.normal(key, (20, 1)) + jax.random.normal(key, (100, 1))
data1D = (x, y)
data2D = (x.reshape(100, 1, 5, 4), y)
model1D = Model(FCNN(layersLinear), optax.sgd(1e-3), losses.mse_loss, [losses.mse_loss])
model2D = Model(FCNN(layersConv), optax.sgd(1e-3), losses.mse_loss, [losses.mse_loss])

# Parameterized variables in global scope
pytestmark = pytest.mark.parametrize(
    "model, data",
    [
        (model1D, data1D),
        (model2D, data2D),
    ],
)


@pytest.mark.dependency(name="predict")
def test_fcnn_predict(model, data):
    x, y = data
    y_pred = model.predict(x)
    assert y_pred.shape == y.shape


@pytest.mark.dependency(name="evaluate", depends=["predict"])
def test_fcnn_evaluate(model, data):
    x, y = data
    loss, metrics = model.evaluate(x, y)
    assert loss.shape == ()
    assert len(metrics) == 1


@pytest.mark.dependency(depends=["evaluate"])
def test_fcnn_fit(model, data):
    x, y = data
    loss_initial, _ = model.evaluate(x, y)
    history = model.fit(x, y, num_epochs=10, batch_size=10)
    loss_final, _ = model.evaluate(x, y)
    assert len(history) == 10
    assert loss_final < loss_initial
