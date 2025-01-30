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
import optax
import equinox as eqx
from scirex.core.dl import Model, Network
from scirex.core.dl.losses import mse_loss


key = jax.random.PRNGKey(0)


class Linear(Network):
    weight: jax.Array
    bias: jax.Array

    def __init__(self):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (20, 1))
        self.bias = jax.random.normal(bkey, (1,))

    def __call__(self, x):
        return self.weight.T @ x + self.bias


class Non_Linear(Linear):
    def __call__(self, x):
        return self.weight.T @ jnp.ravel(x) + self.bias


x = jax.random.normal(key, (100, 20))
y = x @ jax.random.normal(key, (20, 1)) + jax.random.normal(key, (100, 1))
data1D = (x, y)
data2D = (x.reshape(100, 5, 4), y)
data3D = (x.reshape(100, 5, 2, 2), y)
model1D = Model(Linear(), optax.sgd(1e-3), mse_loss, [mse_loss])
model2D = Model(Non_Linear(), optax.sgd(1e-3), mse_loss, [mse_loss])

# Parameterized variables in global scope
pytestmark = pytest.mark.parametrize(
    "model, data",
    [(model1D, data1D), (model2D, data2D), (model2D, data3D)],
)


def test_model_predict_single(model, data):
    x, y = data
    y_pred = model.predict(x[:1])
    assert y_pred.shape == (1, *y[0].shape)


@pytest.mark.dependency(name="predict")
def test_model_predict_multiple(model, data):
    x, y = data
    y_pred = model.predict(x)
    assert y_pred.shape == y.shape


@pytest.mark.dependency(name="create_batches")
def test_create_batches(model, data):
    x, y = data
    (x_train, y_train), (x_val, y_val) = model._create_batches(x, y, 9)
    assert x_train.shape == (10, 9, *x.shape[1:])
    assert y_train.shape == (10, 9, *y.shape[1:])
    assert x_val.shape == (10, *x.shape[1:])
    assert y_val.shape == (10, *y.shape[1:])


@pytest.mark.dependency(name="evaluate", depends=["predict"])
def test_model_evaluate(model, data):
    x, y = data
    loss, metrics = model.evaluate(x, y)
    assert loss.shape == ()
    assert len(metrics) == 1


@pytest.mark.dependency(name="loss_fn", depends=["predict"])
def test_model_loss_fn(model, data):
    x, y = data
    loss = model._loss_fn(model.net, x, y)
    assert loss.shape == ()
    assert model.loss_fn(model.net(x[0]), y[0]) == model._loss_fn(
        model.net, x[:1], y[:1]
    )


@pytest.mark.dependency(name="update_step", depends=["loss_fn"])
def test_update_step(model, data):
    x, y = data
    opt_state = model.optimizer.init(eqx.filter(model.net, eqx.is_array))
    loss, _, _ = model._update_step(x, y, model.net, opt_state)
    assert loss.shape == ()


@pytest.mark.dependency(name="epoch_step", depends=["update_step"])
def test_epoch_step(model, data):
    x, y = data
    (x_train, y_train), (x_val, y_val) = model._create_batches(x, y, 9)
    opt_state = model.optimizer.init(eqx.filter(model.net, eqx.is_array))
    loss, time, _, _ = model._epoch_step(x_train, y_train, model.net, opt_state)
    assert loss.shape == ()
    assert time > 0


@pytest.mark.dependency(depends=["create_batches", "epoch_step"])
def test_model_fit(model, data):
    x, y = data
    loss_initial, _ = model.evaluate(x, y)
    history = model.fit(x, y, num_epochs=10, batch_size=10)
    loss_final, _ = model.evaluate(x, y)
    assert len(history) == 10
    assert loss_final < loss_initial


def test_model_save_load_net(model, data):
    x, y = data
    y_pred = model.predict(x)
    model.save_net("test_model.net")
    model.load_net("test_model.net")
    y_pred_loaded = model.predict(x)
    assert jnp.allclose(y_pred, y_pred_loaded)


def test_model_update_net(model, data):
    class TestNet(Network):
        weight: list

        def __init__(self, weight):
            self.weight = weight

        def __call__(self, x):
            return self.weight @ x

    weight, x = jnp.asarray([2.0]), jnp.asarray([[1.0]])
    model = Model(TestNet(weight), optax.sgd(1e-3), mse_loss)
    pred = model.predict(x)
    model.update_net(weight=2 * weight)
    assert model.predict(x) == 2 * pred
