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


import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from typing import Tuple

from scirex.core.sciml.fno.layers.spectral_conv_1d import SpectralConv1d
from scirex.core.sciml.fno.layers.spectral_conv_2d import SpectralConv2d
from scirex.core.sciml.fno.layers.fno_block_1d import FNOBlock1d
from scirex.core.sciml.fno.layers.fno_block_2d import FNOBlock2d
from scirex.core.sciml.fno.models.fno_1d import FNO1d
from scirex.core.sciml.fno.models.fno_2d import FNO2d


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def dummy_data(rng_key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate dummy data for testing"""
    batch_size = 4
    spatial_size = 64
    channels = 2

    # Input shape: (batch_size, channels, spatial_size)
    inputs = jax.random.normal(rng_key, (batch_size, channels, spatial_size))

    # Output shape: (batch_size, 1, spatial_size)
    outputs = jax.random.normal(rng_key, (batch_size, 1, spatial_size))

    return inputs, outputs


class TestSpectralConv1d:
    """Test suite for the SpectralConv1d layer"""

    def test_initialization(self, rng_key):
        """Test initialization with different parameters"""
        test_configs = [
            (2, 4, 16),  # (in_channels, out_channels, modes)
            (1, 1, 8),
            (4, 2, 32),
        ]

        for in_channels, out_channels, modes in test_configs:
            layer = SpectralConv1d(in_channels, out_channels, modes, key=rng_key)

            # Check attributes
            assert layer.in_channels == in_channels
            assert layer.out_channels == out_channels
            assert layer.modes == modes

            # Check weight shapes
            assert layer.real_weights.shape == (in_channels, out_channels, modes)
            assert layer.imag_weights.shape == (in_channels, out_channels, modes)

    def test_complex_mult1d(self, rng_key):
        """Test complex multiplication operation"""
        layer = SpectralConv1d(2, 2, 4, key=rng_key)

        # Test with simple input
        x_hat = jnp.ones((2, 4), dtype=jnp.complex64)
        weights = jnp.ones((2, 2, 4), dtype=jnp.complex64)

        result = layer.complex_mult1d(x_hat, weights)

        # Check shape and dtype
        assert result.shape == (2, 4)
        assert jnp.issubdtype(result.dtype, jnp.complex64)

        # Test with zeros
        x_hat = jnp.zeros((2, 4), dtype=jnp.complex64)
        result = layer.complex_mult1d(x_hat, weights)
        assert jnp.allclose(result, jnp.zeros_like(result))

    def test_forward_pass(self, rng_key):
        """Test forward pass with different input sizes"""
        spatial_sizes = [32, 64, 128]
        in_channels = 2
        out_channels = 4
        modes = 16

        layer = SpectralConv1d(in_channels, out_channels, modes, key=rng_key)

        for size in spatial_sizes:
            x = jax.random.normal(rng_key, (in_channels, size))
            output = layer(x)

            # Check output shape
            assert output.shape == (out_channels, size)
            # Check output is finite
            assert jnp.all(jnp.isfinite(output))

    def test_fourier_properties(self, rng_key):
        """Test that the layer preserves Fourier transform properties"""
        modes = 4
        layer = SpectralConv1d(1, 1, modes, key=rng_key)

        # Create input with known Fourier properties
        spatial_size = 8  # Must be >= 2*modes for the test
        x = jnp.cos(2 * jnp.pi * jnp.arange(spatial_size) / spatial_size)
        x = x[None, :]  # Add channel dimension

        # Get output
        output = layer(x)

        # Verify output shape and realness
        assert output.shape == (1, spatial_size)
        # The output should be real since input is real
        output_ft = jnp.fft.rfft(output[0])
        assert jnp.allclose(output_ft[modes:].imag, 0.0, atol=1e-6)
        assert jnp.allclose(output_ft[modes:].real, 0.0, atol=1e-6)


class TestFNOBlock1d:
    """Test suite for the FNOBlock1d"""

    def test_initialization(self, rng_key):
        """Test initialization of FNO block"""
        in_channels = 2
        out_channels = 4
        modes = 16

        block = FNOBlock1d(in_channels, out_channels, modes, jax.nn.gelu, key=rng_key)

        # Check components
        assert isinstance(block.spectral_conv, SpectralConv1d)
        assert isinstance(block.bypass_conv, eqx.nn.Conv1d)
        assert block.activation == jax.nn.gelu

    def test_different_activations(self, rng_key):
        """Test block with different activation functions"""
        activations = [
            jax.nn.relu,
            jax.nn.gelu,
            jax.nn.tanh,
            lambda x: x,  # identity
        ]

        x = jax.random.normal(rng_key, (2, 64))

        for activation in activations:
            block = FNOBlock1d(2, 2, 16, activation, key=rng_key)
            output = block(x)

            assert output.shape == x.shape
            assert jnp.all(jnp.isfinite(output))

    def test_bypass_connection(self, rng_key):
        """Test that the bypass connection is working"""
        block = FNOBlock1d(2, 2, 16, lambda x: x, key=rng_key)

        # Create input where spectral component should be zero
        x = jnp.ones((2, 64))
        output_with_bypass = block(x)

        # Modify the block to zero out spectral conv
        modified_block = eqx.tree_at(
            lambda b: b.spectral_conv.real_weights,
            block,
            jnp.zeros_like(block.spectral_conv.real_weights),
        )
        modified_block = eqx.tree_at(
            lambda b: b.spectral_conv.imag_weights,
            modified_block,
            jnp.zeros_like(block.spectral_conv.imag_weights),
        )

        output_only_bypass = modified_block(x)

        # The outputs should be different
        assert not jnp.allclose(output_with_bypass, output_only_bypass)


class TestFNO1d:
    """Test suite for the complete FNO1d model"""

    def test_initialization(self, rng_key):
        """Test model initialization with different configurations"""
        configs = [
            (2, 1, 16, 32, 4),  # (in_channels, out_channels, modes, width, n_blocks)
            (1, 1, 8, 16, 2),
            (4, 2, 32, 64, 6),
        ]

        for in_ch, out_ch, modes, width, n_blocks in configs:
            model = FNO1d(
                in_channels=in_ch,
                out_channels=out_ch,
                modes=modes,
                width=width,
                activation=jax.nn.gelu,
                n_blocks=n_blocks,
                key=rng_key,
            )

            assert isinstance(model.lifting, eqx.nn.Conv1d)
            assert len(model.fno_blocks) == n_blocks
            assert isinstance(model.projection, eqx.nn.Conv1d)

    def test_forward_shapes(self, rng_key):
        """Test output shapes for different input configurations"""
        model = FNO1d(
            in_channels=2,
            out_channels=1,
            modes=16,
            width=32,
            activation=jax.nn.gelu,
            n_blocks=4,
            key=rng_key,
        )

        spatial_sizes = [32, 64, 128]
        for size in spatial_sizes:
            x = jax.random.normal(rng_key, (2, size))
            output = model(x)

            assert output.shape == (1, size)
            assert jnp.all(jnp.isfinite(output))

    def test_jit_compatibility(self, rng_key):
        """Test that the model can be JIT compiled"""
        model = FNO1d(
            in_channels=2,
            out_channels=1,
            modes=16,
            width=32,
            activation=jax.nn.gelu,
            n_blocks=4,
            key=rng_key,
        )

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        x = jax.random.normal(rng_key, (2, 64))
        output = forward(model, x)

        assert output.shape == (1, 64)
        assert jnp.all(jnp.isfinite(output))

    def test_parameter_updates(self, rng_key, dummy_data):
        """Test that parameters can be updated through gradient descent"""
        model = FNO1d(
            in_channels=2,
            out_channels=1,
            modes=16,
            width=32,
            activation=jax.nn.gelu,
            n_blocks=4,
            key=rng_key,
        )

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        inputs, targets = dummy_data

        def loss_fn(model):
            preds = jax.vmap(model)(inputs)
            return jnp.mean((preds - targets) ** 2)

        # Compute initial loss
        initial_loss = loss_fn(model)

        # Update parameters
        grads = eqx.filter_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # Compute new loss
        new_loss = loss_fn(model)

        # Loss should decrease
        assert new_loss < initial_loss

    def test_gradient_flow(self, rng_key, dummy_data):
        """Test that gradients flow through all parts of the network"""
        model = FNO1d(
            in_channels=2,
            out_channels=1,
            modes=16,
            width=32,
            activation=jax.nn.gelu,
            n_blocks=4,
            key=rng_key,
        )

        inputs, targets = dummy_data

        def loss_fn(model):
            preds = jax.vmap(model)(inputs)
            return jnp.mean((preds - targets) ** 2)

        grads = eqx.filter_grad(loss_fn)(model)

        # Check that gradients exist for all parameters
        def check_gradients(grad_tree):
            for grad in jax.tree_util.tree_leaves(grad_tree):
                if isinstance(grad, jnp.ndarray):
                    assert jnp.any(grad != 0)

        check_gradients(grads)

    def test_batch_invariance(self, rng_key):
        """Test that the model gives same results regardless of batch processing"""
        model = FNO1d(
            in_channels=2,
            out_channels=1,
            modes=16,
            width=32,
            activation=jax.nn.gelu,
            n_blocks=4,
            key=rng_key,
        )

        # Create two identical inputs
        x = jax.random.normal(rng_key, (2, 64))
        x_batch = jnp.stack([x, x])

        # Process individually
        y1 = model(x)

        # Process as batch
        y_batch = jax.vmap(model)(x_batch)

        # Results should be identical
        assert jnp.allclose(y1, y_batch[0], atol=1e-5)
        assert jnp.allclose(y1, y_batch[1], atol=1e-5)


def test_end_to_end_training(rng_key, dummy_data):
    """Test complete training loop"""
    model = FNO1d(
        in_channels=2,
        out_channels=1,
        modes=16,
        width=32,
        activation=jax.nn.gelu,
        n_blocks=4,
        key=rng_key,
    )

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, batch):
        inputs, targets = batch

        def loss_fn(model):
            preds = jax.vmap(model)(inputs)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # Train for a few steps
    losses = []
    for _ in range(5):
        loss, model, opt_state = make_step(model, opt_state, dummy_data)
        losses.append(loss)

    # Loss should decrease
    assert losses[-1] < losses[0]


# Tests for the FNO2d model


@pytest.fixture
def sample_input() -> Tuple[int, int, int]:
    # channels, height, width
    return (3, 32, 32)


class TestSpectralConv2d:
    def test_initialization(self, random_key):
        """Test if SpectralConv2d initializes correctly"""
        layer = SpectralConv2d(
            in_channels=2, out_channels=4, modes1=8, modes2=8, key=random_key
        )

        assert layer.in_channels == 2
        assert layer.out_channels == 4
        assert layer.modes1 == 8
        assert layer.modes2 == 8
        assert layer.real_weights.shape == (2, 4, 8, 8)
        assert layer.imag_weights.shape == (2, 4, 8, 8)

    def test_forward_pass(self, random_key, sample_input):
        """Test if forward pass produces expected output shape"""
        in_channels, height, width = sample_input
        layer = SpectralConv2d(
            in_channels=in_channels,
            out_channels=5,  # Different output channels
            modes1=8,
            modes2=8,
            key=random_key,
        )

        x = jax.random.normal(random_key, (in_channels, height, width))
        output = layer(x)

        assert output.shape == (5, height, width)  # Output channels = 5

    def test_fourier_mode_multiplication(self, random_key):
        """Test if Fourier mode multiplication is performed correctly"""
        layer = SpectralConv2d(
            in_channels=1, out_channels=1, modes1=4, modes2=4, key=random_key
        )

        # Create simple input with known Fourier transform
        x = jnp.ones((1, 16, 16))
        output = layer(x)

        # Check if output is real
        assert jnp.isclose(jnp.imag(output).sum(), 0.0, atol=1e-6)
        assert output.shape == (1, 16, 16)


class TestFNOBlock2d:
    def test_initialization(self, random_key):
        """Test if FNOBlock2d initializes correctly"""
        block = FNOBlock2d(
            in_channels=2,
            out_channels=4,
            modes1=8,
            modes2=8,
            activation=jax.nn.gelu,
            key=random_key,
        )

        assert isinstance(block.spectral_conv, SpectralConv2d)
        assert isinstance(block.conv, eqx.nn.Conv2d)
        assert block.activation == jax.nn.gelu

    def test_forward_pass(self, random_key, sample_input):
        """Test if forward pass combines spectral and regular convolution"""
        in_channels, height, width = sample_input
        block = FNOBlock2d(
            in_channels=in_channels,
            out_channels=4,
            modes1=8,
            modes2=8,
            activation=jax.nn.gelu,
            key=random_key,
        )

        x = jax.random.normal(random_key, (in_channels, height, width))
        output = block(x)

        assert output.shape == (4, height, width)
        # Check if output has non-zero values (activation applied)
        assert not jnp.allclose(output, 0.0)

    def test_residual_connection(self, random_key):
        """Test if the residual connection is working"""
        block = FNOBlock2d(
            in_channels=1,
            out_channels=1,
            modes1=4,
            modes2=4,
            activation=lambda x: x,  # Linear activation for testing
            key=random_key,
        )

        x = jnp.ones((1, 16, 16))
        output = block(x)

        # Output should be different from input due to convolutions
        assert not jnp.allclose(output, x)
        assert output.shape == x.shape


class TestFNO2d:
    def test_initialization(self, random_key):
        """Test if FNO2d initializes correctly"""
        model = FNO2d(
            in_channels=2,
            out_channels=1,
            modes1=8,
            modes2=8,
            width=32,
            activation=jax.nn.gelu,
            n_blocks=4,
            key=random_key,
        )

        assert isinstance(model.lifting, eqx.nn.Conv2d)
        assert len(model.fno_blocks) == 4
        assert isinstance(model.projection, eqx.nn.Conv2d)

    def test_forward_pass(self, random_key, sample_input):
        """Test if forward pass produces expected output shape"""
        in_channels, height, width = sample_input
        model = FNO2d(
            in_channels=in_channels,
            out_channels=1,
            modes1=8,
            modes2=8,
            width=32,
            activation=jax.nn.gelu,
            n_blocks=4,
            key=random_key,
        )

        x = jax.random.normal(random_key, (in_channels, height, width))
        output = model(x)

        assert output.shape == (1, height, width)

    def test_model_training(self, random_key):
        """Test if model can be trained on a simple problem"""
        model = FNO2d(
            in_channels=1,
            out_channels=1,
            modes1=4,
            modes2=4,
            width=16,
            activation=jax.nn.gelu,
            n_blocks=2,
            key=random_key,
        )

        # Create simple training data
        x = jnp.ones((1, 16, 16))
        y = jnp.ones((1, 16, 16)) * 2

        # Simple training step
        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        loss, grad = eqx.filter_value_and_grad(loss_fn)(model)
        assert not jnp.isnan(loss)
        assert all(not jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grad))

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_processing(self, random_key, batch_size):
        """Test if model can handle different batch sizes using vmap"""
        model = FNO2d(
            in_channels=2,
            out_channels=1,
            modes1=4,
            modes2=4,
            width=16,
            activation=jax.nn.gelu,
            n_blocks=2,
            key=random_key,
        )

        # Create batch of inputs
        x = jax.random.normal(random_key, (batch_size, 2, 16, 16))

        # Process batch using vmap
        batch_forward = jax.vmap(model)
        output = batch_forward(x)

        assert output.shape == (batch_size, 1, 16, 16)


def test_end_to_end_training():
    """Test end-to-end training on a simple Poisson problem"""
    key = jax.random.PRNGKey(0)

    # Create simple Poisson problem
    nx = ny = 16
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)

    # Source term (simple Gaussian)
    f = jnp.exp(-100 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    # Initialize model
    model = FNO2d(
        in_channels=3,  # source + coordinates
        out_channels=1,
        modes1=4,
        modes2=4,
        width=16,
        activation=jax.nn.gelu,
        n_blocks=2,
        key=key,
    )

    # Prepare input (include spatial coordinates)
    input_data = jnp.stack([f, X, Y], axis=0)

    # Simple training step
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model):
        pred = model(input_data)
        # Simple loss: prediction should be smooth and match boundary conditions
        return (
            jnp.mean(jnp.square(pred))
            + jnp.mean(jnp.square(pred[:, 0, :]))  # Boundary penalties
            + jnp.mean(jnp.square(pred[:, -1, :]))
            + jnp.mean(jnp.square(pred[:, :, 0]))
            + jnp.mean(jnp.square(pred[:, :, -1]))
        )

    # Run few training steps
    for _ in range(5):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        assert not jnp.isnan(loss)
        assert all(not jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grads))
