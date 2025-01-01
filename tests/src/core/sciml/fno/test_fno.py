import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from typing import Tuple

from scirex.core.sciml.fno.layers.spectral_conv import SpectralConv1d
from scirex.core.sciml.fno.layers.fno_block import FNOBlock1d
from scirex.core.sciml.fno.models.fno import FNO1d


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)


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
