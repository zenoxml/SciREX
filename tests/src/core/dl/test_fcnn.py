import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import List
from scirex.core.dl.fcnn import (
    Model,
    cross_entropy_loss,
    compute_accuracy,
    evaluate,
    make_step,
    train,
)


class DummyLayer(eqx.Module):
    """A simple dummy layer for testing"""

    weight: jax.Array

    def __init__(self, key: jax.random.PRNGKey):
        self.weight = jax.random.normal(key, (2, 2))

    def __call__(self, x):
        return jnp.dot(x, self.weight)


@pytest.fixture
def simple_model():
    """Fixture to create a simple model with dummy layers"""
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    layers = [
        DummyLayer(key1),
        DummyLayer(key2),
        lambda x: jax.nn.log_softmax(
            x, axis=-1
        ),  # Add softmax for proper loss computation
    ]
    return Model(layers)


@pytest.fixture
def nn_model():
    """Fixture to create a model with equinox layers"""
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 3)

    layers = [
        eqx.nn.Linear(784, 128, key=keys[0]),
        jax.nn.relu,
        eqx.nn.Linear(128, 64, key=keys[1]),
        jax.nn.relu,
        eqx.nn.Linear(64, 10, key=keys[2]),
    ]
    return Model(layers)


@pytest.fixture
def dummy_data():
    """Fixture to create dummy data for testing simple model"""
    # Create 4 samples with 2 features each
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = jnp.array([0, 1, 0, 1])
    return x, y


@pytest.fixture
def nn_data():
    """Fixture to create dummy data for testing nn model"""
    batch_size = 4
    x = jnp.ones((batch_size, 784))  # MNIST-like flattened images
    y = jnp.array([0, 1, 2, 3])  # Multiple classes
    return x, y


def test_model_initialization(simple_model):
    """Test if the model initializes correctly"""
    assert isinstance(simple_model, Model)
    assert len(simple_model.layers) == 3
    assert isinstance(simple_model.layers[0], DummyLayer)


def test_nn_model_initialization(nn_model):
    """Test if the nn model initializes correctly"""
    assert isinstance(nn_model, Model)
    assert len(nn_model.layers) == 5
    assert isinstance(nn_model.layers[0], eqx.nn.Linear)
    assert callable(nn_model.layers[1])  # ReLU function
    assert isinstance(nn_model.layers[2], eqx.nn.Linear)


def test_model_forward_pass(simple_model, dummy_data):
    """Test if the model's forward pass works"""
    x, _ = dummy_data
    output = simple_model(x[0])
    assert isinstance(output, jax.Array)
    assert output.shape == (2,)


def test_nn_model_forward_pass(nn_model, nn_data):
    """Test if the nn model's forward pass works"""
    x, _ = nn_data
    output = nn_model(x[0])
    assert isinstance(output, jax.Array)
    assert output.shape == (10,)  # 10 classes output


def test_cross_entropy_loss(simple_model, dummy_data):
    """Test if loss computation works"""
    x, y = dummy_data
    loss = cross_entropy_loss(simple_model, x, y)
    assert isinstance(loss, jax.Array)
    assert loss.shape == ()  # scalar
    assert not jnp.isnan(loss).any()


def test_compute_accuracy(nn_model, nn_data):
    """Test if accuracy computation works"""
    x, y = nn_data
    accuracy = compute_accuracy(nn_model, x, y)
    assert isinstance(accuracy, jax.Array)
    assert accuracy.shape == ()  # scalar
    assert 0 <= float(accuracy) <= 1
    assert not jnp.isnan(accuracy)


def test_evaluate(nn_model):
    """Test if evaluation works on batched data"""
    # Create batched test data: 2 batches of 4 samples each
    x_test = jnp.ones((2, 4, 784))  # [num_batches, batch_size, features]
    y_test = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])  # [num_batches, batch_size]

    loss, accuracy = evaluate(nn_model, x_test, y_test)
    assert isinstance(loss, jax.Array)
    assert isinstance(accuracy, jax.Array)
    assert loss.shape == ()  # scalar
    assert accuracy.shape == ()  # scalar
    assert not jnp.isnan(loss)
    assert not jnp.isnan(accuracy)
    assert 0 <= float(accuracy) <= 1


def test_make_step(nn_model, nn_data):
    """Test if make_step function works correctly"""
    x, y = nn_data
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(eqx.filter(nn_model, eqx.is_array))

    updated_model, updated_opt_state, loss = make_step(
        nn_model, opt_state, x, y, optimizer
    )

    assert isinstance(updated_model, Model)
    assert isinstance(loss, jax.Array)
    assert loss.shape == ()
    assert not jnp.isnan(loss)


def test_train(nn_model):
    """Test if training works"""
    # Create training data: 2 batches of 4 samples each
    x_train = jnp.ones((2, 4, 784))
    y_train = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])

    # Create test data
    x_test = x_train
    y_test = y_train

    optimizer = optax.adam(learning_rate=0.001)

    trained_model = train(
        model=nn_model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        optim=optimizer,
        num_epochs=2,
        print_every=1,
    )

    assert isinstance(trained_model, Model)
    assert len(trained_model.layers) == len(nn_model.layers)

    # Test if model parameters have been updated
    for old_layer, new_layer in zip(nn_model.layers, trained_model.layers):
        if isinstance(old_layer, eqx.nn.Linear):
            assert not jnp.array_equal(old_layer.weight, new_layer.weight)
            assert not jnp.array_equal(old_layer.bias, new_layer.bias)


def test_invalid_input_shapes(nn_model):
    """Test if model handles invalid input shapes appropriately"""
    with pytest.raises(Exception):  # Could be ValueError or JAXException
        invalid_input = jnp.ones((5,))  # Wrong shape
        nn_model(invalid_input)


def test_input_output_shapes(nn_model, nn_data):
    """Test input/output shapes throughout the model"""
    x, y = nn_data

    # Test forward pass shapes
    output = nn_model(x[0])
    assert output.shape == (10,)

    # Test loss shape
    loss = cross_entropy_loss(nn_model, x, y)
    assert loss.shape == ()

    # Test accuracy shape
    acc = compute_accuracy(nn_model, x, y)
    assert acc.shape == ()

    # Test evaluation shapes
    x_batched = jnp.stack([x, x])
    y_batched = jnp.stack([y, y])
    loss, acc = evaluate(nn_model, x_batched, y_batched)
    assert loss.shape == ()
    assert acc.shape == ()
