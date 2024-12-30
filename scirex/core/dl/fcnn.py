import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Array, Float, PyTree
from typing import Callable, Tuple
import equinox as eqx
import optax


class Model(eqx.Module):
    """A simple neural network model"""

    layers: list

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: Array) -> Array:
        """Forward pass of the model"""
        for layer in self.layers:
            x = layer(x)
        return x


def cross_entropy_loss(model: Model, x: Array, y: Array) -> Float[Array, ""]:
    """
    Compute the cross-entropy loss

    Args:
        model: Neural network model
        x: Batch of input data
        y: Batch of target labels
    """
    logits = jax.vmap(model)(x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, jnp.expand_dims(y, 1), axis=1)
    return jnp.mean(nll)


@eqx.filter_jit
def compute_accuracy(model: Model, x: Array, y: Array) -> Float[Array, ""]:
    """
    Compute accuracy

    Args:
        model: Neural network model
        x: Batch of input data
        y: Batch of target labels
    """
    pred_y = jax.vmap(model)(x)
    pred_class = jnp.argmax(pred_y, axis=1)
    return jnp.mean(pred_class == y)


def evaluate(
    model: Model, x: Array, y: Array
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """
    Evaluate model on data

    Args:
        model: Neural network model
        x: Batched input data
        y: Batched target labels

    Returns:
        Tuple of (average loss, average accuracy)
    """
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(x)

    for batch_x, batch_y in zip(x, y):
        loss = cross_entropy_loss(model, batch_x, batch_y)
        acc = compute_accuracy(model, batch_x, batch_y)
        total_loss += loss
        total_acc += acc

    return total_loss / n_batches, total_acc / n_batches


@eqx.filter_jit
def make_step(
    model: Model,
    opt_state: PyTree,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformation,
) -> Tuple[Model, PyTree, Float[Array, ""]]:
    """
    Perform a single training step

    Args:
        model: Neural network model
        opt_state: Optimizer state
        x: Batch of input data
        y: Batch of target labels
        optimizer: Optax optimizer

    Returns:
        Tuple of (updated model, updated optimizer state, loss value)
    """
    loss_val, grads = eqx.filter_value_and_grad(cross_entropy_loss)(model, x, y)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def train(
    model: Model,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    optim: optax.GradientTransformation,
    num_epochs: int,
    print_every: int,
) -> Model:
    """
    Train the model

    Args:
        model: Neural network model
        x_train: Training data batches (shape: [num_batches, batch_size, ...])
        y_train: Training labels batches (shape: [num_batches, batch_size])
        x_test: Test data batches
        y_test: Test labels batches
        optim: Optax optimizer
        num_epochs: Number of training epochs
        print_every: Print progress every N batches

    Returns:
        Trained model
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    num_batches = len(x_train)

    for epoch in range(num_epochs):
        total_loss = 0.0

        # Training loop
        for batch_idx, (batch_x, batch_y) in enumerate(zip(x_train, y_train)):
            # Perform optimization step
            model, opt_state, loss_val = make_step(
                model, opt_state, batch_x, batch_y, optim
            )
            total_loss += loss_val

            # Print progress
            if (batch_idx + 1) % print_every == 0:
                avg_loss = total_loss / print_every
                # Evaluate on test set
                test_loss, test_acc = evaluate(model, x_test, y_test)
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}"
                )
                print(
                    f"Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                )
                total_loss = 0.0

        # End of epoch evaluation
        test_loss, test_acc = evaluate(model, x_test, y_test)
        print(f"\nEpoch {epoch+1}/{num_epochs} completed")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")

    return model
