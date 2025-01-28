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

import jax
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from scirex.core.sciml.fno.models.fno_1d import FNO1d


def generate_burgers_data(num_samples=1200, nx=256, t_final=1.0, nu=0.01 / jnp.pi):
    """
    Generate synthetic data for the Burgers equation:
    u_t + u*u_x = nu*u_xx
    with random initial conditions
    """
    x = jnp.linspace(0, 2 * jnp.pi, nx)
    dx = x[1] - x[0]

    # Generate random initial conditions (smooth functions)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_samples)

    def generate_ic(key):
        # Generate random coefficients for Fourier series with smaller amplitudes
        n_modes = 4
        key1, key2 = jax.random.split(key)
        a_n = jax.random.normal(key1, (n_modes,)) * 0.1  # Reduced amplitude
        b_n = jax.random.normal(key2, (n_modes,)) * 0.1  # Reduced amplitude

        # Construct smooth initial condition
        u0 = jnp.zeros_like(x)
        for n in range(n_modes):
            u0 += a_n[n] * jnp.sin((n + 1) * x) + b_n[n] * jnp.cos((n + 1) * x)
        return u0

    initial_conditions = jax.vmap(generate_ic)(keys)

    # Use smaller timesteps for more stable integration
    dt = 0.0005  # Reduced timestep
    nt = int(t_final / dt)

    def solve_burgers(u0):
        u = u0
        for _ in range(nt):
            # Spatial derivatives using central differences
            u_x = (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2 * dx)
            u_xx = (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / (dx**2)

            # Forward Euler step with stability check
            du = -dt * (u * u_x - nu * u_xx)
            u = u + jnp.clip(du, -1.0, 1.0)  # Limit the update magnitude
        return u

    # Solve for all initial conditions
    final_solutions = jax.vmap(solve_burgers)(initial_conditions)

    # Format data similar to the original
    a = initial_conditions[:, jnp.newaxis, :]
    u = final_solutions[:, jnp.newaxis, :]

    # Normalize the data
    a = (a - jnp.mean(a)) / (jnp.std(a) + 1e-8)
    u = (u - jnp.mean(u)) / (jnp.std(u) + 1e-8)

    # Add mesh information
    mesh = jnp.linspace(0, 2 * jnp.pi, nx)
    mesh_shape_corrected = jnp.repeat(
        mesh[jnp.newaxis, jnp.newaxis, :], num_samples, axis=0
    )
    a_with_mesh = jnp.concatenate((a, mesh_shape_corrected), axis=1)

    return a_with_mesh, u


# Generate synthetic data
print("Generating synthetic data...")
a_with_mesh, u = generate_burgers_data()

# Split into train and test sets
train_x, test_x = a_with_mesh[:1000], a_with_mesh[1000:1200]
train_y, test_y = u[:1000], u[1000:1200]

# Initialize the FNO model
fno = FNO1d(
    2,  # in_channels (initial condition + mesh)
    1,  # out_channels (solution at t=1)
    5,  # width (number of hidden channels)
    4,  # modes (number of Fourier modes)
    jax.nn.relu,  # activation function
    4,  # number of layers
    key=jax.random.PRNGKey(0),
)


def dataloader(
    key,
    dataset_x,
    dataset_y,
    batch_size,
):
    n_samples = dataset_x.shape[0]
    n_batches = int(jnp.ceil(n_samples / batch_size))
    permutation = jax.random.permutation(key, n_samples)

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, n_samples)
        batch_indices = permutation[start:end]
        yield dataset_x[batch_indices], dataset_y[batch_indices]


def loss_fn(model, x, y):
    y_pred = jax.vmap(model)(x)
    # Add L2 regularization
    params = eqx.filter(model, eqx.is_array)
    l2_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    mse = jnp.mean(jnp.square(y_pred - y))
    return mse + 1e-5 * l2_reg


# Use a more stable optimizer configuration
optimizer = optax.chain(
    optax.clip(1.0), optax.adam(1e-4)  # Gradient clipping  # Reduced learning rate
)
opt_state = optimizer.init(eqx.filter(fno, eqx.is_array))


@eqx.filter_jit
def make_step(model, state, x, y):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    val_loss = loss_fn(model, test_x[..., ::32], test_y[..., ::32])
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss, val_loss


# Training loop with early stopping
loss_history = []
val_loss_history = []
best_val_loss = float("inf")
patience = 10
patience_counter = 0

shuffle_key = jax.random.PRNGKey(10)
for epoch in tqdm(range(200)):
    shuffle_key, subkey = jax.random.split(shuffle_key)
    epoch_losses = []
    epoch_val_losses = []

    for batch_x, batch_y in dataloader(
        subkey,
        train_x[..., ::32],
        train_y[..., ::32],
        batch_size=100,
    ):
        fno, opt_state, loss, val_loss = make_step(fno, opt_state, batch_x, batch_y)
        epoch_losses.append(loss)
        epoch_val_losses.append(val_loss)

    avg_loss = jnp.mean(jnp.array(epoch_losses))
    avg_val_loss = jnp.mean(jnp.array(epoch_val_losses))
    loss_history.append(avg_loss)
    val_loss_history.append(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Evaluate the model
test_pred = jax.vmap(fno)(test_x)


def relative_l2_norm(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / (ref_norm + 1e-8)  # Added small constant for stability


rel_l2_set = jax.vmap(relative_l2_norm)(test_pred, test_y)
print(f"Mean relative L2 error: {jnp.mean(rel_l2_set):.3e}")

# Plotting code remains the same...
import os

# Create the output directory
output_dir = os.path.join(os.path.dirname(__file__), "outputs", "burgers")
os.makedirs(output_dir, exist_ok=True)

# Plot initial condition vs final solution
plt.figure()
plt.plot(
    jnp.linspace(0, 2 * jnp.pi, train_x.shape[-1]),
    train_x[0, 0],
    label="Initial condition",
)
plt.plot(
    jnp.linspace(0, 2 * jnp.pi, train_y.shape[-1]),
    train_y[0, 0],
    label="After 1 time unit",
)
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "initial_vs_after.png"))
plt.close()

# Plot training history
plt.figure()
plt.plot(loss_history, label="Train loss")
plt.plot(val_loss_history, label="Validation loss")
plt.legend()
plt.yscale("log")
plt.grid()
plt.savefig(os.path.join(output_dir, "loss.png"))
plt.close()

# Plot prediction comparison
plt.figure()
plt.plot(test_x[1, 0, ::32], label="Initial condition")
plt.plot(test_y[1, 0, ::32], label="Ground Truth")
plt.plot(fno(test_x[1, :, ::32])[0], label="FNO prediction")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "prediction.png"))
plt.close()
