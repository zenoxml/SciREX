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
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os

from scirex.core.sciml.fno.models.fno_1d import FNO1d


def generate_heat_data(n_samples=1200, nx=64, nt=100, D=0.1):
    """Generate data for the heat equation:
    ∂u/∂t = D * ∂²u/∂x²
    """
    key = jax.random.PRNGKey(0)

    # Spatial domain
    L = 2 * jnp.pi  # Length of domain
    dx = L / nx
    x = jnp.linspace(0, L, nx)

    # Time domain
    T = 1.0  # Total time
    dt = T / nt
    t = jnp.linspace(0, T, nt)

    def generate_initial_condition(key):
        # Split keys for different random operations
        k1, k2, k3 = jax.random.split(key, 3)

        # Generate multiple Gaussian pulses
        max_pulses = 3
        positions = jax.random.uniform(k1, (max_pulses,)) * L
        widths = jax.random.uniform(k2, (max_pulses,)) * 0.2 + 0.1
        amplitudes = jax.random.uniform(k3, (max_pulses,)) * 0.8 + 0.2

        # Sum up the pulses
        u0 = jnp.zeros(nx)
        for pos, width, amp in zip(positions, widths, amplitudes):
            u0 += amp * jnp.exp(-((x - pos) ** 2) / (2 * width**2))

        return u0

    def solve_heat_equation(u0):
        """Solve using explicit finite differences"""
        u = jnp.zeros((nt, nx))
        u = u.at[0].set(u0)

        # Solve using central difference in space
        for n in range(1, nt):
            u = u.at[n].set(
                u[n - 1]
                + D
                * dt
                / dx**2
                * (jnp.roll(u[n - 1], 1) - 2 * u[n - 1] + jnp.roll(u[n - 1], -1))
            )
        return u

    # Generate dataset
    keys = jax.random.split(key, n_samples)
    initial_conditions = jax.vmap(generate_initial_condition)(keys)
    solutions = jax.vmap(solve_heat_equation)(initial_conditions)

    # Prepare input-output pairs
    mesh = jnp.repeat(x[jnp.newaxis, :], n_samples, axis=0)
    input_data = jnp.stack([initial_conditions, mesh], axis=1)
    output_data = solutions[:, -1:]

    return input_data, output_data, x


# Generate data
input_data, output_data, spatial_grid = generate_heat_data()

# Split into train and test sets
train_x, test_x = input_data[:1000], input_data[1000:]
train_y, test_y = output_data[:1000], output_data[1000:]

# Initialize model with the modular blocks
model = FNO1d(
    in_channels=2,  # Initial temperature + spatial coordinate
    out_channels=1,  # Final temperature
    modes=16,
    width=64,
    activation=jax.nn.gelu,
    n_blocks=4,
    key=jax.random.PRNGKey(0),
)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


# Training loop
@eqx.filter_jit
def make_step(model, opt_state, batch):
    def loss_fn(model):
        pred = jax.vmap(model)(batch[0])
        return jnp.mean((pred - batch[1]) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


# Train for a few epochs
batch_size = 50
n_epochs = 50
losses = []

for epoch in range(n_epochs):
    for i in range(0, len(train_x), batch_size):
        batch = (train_x[i : i + batch_size], train_y[i : i + batch_size])
        loss, model, opt_state = make_step(model, opt_state, batch)
        losses.append(loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Evaluate on test set
test_pred = jax.vmap(model)(test_x)
test_error = jnp.mean((test_pred - test_y) ** 2)
print(f"Test MSE: {test_error:.6f}")

# Create the output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "outputs", "heat")
os.makedirs(output_dir, exist_ok=True)

# Visualize results

plt.figure()
plt.plot(spatial_grid, test_x[0, 0], label="Initial temperature")
plt.plot(spatial_grid, test_y[0, 0], label="True solution")
plt.plot(spatial_grid, test_pred[0, 0], "--", label="FNO prediction")
plt.legend()
plt.title("Example prediction")
plt.xlabel("x")
plt.ylabel("Temperature")
output_file = os.path.join(output_dir, "example_prediction.png")
plt.savefig(output_file)

plt.figure()
plt.semilogy(losses)
plt.title("Training loss")
plt.xlabel("Step")
plt.ylabel("MSE")
output_file = os.path.join(output_dir, "training_loss.png")
plt.savefig(output_file)

plt.figure()
plt.plot(spatial_grid, jnp.abs(test_pred[0, 0] - test_y[0, 0]))
plt.title("Absolute error")
plt.xlabel("x")
plt.ylabel("|Error|")
output_file = os.path.join(output_dir, "absolute_error.png")
plt.savefig(output_file)
