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


def generate_wave_data(n_samples=1200, nx=64, nt=100, c=1.0):
    """Generate data for the wave equation:
    ∂²u/∂t² = c² * ∂²u/∂x²
    """
    key = jax.random.PRNGKey(0)

    # Spatial domain
    L = 2 * jnp.pi  # Length of domain
    dx = L / nx
    x = jnp.linspace(0, L, nx)

    # Time domain
    T = 2.0  # Total time
    dt = T / nt
    t = jnp.linspace(0, T, nt)

    def generate_initial_conditions(key):
        # Split keys for different random operations
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Generate initial displacement (sum of Gaussian pulses)
        max_pulses = 2
        positions = jax.random.uniform(k1, (max_pulses,)) * L
        widths = jax.random.uniform(k2, (max_pulses,)) * 0.2 + 0.1
        amplitudes = jax.random.uniform(k3, (max_pulses,)) * 0.8 + 0.2

        # Initial displacement
        u0 = jnp.zeros(nx)
        for pos, width, amp in zip(positions, widths, amplitudes):
            u0 += amp * jnp.exp(-((x - pos) ** 2) / (2 * width**2))

        # Initial velocity (smooth random function)
        v0 = jax.random.normal(k4, (nx,)) * 0.1
        v0 = jnp.convolve(v0, jnp.ones(10) / 10, mode="same")  # Smoothing

        return u0, v0

    def solve_wave_equation(init_conditions):
        """Solve using central differences in time and space"""
        u0, v0 = init_conditions

        # Initialize solution array
        u = jnp.zeros((nt, nx))
        u = u.at[0].set(u0)

        # Set first timestep using initial velocity
        u = u.at[1].set(u0 + dt * v0)

        # Solve using central difference in time and space
        for n in range(2, nt):
            u = u.at[n].set(
                2 * u[n - 1]
                - u[n - 2]
                + (c * dt / dx) ** 2
                * (jnp.roll(u[n - 1], 1) - 2 * u[n - 1] + jnp.roll(u[n - 1], -1))
            )
        return u

    # Generate dataset
    keys = jax.random.split(key, n_samples)
    initial_conditions = jax.vmap(generate_initial_conditions)(keys)
    solutions = jax.vmap(solve_wave_equation)(initial_conditions)

    # Prepare input-output pairs
    mesh = jnp.repeat(x[jnp.newaxis, :], n_samples, axis=0)
    input_data = jnp.stack(
        [
            initial_conditions[0],  # displacement
            initial_conditions[1],  # velocity
            mesh,
        ],
        axis=1,
    )
    output_data = solutions[:, -1:]

    return input_data, output_data, x


# Generate data
input_data, output_data, spatial_grid = generate_wave_data()

# Split into train and test sets
train_x, test_x = input_data[:1000], input_data[1000:]
train_y, test_y = output_data[:1000], output_data[1000:]

# Initialize model with the modular blocks
model = FNO1d(
    in_channels=3,  # Initial displacement + initial velocity + spatial coordinate
    out_channels=1,  # Final displacement
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

# Visualize results
# Create the output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "outputs", "wave")
os.makedirs(output_dir, exist_ok=True)

plt.figure()
plt.plot(spatial_grid, test_x[0, 0], label="Initial displacement")
plt.plot(spatial_grid, test_x[0, 1], label="Initial velocity")
plt.plot(spatial_grid, test_y[0, 0], label="True solution")
plt.plot(spatial_grid, test_pred[0, 0], "--", label="FNO prediction")
plt.legend()
plt.title("Example prediction")
plt.xlabel("x")
plt.ylabel("Displacement")
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
