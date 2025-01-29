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

from scirex.core.sciml.fno.models.fno_2d import FNO2d


def generate_poisson_data(n_samples=1200, nx=64, ny=64):
    """Generate data for the 2D Poisson equation:
    ∇²u = f with Dirichlet boundary conditions

    The equation: -∂²u/∂x² - ∂²u/∂y² = f(x,y)
    Domain: [0,1] × [0,1]
    Boundary conditions: u = 0 on the boundary
    """
    key = jax.random.PRNGKey(0)

    # Spatial domain [0,1] × [0,1]
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)

    def generate_source_term(key):
        """Generate random source term f(x,y) as a sum of Gaussian functions"""
        k1, k2, k3 = jax.random.split(key, 3)
        n_sources = 3

        # Random parameters for Gaussian sources
        amplitudes = jax.random.uniform(k1, (n_sources,), minval=-1.0, maxval=1.0)
        centers_x = jax.random.uniform(k2, (n_sources,), minval=0.2, maxval=0.8)
        centers_y = jax.random.uniform(k3, (n_sources,), minval=0.2, maxval=0.8)

        f = jnp.zeros((nx, ny))
        for amp, cx, cy in zip(amplitudes, centers_x, centers_y):
            f += amp * jnp.exp(-50 * ((X - cx) ** 2 + (Y - cy) ** 2))

        return f

    def solve_poisson(f):
        """Solve Poisson equation using spectral method"""
        # Wave numbers
        kx = 2 * jnp.pi * jnp.fft.fftfreq(nx)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(ny)
        KX, KY = jnp.meshgrid(kx, ky)

        # Compute solution in Fourier space
        f_hat = jnp.fft.fft2(f)
        denominator = -(KX**2 + KY**2)
        denominator = denominator.at[0, 0].set(1)  # Avoid division by zero
        u_hat = f_hat / denominator
        u_hat = u_hat.at[0, 0].set(0)  # Set mean to zero

        # Transform back to real space
        u = jnp.real(jnp.fft.ifft2(u_hat))

        # Enforce Dirichlet boundary conditions
        u = u.at[0, :].set(0)
        u = u.at[-1, :].set(0)
        u = u.at[:, 0].set(0)
        u = u.at[:, -1].set(0)

        return u

    # Generate dataset
    keys = jax.random.split(key, n_samples)
    source_terms = jax.vmap(generate_source_term)(keys)
    solutions = jax.vmap(solve_poisson)(source_terms)

    # Prepare input-output pairs
    # Include spatial coordinates in input
    mesh_x = jnp.repeat(X[jnp.newaxis, :, :], n_samples, axis=0)
    mesh_y = jnp.repeat(Y[jnp.newaxis, :, :], n_samples, axis=0)
    input_data = jnp.stack([source_terms, mesh_x, mesh_y], axis=1)
    output_data = solutions[:, jnp.newaxis, :, :]

    return input_data, output_data, x, y


# Generate training data
input_data, output_data, x, y = generate_poisson_data()

# Split into train and test sets
train_x, test_x = input_data[:1000], input_data[1000:]
train_y, test_y = output_data[:1000], output_data[1000:]

# Initialize model
model = FNO2d(
    in_channels=3,  # Source term + x,y coordinates
    out_channels=1,  # Solution u(x,y)
    modes1=12,  # Number of Fourier modes in x
    modes2=12,  # Number of Fourier modes in y
    width=64,  # Width of the network
    activation=jax.nn.gelu,
    n_blocks=4,
    key=jax.random.PRNGKey(0),
)

# Training setup
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def make_step(model, opt_state, batch):
    def loss_fn(model):
        pred = jax.vmap(model)(batch[0])
        return jnp.mean((pred - batch[1]) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


# Training loop
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

# Evaluate and visualize results
test_pred = jax.vmap(model)(test_x)
test_error = jnp.mean((test_pred - test_y) ** 2)
print(f"Test MSE: {test_error:.6f}")

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson")
os.makedirs(output_dir, exist_ok=True)

# Plot example prediction
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(test_x[0, 0], cmap="viridis")
plt.colorbar()
plt.title("Source term f(x,y)")

plt.subplot(132)
plt.imshow(test_y[0, 0], cmap="viridis")
plt.colorbar()
plt.title("True solution u(x,y)")

plt.subplot(133)
plt.imshow(test_pred[0, 0], cmap="viridis")
plt.colorbar()
plt.title("FNO prediction")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "poisson_example.png"))

# Plot training loss
plt.figure()
plt.semilogy(losses)
plt.title("Training loss")
plt.xlabel("Step")
plt.ylabel("MSE")
plt.savefig(os.path.join(output_dir, "poisson_loss.png"))

# Plot error
plt.figure()
plt.imshow(jnp.abs(test_pred[0, 0] - test_y[0, 0]), cmap="viridis")
plt.colorbar()
plt.title("Absolute error")
plt.savefig(os.path.join(output_dir, "poisson_error.png"))
