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
Example: Fourier Neural Operator (FNO) for 2D Poisson Equation (PyTorch)

Problem:
    -∇²u(x,y) = f(x,y),   (x,y) ∈ [0,1]²
    u = 0 on ∂Ω

Input  : f(x,y), x, y
Output : u(x,y)

"""


# Imports

import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from scirex.torch.sciml.models.fno.simple_fno import SimpleFNO


# Data generation

def generate_poisson_data(n_samples=200, nx=64, ny=64):
    """
    Generate data for the 2D Poisson equation:
        -∂²u/∂x² - ∂²u/∂y² = f(x,y)

    Domain:
        [0,1] × [0,1]

    Boundary conditions:
        u = 0 on the boundary

    A spectral Poisson solver is used to compute
    the ground-truth solution.
    """

    # Spatial domain [0,1] × [0,1]
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    def generate_source_term():
        """
        Generate a random source term f(x,y)
        as a sum of Gaussian functions.
        """

        n_sources = 3

        # Random parameters for Gaussian sources
        amplitudes = torch.empty(n_sources).uniform_(-1.0, 1.0)
        centers_x = torch.empty(n_sources).uniform_(0.2, 0.8)
        centers_y = torch.empty(n_sources).uniform_(0.2, 0.8)

        f = torch.zeros(nx, ny)

        for amp, cx, cy in zip(amplitudes, centers_x, centers_y):
            f += amp * torch.exp(
                -50.0 * ((X - cx) ** 2 + (Y - cy) ** 2)
            )

        return f

    def solve_poisson(f):
        """
        Solve the Poisson equation using a spectral method.

        Steps:
        1. Transform f(x,y) to Fourier space
        2. Solve algebraic equation in Fourier domain
        3. Transform back to physical space
        """

        # Wave numbers
        kx = 2 * torch.pi * torch.fft.fftfreq(nx)
        ky = 2 * torch.pi * torch.fft.fftfreq(ny)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")

        # Fourier transform of source term
        f_hat = torch.fft.fft2(f)

        # Spectral Poisson operator
        denominator = -(KX**2 + KY**2)
        denominator[0, 0] = 1.0  # avoid division by zero

        # Solve in Fourier space
        u_hat = f_hat / denominator
        u_hat[0, 0] = 0.0  # enforce zero mean

        # Transform back to real space
        u = torch.real(torch.fft.ifft2(u_hat))

        # Enforce Dirichlet boundary conditions
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

        return u

    # Generate dataset
    inputs, outputs = [], []

    for _ in range(n_samples):
        f = generate_source_term()
        u = solve_poisson(f)

        # Input: [f, x, y]
        inputs.append(torch.stack([f, X, Y], dim=0))

        # Output: solution u(x,y)
        outputs.append(u.unsqueeze(0))

    return torch.stack(inputs), torch.stack(outputs), x, y


# Generate training data
input_data, output_data, x, y = generate_poisson_data()

# Split into train and test sets
train_x, test_x = input_data[:160], input_data[160:]
train_y, test_y = output_data[:160], output_data[160:]




class PoissonDataset(Dataset):
    """Dataset wrapper for Poisson equation data."""

    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_loader = DataLoader(
    PoissonDataset(train_x, train_y),
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    PoissonDataset(test_x, test_y),
    batch_size=1,
    shuffle=False
)


# Model

model = SimpleFNO(
    n_modes=(12, 12),
    in_channels=3,     # f(x,y), x, y
    out_channels=1,    # u(x,y)
    hidden_channels=64,
    n_layers=4,
    positional_embedding="grid",
    use_channel_mlp=True,
    channel_mlp_expansion=0.5,
    channel_mlp_skip="soft-gating",
    norm=None,
    fno_skip="linear",
    domain_padding=0.05,
    fno_block_precision="full",
    implementation="factorized",
    complex_data=False
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()


# Training loop

epochs = 300
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.6e}")


# Evaluation

model.eval()
with torch.no_grad():
    xb, y_true = next(iter(test_loader))
    y_pred = model(xb)


# Create output directory

output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson")
os.makedirs(output_dir, exist_ok=True)

# Visualization

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(xb[0, 0], cmap="viridis")
plt.colorbar()
plt.title("Source term f(x,y)")

plt.subplot(132)
plt.imshow(y_true[0, 0], cmap="viridis")
plt.colorbar()
plt.title("True solution u(x,y)")

plt.subplot(133)
plt.imshow(y_pred[0, 0], cmap="viridis")
plt.colorbar()
plt.title("FNO prediction")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "poisson_example.png"))

plt.figure()
plt.semilogy(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training loss")
plt.savefig(os.path.join(output_dir, "poisson_loss.png"))

plt.figure()
plt.imshow(torch.abs(y_pred[0, 0] - y_true[0, 0]), cmap="viridis")
plt.colorbar()
plt.title("Absolute error")
plt.savefig(os.path.join(output_dir, "poisson_error.png"))
