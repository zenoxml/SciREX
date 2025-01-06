# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab,
# Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# SciREX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with SciREX. If not, see <https://www.gnu.org/licenses/>.
#
# For any clarifications or special considerations,
# please contact <scirex@zenteiq.ai>

"""
Example script for solving a 2D Poisson equation using FastvPINNs.

Author: Thivin Anandh (https://thivinanandh.github.io/)

Versions:
    - 27-Dec-2024 (Version 0.1): Initial Implementation
"""

# Common library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import time
from tqdm import tqdm
from pyDOE import lhs

# Fastvpinns Modules
from scirex.core.sciml.geometry.geometry_2d import Geometry_2D
from scirex.core.sciml.fe.fespace2d import Fespace2D
from scirex.core.sciml.fastvpinns.data.datahandler2d import DataHandler2D
from scirex.core.sciml.pinns.model.model import DenseModel
from scirex.core.sciml.pinns.physics.poisson2d import pde_loss_poisson2d

i_mesh_type = "quadrilateral"  # "quadrilateral"
i_mesh_generation_method = "internal"  # "internal" or "external"
i_x_min = 0  # minimum x value
i_x_max = 1  # maximum x value
i_y_min = 0  # minimum y value
i_y_max = 1  # maximum y value
i_n_collocation_points = 1600  # Number of collocation points
i_n_boundary_points = 400  # Number of points on the boundary
i_output_path = "output/poisson_2d"  # Output path

i_n_test_points_x = 100  # Number of test points in the x direction
i_n_test_points_y = 100  # Number of test points in the y direction

i_activation = "tanh"  # Activation function
# Neural Network Variables
i_learning_rate_dict = {
    "initial_learning_rate": 0.001,  # Initial learning rate
    "use_lr_scheduler": True,  # Use learning rate scheduler
    "decay_steps": 1000,  # Decay steps
    "decay_rate": 0.96,  # Decay rate
    "staircase": True,  # Staircase Decay
}

i_dtype = tf.float32
i_num_epochs = 20000  # Number of epochs

# penalty parameter for boundary loss
i_beta = 100.0


#############################################################################
## Generate the input data (Collacation points) within the domain
#############################################################################

# Generate the collocation points in the domain between x_min and x_max and y_min and y_max
input_points = lhs(2, i_n_collocation_points)

# Rescale the points to the x_min, x_max, y_min and y_max
input_points[:, 0] = input_points[:, 0] * (i_x_max - i_x_min) + i_x_min
input_points[:, 1] = input_points[:, 1] * (i_y_max - i_y_min) + i_y_min


#############################################################################
## Generate the Boundary Data - Both the input and the output points
#############################################################################


def boundary_data(x, y):
    """
    Function to generate the boundary data for the Poisson 2D problem
    """
    val = 0
    return np.ones_like(x) * val


i_n_boundary_points_per_edge = i_n_boundary_points // 4
# top boundary - y = i_y_max
top_x = np.linspace(i_x_min, i_x_max, i_n_boundary_points_per_edge)
top_y = np.ones(i_n_boundary_points_per_edge) * i_y_max
# concatenate the points into 2D array
top_boundary_points = np.vstack((top_x, top_y)).T
top_boundary_values = boundary_data(top_x, top_y)

# bottom boundary - y = i_y_min
bottom_x = np.linspace(i_x_min, i_x_max, i_n_boundary_points_per_edge)
bottom_y = np.ones(i_n_boundary_points_per_edge) * i_y_min
# concatenate the points into 2D array
bottom_boundary_points = np.vstack((bottom_x, bottom_y)).T
bottom_boundary_values = boundary_data(bottom_x, bottom_y)

# left boundary - x = i_x_min
left_x = np.ones(i_n_boundary_points_per_edge) * i_x_min
left_y = np.linspace(i_y_min, i_y_max, i_n_boundary_points_per_edge)
# concatenate the points into 2D array
left_boundary_points = np.vstack((left_x, left_y)).T
left_boundary_values = boundary_data(left_x, left_y)

# right boundary - x = i_x_max
right_x = np.ones(i_n_boundary_points_per_edge) * i_x_max
right_y = np.linspace(i_y_min, i_y_max, i_n_boundary_points_per_edge)
# concatenate the points into 2D array
right_boundary_points = np.vstack((right_x, right_y)).T
right_boundary_values = boundary_data(right_x, right_y)

# concatenate all the boundary points into a single array
boundary_points = np.vstack(
    (
        top_boundary_points,
        bottom_boundary_points,
        left_boundary_points,
        right_boundary_points,
    )
)
boundary_values = np.concatenate(
    (
        top_boundary_values,
        bottom_boundary_values,
        left_boundary_values,
        right_boundary_values,
    )
)


# Plot the boundary points
plt.figure(figsize=(6, 6))
plt.scatter(
    boundary_points[:, 0], boundary_points[:, 1], s=1, c="r", label="Boundary Points"
)
plt.scatter(
    input_points[:, 0], input_points[:, 1], s=1, c="b", label="Collocation Points"
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Input and Boundary Points")
plt.legend()
plt.savefig("points.png")


#############################################################################
## Fill the RHS of the Poisson Equation
#############################################################################


def rhs(x, y):
    """
    Function to generate the right hand side of the Poisson 2D problem
    """
    omegaX = 2.0 * np.pi
    omegaY = 2.0 * np.pi
    f_temp = -2.0 * (omegaX**2) * (np.sin(omegaX * x) * np.sin(omegaY * y))

    return f_temp


# For all the collocation points, compute the right hand side
rhs_values = rhs(input_points[:, 0], input_points[:, 1])


#############################################################################
## Setup Test points
#############################################################################

# Generate the test points in the domain between x_min and x_max and y_min and y_max
test_points_x = np.linspace(i_x_min, i_x_max, i_n_test_points_x)
test_points_y = np.linspace(i_y_min, i_y_max, i_n_test_points_y)
test_points = np.array(np.meshgrid(test_points_x, test_points_y)).T.reshape(-1, 2)

#############################################################################
## Setup PDE Parameters Dictionary
#############################################################################
i_bilinear_params_dict = {"eps": tf.constant(1.0, dtype=i_dtype)}

#############################################################################
## Define the exact solution for the Poisson 2D problem
#############################################################################


def exact_solution(x, y):
    """
    Function to generate the exact solution for the Poisson 2D problem
    """
    omegaX = 2.0 * np.pi
    omegaY = 2.0 * np.pi
    val = -1.0 * np.sin(omegaX * x) * np.sin(omegaY * y)

    return val


exact_values = exact_solution(test_points[:, 0], test_points[:, 1])


#############################################################################
##  Convert the input data into tensors
#############################################################################

# Convert the input points, boundary points, boundary values and rhs values into tensors
input_points = tf.convert_to_tensor(input_points, dtype=tf.float32)
boundary_points = tf.convert_to_tensor(boundary_points, dtype=tf.float32)
boundary_values = tf.reshape(
    tf.convert_to_tensor(boundary_values, dtype=tf.float32), (-1, 1)
)
rhs_values = tf.reshape(tf.convert_to_tensor(rhs_values, dtype=tf.float32), (-1, 1))
test_points = tf.convert_to_tensor(test_points, dtype=tf.float32)

# convert input points into numpy array and plot
input_points_np = input_points.numpy()
plt.figure(figsize=(6, 6))
plt.scatter(
    input_points_np[:, 0], input_points_np[:, 1], s=1, c="b", label="Collocation Points"
)
plt.xlabel("x")
plt.savefig("input_points.png")


## CREATE OUTPUT FOLDER
# use pathlib to create the folder,if it does not exist
folder = Path(i_output_path)
# create the folder if it does not exist
if not folder.exists():
    folder.mkdir(parents=True, exist_ok=True)


model = DenseModel(
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict,
    loss_function=pde_loss_poisson2d,
    input_tensors_list=[
        input_points,
        boundary_points,
        boundary_values,
    ],
    force_function_values=rhs_values,
    tensor_dtype=i_dtype,
    activation=i_activation,
)


loss_array = []  # total loss
time_array = []  # time taken for each epoch


for epoch in tqdm(range(i_num_epochs)):
    # Train the model
    batch_start_time = time.time()
    loss = model.train_step(beta=i_beta, bilinear_params_dict=i_bilinear_params_dict)
    elapsed = time.time() - batch_start_time
    # print(elapsed)
    time_array.append(elapsed)

    loss_array.append(loss["loss"])

    if epoch % 1000 == 0:
        model_sol = model(test_points).numpy().reshape(-1)
        error = np.abs(exact_values - model_sol)
        l_inf_error = np.max(np.abs(error))
        pde_loss = loss["loss_pde"]
        boundary_loss = loss["loss_dirichlet"]
        print(
            f"Epoch: {epoch}, Loss: {loss['loss'] :.4e}, L2 Error: {np.sqrt(np.mean(error**2)):.4e}, L_inf Error: {l_inf_error:.4e}",
            f"PDE Loss: {pde_loss:.4e}, Boundary Loss: {boundary_loss:.4e}",
        )


# Get predicted values from the model
y_pred = model(test_points).numpy()
y_pred = y_pred.reshape(-1)

# compute the error
error = np.abs(exact_values - y_pred)

# plot a 2x2 Grid, loss plot, exact solution, predicted solution and error
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# loss plot
axs[0, 0].plot(loss_array)
axs[0, 0].set_title("Loss Plot")
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].set_yscale("log")

# exact solution
# contour plot of the exact solution
axs[0, 1].tricontourf(test_points[:, 0], test_points[:, 1], exact_values, 100)
axs[0, 1].set_title("Exact Solution")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
# add colorbar
cbar = plt.colorbar(axs[0, 1].collections[0], ax=axs[0, 1])


# predicted solution
# contour plot of the predicted solution
axs[1, 0].tricontourf(test_points[:, 0], test_points[:, 1], y_pred, 100)
axs[1, 0].set_title("Predicted Solution")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
# add colorbar
cbar = plt.colorbar(axs[1, 0].collections[0], ax=axs[1, 0])

# error plot
# contour plot of the error
axs[1, 1].tricontourf(test_points[:, 0], test_points[:, 1], error, 100)
axs[1, 1].set_title("Error")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")
# add colorbar
cbar = plt.colorbar(axs[1, 1].collections[0], ax=axs[1, 1])


plt.tight_layout()
plt.savefig("results.png")


# print error statistics
l2_error = np.sqrt(np.mean(error**2))
l1_error = np.mean(np.abs(error))
l_inf_error = np.max(np.abs(error))
rel_l2_error = l2_error / np.sqrt(np.mean(exact_values**2))
rel_l1_error = l1_error / np.mean(np.abs(exact_values))
rel_l_inf_error = l_inf_error / np.max(np.abs(exact_values))

# print the error statistics in a formatted table
error_df = pd.DataFrame(
    {
        "L2 Error": [l2_error],
        "L1 Error": [l1_error],
        "L_inf Error": [l_inf_error],
        "Relative L2 Error": [rel_l2_error],
        "Relative L1 Error": [rel_l1_error],
        "Relative L_inf Error": [rel_l_inf_error],
    }
)
print(error_df)
