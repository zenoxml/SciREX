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
Example script for solving a 2D Poisson equation using FastvPINNs.

Author: Divij Ghose (https://divijghose.github.io/)

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

from scirex.core.sciml.pinns.model.model_vector_transient import DenseModel
from scirex.core.sciml.pinns.physics.maxwell import pde_loss_maxwell
from scirex.core.sciml.pinns.optimizers.lbfgs import *

i_mesh_type = "quadrilateral"  # "quadrilateral"
i_mesh_generation_method = "internal"  # "internal" or "external"


i_x_min = 0  # minimum x value
i_x_max = 1  # maximum x value
i_y_min = 0  # minimum y value
i_y_max = 1  # maximum y value

i_t_min = 0  # minimum time value
i_t_max = 2  # maximum time value

i_n_collocation_points = 3000  # Number of collocation points
i_n_boundary_points = 1000  # Number of points on the boundary
i_output_path = "output/poisson_2d"  # Output path

i_n_test_points_x = 100  # Number of test points in the x direction
i_n_test_points_y = 100  # Number of test points in the y direction
i_n_test_points_t = 100  # Number of test points in the t direction

i_activation = "tanh"  # Activation function
# Neural Network Variables
i_learning_rate_dict = {
    "initial_learning_rate": 0.001,  # Initial learning rate
    "use_lr_scheduler": False,  # Use learning rate scheduler
    "decay_steps": 5000,  # Decay steps
    "decay_rate": 0.98,  # Decay rate
    "staircase": False,  # Staircase Decay
}

i_dtype = tf.float32
i_num_epochs = 150000  # Number of epochs

# penalty parameter for boundary loss
i_beta_boundary = 0.01
i_beta_initial = 10.0


#############################################################################
## Generate the input data (Collacation points) within the domain
#############################################################################

# Generate the collocation points in the domain between x_min and x_max and y_min and y_max
input_points = lhs(3, i_n_collocation_points)

# Rescale the points to the x_min, x_max, y_min and y_max
input_points[:, 0] = input_points[:, 0] * (i_x_max - i_x_min) + i_x_min
input_points[:, 1] = input_points[:, 1] * (i_y_max - i_y_min) + i_y_min
input_points[:, 2] = input_points[:, 2] * (i_t_max - i_t_min) + i_t_min


#############################################################################
################# Generate the Initial Condition Data #######################
#############################################################################
def initial_data_Hx(x, y):
    """
    Function to generate the initial data for the Poisson 2D problem
    """
    val = 0
    return np.ones_like(x) * val


def initial_data_Hy(x, y):
    """
    Function to generate the initial data for the Poisson 2D problem
    """
    val = 0
    return np.ones_like(x) * val


def initial_data_Ez(x, y):
    """
    Function to generate the initial data for the Poisson 2D problem
    """
    val = np.sin(np.pi * x) * np.sin(np.pi * y)
    return val


#############################################################################
####################### Generate the Boundary Data ##########################
#############################################################################
def boundary_data(x, y, t):
    """
    Function to generate the boundary data for the Poisson 2D problem
    """
    val = 0
    return np.ones_like(x) * val


#############################################################################
################# Generate the Boundary Point ################################
#############################################################################


# top boundary: y = i_y_max
top_x = np.linspace(i_x_min, i_x_max, i_n_boundary_points)
top_y = np.ones(i_n_boundary_points) * i_y_max
top_t = np.linspace(i_t_min, i_t_max, i_n_boundary_points)
# concatenate the points into 3D array
top_boundary_points = np.vstack((top_x, top_y, top_t)).T
top_boundary_values = boundary_data(top_x, top_y, top_t)

# bottom boundary: y = i_y_min
bottom_x = np.linspace(i_x_min, i_x_max, i_n_boundary_points)
bottom_y = np.ones(i_n_boundary_points) * i_y_min
bottom_t = np.linspace(i_t_min, i_t_max, i_n_boundary_points)
# concatenate the points into 2D array
bottom_boundary_points = np.vstack((bottom_x, bottom_y, bottom_t)).T
bottom_boundary_values = boundary_data(bottom_x, bottom_y, bottom_t)

# left boundary: x = i_x_min
left_x = np.ones(i_n_boundary_points) * i_x_min
left_y = np.linspace(i_y_min, i_y_max, i_n_boundary_points)
left_t = np.linspace(i_t_min, i_t_max, i_n_boundary_points)
# concatenate the points into 2D array
left_boundary_points = np.vstack((left_x, left_y, left_t)).T
left_boundary_values = boundary_data(left_x, left_y, left_t)

# right boundary: x = i_x_max
right_x = np.ones(i_n_boundary_points) * i_x_max
right_y = np.linspace(i_y_min, i_y_max, i_n_boundary_points)
right_t = np.linspace(i_t_min, i_t_max, i_n_boundary_points)
# concatenate the points into 2D array
right_boundary_points = np.vstack((right_x, right_y, right_t)).T
right_boundary_values = boundary_data(right_x, right_y, right_t)

# initial condition boundary: t = i_t_min for Hx
initial_collocation_points = lhs(2, i_n_boundary_points)
initial_x_Hx = initial_collocation_points[:, 0] * (i_x_max - i_x_min) + i_x_min
initial_y_Hx = initial_collocation_points[:, 1] * (i_y_max - i_y_min) + i_y_min
initial_t_Hx = np.ones(i_n_boundary_points) * i_t_min


# concatenate the points into 2D array
initial_boundary_points_Hx = np.vstack((initial_x_Hx, initial_y_Hx, initial_t_Hx)).T
initial_boundary_values_Hx = initial_data_Hx(initial_x_Hx, initial_y_Hx)


# initial condition boundary: t = i_t_min for Hy
initial_x_Hy = initial_collocation_points[:, 0] * (i_x_max - i_x_min) + i_x_min
initial_y_Hy = initial_collocation_points[:, 1] * (i_y_max - i_y_min) + i_y_min
initial_t_Hy = np.ones(i_n_boundary_points) * i_t_min
# concatenate the points into 2D array
initial_boundary_points_Hy = np.vstack((initial_x_Hy, initial_y_Hy, initial_t_Hy)).T
initial_boundary_values_Hy = initial_data_Hy(initial_x_Hy, initial_y_Hy)


# initial condition boundary: t = i_t_min for Ez
initial_x_Ez = initial_collocation_points[:, 0] * (i_x_max - i_x_min) + i_x_min
initial_y_Ez = initial_collocation_points[:, 1] * (i_y_max - i_y_min) + i_y_min
initial_t_Ez = np.ones(i_n_boundary_points) * i_t_min
# concatenate the points into 2D array
initial_boundary_points_Ez = np.vstack((initial_x_Ez, initial_y_Ez, initial_t_Ez)).T
initial_boundary_values_Ez = initial_data_Ez(initial_x_Ez, initial_y_Ez)

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


#############################################################################
## Fill the RHS of the Poisson Equation
#############################################################################


def rhs(x, y, t):
    """
    Function to generate the right hand side of the Poisson 2D problem
    """
    val = 0

    return np.ones_like(x) * val


# For all the collocation points, compute the right hand side
rhs_values = rhs(input_points[:, 0:1], input_points[:, 1:2], input_points[:, 2:3])


#############################################################################
## Setup Test points
#############################################################################

# Generate the test points in the domain between x_min and x_max and y_min and y_max

test_points_x = np.linspace(i_x_min, i_x_max, i_n_test_points_x)
test_points_y = np.linspace(i_y_min, i_y_max, i_n_test_points_y)
test_points_t = np.linspace(i_t_min, i_t_max, i_n_test_points_t)


test_points = np.array(
    np.meshgrid(test_points_x, test_points_y, test_points_t)
).T.reshape(-1, 3)

#############################################################################
## Setup PDE Parameters Dictionary
#############################################################################
i_bilinear_params_dict = {
    "epsilon": tf.constant(1.0, dtype=i_dtype),
    "mu": tf.constant(1.0, dtype=i_dtype),
}

#############################################################################
## Define the exact solution for the Poisson 2D problem
#############################################################################


def exact_solution_Hx(x, y, t):
    """
    Function to generate the exact solution for the Poisson 2D problem
    """
    omega = np.pi * np.sqrt(2)
    val = (
        (-1.0 * np.pi / omega)
        * np.sin(np.pi * x)
        * np.cos(np.pi * y)
        * np.sin(omega * t)
    )
    return val


def exact_solution_Hy(x, y, t):
    """
    Function to generate the exact solution for the Poisson 2D problem
    """
    omega = np.pi * np.sqrt(2)
    val = (
        (1.0 * np.pi / omega)
        * np.cos(np.pi * x)
        * np.sin(np.pi * y)
        * np.sin(omega * t)
    )
    return val


def exact_solution_Ez(x, y, t):
    """
    Function to generate the exact solution for the Poisson 2D problem
    """
    omega = np.pi * np.sqrt(2)
    val = np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)
    return val


exact_values_Hx = exact_solution_Hx(
    test_points[:, 0], test_points[:, 1], test_points[:, 2]
)
exact_values_Hy = exact_solution_Hy(
    test_points[:, 0], test_points[:, 1], test_points[:, 2]
)
exact_values_Ez = exact_solution_Ez(
    test_points[:, 0], test_points[:, 1], test_points[:, 2]
)


#############################################################################
##  Convert the input data into tensors
#############################################################################

# Convert the input points, boundary points, boundary values and rhs values into tensors
input_points = tf.convert_to_tensor(input_points, dtype=tf.float32)
boundary_points = tf.convert_to_tensor(boundary_points, dtype=tf.float32)
boundary_values = tf.reshape(
    tf.convert_to_tensor(boundary_values, dtype=tf.float32), (-1, 1)
)
initial_boundary_points_Ez = tf.convert_to_tensor(
    initial_boundary_points_Ez, dtype=tf.float32
)
initial_boundary_values_Ez = tf.reshape(
    tf.convert_to_tensor(initial_boundary_values_Ez, dtype=tf.float32), (-1, 1)
)
initial_boundary_points_Hx = tf.convert_to_tensor(
    initial_boundary_points_Hx, dtype=tf.float32
)
initial_boundary_values_Hx = tf.reshape(
    tf.convert_to_tensor(initial_boundary_values_Hx, dtype=tf.float32), (-1, 1)
)
initial_boundary_points_Hy = tf.convert_to_tensor(
    initial_boundary_points_Hy, dtype=tf.float32
)
initial_boundary_values_Hy = tf.reshape(
    tf.convert_to_tensor(initial_boundary_values_Hy, dtype=tf.float32), (-1, 1)
)

rhs_values = tf.reshape(tf.convert_to_tensor(rhs_values, dtype=tf.float32), (-1, 1))
test_points = tf.convert_to_tensor(test_points, dtype=tf.float32)


## CREATE OUTPUT FOLDER
# use pathlib to create the folder,if it does not exist
folder = Path(i_output_path)
# create the folder if it does not exist
if not folder.exists():
    folder.mkdir(parents=True, exist_ok=True)


model = DenseModel(
    layer_dims=[3, 30, 50, 50, 50, 50, 30, 3],
    learning_rate_dict=i_learning_rate_dict,
    loss_function=pde_loss_maxwell,
    input_tensors_list=[
        input_points,
        boundary_points,
        boundary_values,
        initial_boundary_points_Ez,
        initial_boundary_values_Ez,
        initial_boundary_points_Hx,
        initial_boundary_values_Hx,
        initial_boundary_points_Hy,
        initial_boundary_values_Hy,
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
    if epoch <= i_num_epochs:
        loss = model.train_step(
            beta_boundary=i_beta_boundary,
            beta_initial=i_beta_initial,
            bilinear_params_dict=i_bilinear_params_dict,
        )
        loss_array.append(loss["loss"])
        if epoch % 1000 == 0:
            pde_loss = loss["loss_pde"]
            boundary_loss = loss["loss_dirichlet"]
            initial_loss = loss["loss_initial"]
            print(
                f"Epoch: {epoch}, Loss: {loss['loss'] :.4e}",
                f"PDE Loss: {pde_loss:.4e}, Boundary Loss: {boundary_loss:.4e}",
                f"Inital Loss: {initial_loss:.4e}",
            )
    else:
        results = lbfgs_minimize(
            model.trainable_variables, model.calculate_train_loss_wrapper
        )
    elapsed = time.time() - batch_start_time
    # print(elapsed)
    time_array.append(elapsed)

    if epoch % 500 == 0:
        model_sol = model(test_points)
        sol_Ez = model_sol[:, 0:1].numpy().reshape(-1)
        error_Ez = np.abs(exact_values_Ez - sol_Ez)
        l_inf_error_Ez = np.max(np.abs(error_Ez))
        l2_error_Ez = np.sqrt(np.mean(error_Ez**2))

        sol_Hx = model_sol[:, 1:2].numpy().reshape(-1)
        error_Hx = np.abs(exact_values_Hx - sol_Hx)
        l_inf_error_Hx = np.max(np.abs(error_Hx))
        l2_error_Hx = np.sqrt(np.mean(error_Hx**2))

        sol_Hy = model_sol[:, 2:3].numpy().reshape(-1)
        error_Hy = np.abs(exact_values_Hy - sol_Hy)
        l_inf_error_Hy = np.max(np.abs(error_Hy))
        l2_error_Hy = np.sqrt(np.mean(error_Hy**2))

        print(
            f"L2 Error Ez: {l2_error_Ez:.4e}, L_inf Error Ez: {l_inf_error_Ez:.4e}",
            f"L2 Error Hx: {l2_error_Hx:.4e}, L_inf Error Hx: {l_inf_error_Hx:.4e}",
            f"L2 Error Hy: {l2_error_Hy:.4e}, L_inf Error Hy: {l_inf_error_Hy:.4e}",
        )


# Get predicted values of Ez at x = 0.75 and y = 0.75 for 200 time points
test_points_t = np.linspace(i_t_min, i_t_max, 200)
test_points = np.ones((200, 3)) * np.array([0.75, 0.75, 0.0])
test_points[:, 2] = test_points_t
test_points = tf.convert_to_tensor(test_points, dtype=tf.float32)
model_sol = model(test_points)
sol_Ez = model_sol[:, 0:1].numpy().reshape(-1)
exact_values_Ez = exact_solution_Ez(
    test_points[:, 0], test_points[:, 1], test_points[:, 2]
)
error_Ez = np.abs(exact_values_Ez - sol_Ez)
l_inf_error_Ez = np.max(np.abs(error_Ez))
l2_error_Ez = np.sqrt(np.mean(error_Ez**2))
print(f"L2 Error Ez: {l2_error_Ez:.4e}, L_inf Error Ez: {l_inf_error_Ez:.4e}")
# plot the results
sol_Ez = sol_Ez.reshape(200, 1)
exact_values_Ez = exact_values_Ez.reshape(200, 1)
error_Ez = error_Ez.reshape(200, 1)
plt.figure()
plt.plot(test_points_t, sol_Ez, label="Predicted")
plt.plot(test_points_t, exact_values_Ez, label="Exact")
plt.xlabel("Time")
plt.ylabel("Ez")
plt.legend()
plt.savefig("Ez.png")
plt.close()
exit()


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
