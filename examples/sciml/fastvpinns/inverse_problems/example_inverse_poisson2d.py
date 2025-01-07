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
Example script for solving a 2D Inverse Poisson equation using FastVPINNs.

Author: Thivin Anandh (https://thivinanandh.github.io/)

Versions:
    - 07-Jan-2024 (Version 0.1): Initial Implementation
"""

# Common library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import time
from tqdm import tqdm

# Fastvpinns Modules
from scirex.core.sciml.geometry.geometry_2d import Geometry_2D
from scirex.core.sciml.fe.fespace2d import Fespace2D
from scirex.core.sciml.fastvpinns.data.datahandler2d import DataHandler2D

# Section: Inputs

i_mesh_type = "quadrilateral"  # "quadrilateral"
i_mesh_generation_method = "internal"  # "internal" or "external"
i_x_min = -1  # minimum x value
i_x_max = 1  # maximum x value
i_y_min = -1  # minimum y value
i_y_max = 1  # maximum y value
i_n_cells_x = 2  # Number of cells in the x direction
i_n_cells_y = 2  # Number of cells in the y direction
i_n_boundary_points = 400  # Number of points on the boundary
i_output_path = "output/poisson_2d"  # Output path

i_n_test_points_x = 100  # Number of test points in the x direction
i_n_test_points_y = 100  # Number of test points in the y direction

i_num_sensor_points = 50  # Number of sensor points

# fe Variables
i_fe_order = 10  # Order of the finite element space
i_fe_type = "legendre"
i_quad_order = 20  # 20 points in 1D, so 400 points in 2D for one cell
i_quad_type = "gauss-jacobi"

# Neural Network Variables
i_learning_rate_dict = {
    "initial_learning_rate": 0.002,  # Initial learning rate
    "use_lr_scheduler": False,  # Use learning rate scheduler
    "decay_steps": 1000,  # Decay steps
    "decay_rate": 0.96,  # Decay rate
    "staircase": True,  # Staircase Decay
}

i_dtype = tf.float32
i_activation = "tanh"
i_beta = 10  # Boundary Loss Penalty ( Adds more weight to the boundary loss)

# Epochs
i_num_epochs = 5000


## Section: Setting up boundary conditions and Inverse parameters

EPS = 0.3


def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    return val


def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    return val


def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    return val


def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10
    return val


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """

    X = x
    Y = y
    eps = EPS

    return (
        -EPS
        * (
            40.0 * X * eps * (np.tanh(X) ** 2 - 1) * np.sin(X)
            - 40.0 * X * eps * np.cos(X) * np.tanh(X)
            + 10 * eps * (4.0 * X**2 * eps - 2.0) * np.sin(X) * np.tanh(X)
            + 20 * (np.tanh(X) ** 2 - 1) * np.sin(X) * np.tanh(X)
            - 20 * (np.tanh(X) ** 2 - 1) * np.cos(X)
            - 10 * np.sin(X) * np.tanh(X)
        )
        * np.exp(-1.0 * X**2 * eps)
    )


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """

    val = np.sin(x) * np.tanh(x) * np.exp(-1.0 * EPS * (x**2)) * 10

    return val


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {
        1000: bottom_boundary,
        1001: right_boundary,
        1002: top_boundary,
        1003: left_boundary,
    }


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    # Initial Guess
    eps = EPS

    return {"eps": eps}


def get_inverse_params_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = 2

    return {"eps": eps}


def get_inverse_params_actual_dict():
    """
    This function will return a dictionary of inverse parameters
    """
    # Initial Guess
    eps = EPS

    return {"eps": eps}


## CREATE OUTPUT FOLDER
# use pathlib to create the folder,if it does not exist
folder = Path(i_output_path)
# create the folder if it does not exist
if not folder.exists():
    folder.mkdir(parents=True, exist_ok=True)


# get the boundary function dictionary from example file
bound_function_dict, bound_condition_dict = (
    get_boundary_function_dict(),
    get_bound_cond_dict(),
)

# Initiate a Geometry_2D object
domain = Geometry_2D(
    i_mesh_type,
    i_mesh_generation_method,
    i_n_test_points_x,
    i_n_test_points_y,
    i_output_path,
)

# load the mesh
cells, boundary_points = domain.generate_quad_mesh_internal(
    x_limits=[i_x_min, i_x_max],
    y_limits=[i_y_min, i_y_max],
    n_cells_x=i_n_cells_x,
    n_cells_y=i_n_cells_y,
    num_boundary_points=i_n_boundary_points,
)

# fe Space
fespace = Fespace2D(
    mesh=domain.mesh,
    cells=cells,
    boundary_points=boundary_points,
    cell_type=domain.mesh_type,
    fe_order=i_fe_order,
    fe_type=i_fe_type,
    quad_order=i_quad_order,
    quad_type=i_quad_type,
    fe_transformation_type="bilinear",
    bound_function_dict=bound_function_dict,
    bound_condition_dict=bound_condition_dict,
    forcing_function=rhs,
    output_path=i_output_path,
    generate_mesh_plot=True,
)


# instantiate data handler
datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

params_dict = {}
params_dict["n_cells"] = fespace.n_cells

from scirex.core.sciml.fastvpinns.model.model_inverse import DenseModel_Inverse
from scirex.core.sciml.fastvpinns.physics.poisson2d_inverse import (
    pde_loss_poisson_inverse,
)

params_dict = {}
params_dict["n_cells"] = fespace.n_cells

# get the input data for the PDE
train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

# get bilinear parameters
# this function will obtain the values of the bilinear parameters from the model
# and convert them into tensors of desired dtype
bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(
    get_bilinear_params_dict
)

# Obtain the sensor data
points, sensor_values = datahandler.get_sensor_data(
    exact_solution,
    num_sensor_points=i_num_sensor_points,
    mesh_type=i_mesh_generation_method,
)

### Obtain the Inverse Parameters
# Obtain the inverse parameters
inverse_params_dict = datahandler.get_inverse_params(get_inverse_params_dict)

# obtain the target inverse parameters
target_inverse_params_dict = get_inverse_params_actual_dict()

# get actual Epsilon
actual_epsilon = target_inverse_params_dict["eps"]


# obtain test points
test_points = domain.get_test_points()
print(f"Number of Test Points =  {test_points.shape[0]}")
y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

model = DenseModel_Inverse(
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict,
    params_dict=params_dict,
    loss_function=pde_loss_poisson_inverse,
    input_tensors_list=[
        datahandler.x_pde_list,
        train_dirichlet_input,
        train_dirichlet_output,
    ],
    orig_factor_matrices=[
        datahandler.shape_val_mat_list,
        datahandler.grad_x_mat_list,
        datahandler.grad_y_mat_list,
    ],
    force_function_list=datahandler.forcing_function_list,
    sensor_list=[points, sensor_values],
    inverse_params_dict=inverse_params_dict,
    tensor_dtype=i_dtype,
    activation=i_activation,
)

loss_array = []  # total loss
time_array = []  # time taken for each epoch
inverse_params_array = []  # inverse parameters
sensor_loss_array = []  # sensor loss

for epoch in tqdm(range(i_num_epochs), desc="Training Model"):
    # Train the model
    batch_start_time = time.time()
    loss = model.train_step(beta=i_beta, bilinear_params_dict=bilinear_params_dict)
    elapsed = time.time() - batch_start_time

    # print(elapsed)
    time_array.append(elapsed)

    loss_array.append(loss["loss"])
    inverse_params_array.append(loss["inverse_params"]["eps"].numpy())
    sensor_loss_array.append(loss["sensor_loss"])

    y_pred_test = model(test_points).numpy().reshape(-1)
    error = np.abs(y_exact - y_pred_test)

    l2_error = np.sqrt(np.mean(error**2))
    l1_error = np.mean(np.abs(error))
    l_inf_error = np.max(np.abs(error))

    loss_pde = float(loss["loss_pde"].numpy())
    loss_dirichlet = float(loss["loss_dirichlet"].numpy())
    total_loss = float(loss["loss"].numpy())

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}")
        print(
            f"Variational Losses   || Pde Loss: {loss_pde:.3e} Dirichlet Loss: {loss_dirichlet:.3e} Total Loss: {total_loss:.3e}"
        )
        print(
            f"Predicted Parameter  || {loss['inverse_params']['eps'].numpy():.3e}",
            f"Actual Parameter: {actual_epsilon:.3e}",
            f"Sensor Loss: {float(loss['sensor_loss'].numpy()):.3e}",
        )

# predict the values for the test points
test_points = domain.get_test_points()
print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

# Get predicted values from the model
y_pred = model(test_points).numpy()
y_pred = y_pred.reshape(-1)

# compute the error
error = np.abs(y_exact - y_pred)

# plot a 2x2 Grid, loss plot, exact solution, predicted solution and error
fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# loss plot
axs[0, 0].plot(loss_array)
axs[0, 0].set_title("Loss Plot")
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].set_yscale("log")

# exact solution
# contour plot of the exact solution
axs[0, 1].tricontourf(test_points[:, 0], test_points[:, 1], y_exact, 100)
axs[0, 1].set_title("Exact Solution")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
# plot the sensor points as black dots
axs[0, 1].scatter(points[:, 0], points[:, 1], color="black", label="Sensor Points")
# add text box saying that these are sensor points
# axs[0, 1].text(0.5, 0.5, "black dots -> Sensor Points", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
# add colorbar
cbar = plt.colorbar(axs[0, 1].collections[0], ax=axs[0, 1])
# foorce to have only 3 ticks in the colorbar
cbar.set_ticks([y_exact.min(), (y_exact.min() + y_exact.max()) / 2, y_exact.max()])


# predicted solution
# contour plot of the predicted solution
axs[1, 0].tricontourf(test_points[:, 0], test_points[:, 1], y_pred, 100)
axs[1, 0].set_title("Predicted Solution")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
# add colorbar
cbar = plt.colorbar(axs[1, 0].collections[0], ax=axs[1, 0])
# foorce to have only 3 ticks in the colorbar
cbar.set_ticks([y_pred.min(), (y_pred.min() + y_pred.max()) / 2, y_pred.max()])

# error plot
# contour plot of the error
axs[1, 1].tricontourf(test_points[:, 0], test_points[:, 1], error, 100)
axs[1, 1].set_title("Error")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")
# add colorbar
cbar = plt.colorbar(axs[1, 1].collections[0], ax=axs[1, 1])
# foorce to have only 3 ticks in the colorbar
cbar.set_ticks([error.min(), (error.min() + error.max()) / 2, error.max()])

## Add the Inverse Parameter Prediction
axs[2, 0].plot(inverse_params_array)
axs[2, 0].set_title("Inverse Parameter Prediction")
axs[2, 0].set_xlabel("Epochs")
axs[2, 0].set_ylabel("Epsilon")
axs[2, 0].axhline(y=actual_epsilon, color="r", linestyle="--", label="Actual Epsilon")
axs[2, 0].legend()

# plot the Sensor Loss
axs[2, 1].plot(sensor_loss_array)
axs[2, 1].set_title("Sensor Loss")
axs[2, 1].set_xlabel("Epochs")
axs[2, 1].set_ylabel("Loss")
# set the yscale to log
axs[2, 1].set_yscale("log")


plt.tight_layout()
plt.savefig("results.png")


# print error statistics
l2_error = np.sqrt(np.mean(error**2))
l1_error = np.mean(np.abs(error))
l_inf_error = np.max(np.abs(error))
rel_l2_error = l2_error / np.sqrt(np.mean(y_exact**2))
rel_l1_error = l1_error / np.mean(np.abs(y_exact))
rel_l_inf_error = l_inf_error / np.max(np.abs(y_exact))

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
