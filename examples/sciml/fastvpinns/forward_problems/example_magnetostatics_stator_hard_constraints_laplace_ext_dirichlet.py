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
Example script for solving a 2D Poisson equation for magnetostatics
using FastvPINNs.

Author: Sai Bhargav P.

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
from scirex.core.dl.tf_backend.datautils import convert_to_tensor, reshape

from scirex.core.sciml.fastvpinns.model.model_magnetostatic_bvp_hard_constraints_laplace import (
    DenseModel,
    MagnetisationModel,
)
from scirex.core.sciml.fastvpinns.physics.magnetostatics_laplace import (
    pde_loss_magnetostatics,
)

i_mesh_type = "quadrilateral"  # "quadrilateral"
i_mesh_generation_method = "external"  # "internal" or "external"
i_mesh_file_name = "tests/support_files/Case2/test.mesh"  # should be a .mesh file
i_boundary_refinement_level = 4
i_boundary_sampling_method = "uniform"  # "uniform"
i_x_min = -1  # minimum x value
i_x_max = 1  # maximum x value
i_y_min = -1  # minimum y value
i_y_max = 1  # maximum y value
i_n_cells_x = 4  # Number of cells in the x direction
i_n_cells_y = 4  # Number of cells in the y direction
i_n_boundary_points = 400  # Number of points on the boundary
i_output_path = "output/Case2/magnetostatics_stator_inference_tanh"  # Output path
i_external_dirichlet_data = "tests/support_files/Case2/dirichlet_case2.txt"

i_n_test_points_x = 100  # Number of test points in the x direction
i_n_test_points_y = 100  # Number of test points in the y direction

# fe Variables
i_fe_order = 4  # Order of the finite element space
i_fe_type = "legendre"
i_quad_order = 3  # 10 points in 1D, so 100 points in 2D for one cell
i_quad_type = "gauss-jacobi"

# Neural Network Variables
i_learning_rate_dict = {
    "initial_learning_rate": 0.001,  # Initial learning rate
    "use_lr_scheduler": True,  # Use learning rate scheduler
    "decay_steps": 5000,  # Decay steps
    "decay_rate": 0.95,  # Decay rate
    "staircase": True,  # Staircase Decay
}

i_dtype = tf.float64
i_activation = "tanh"
i_use_adaptive = False
i_use_polynomial = False
i_polynomial_coeffs = [0.1, 0.5, 0.3]
i_beta = 1e8  # Boundary Loss Penalty ( Adds more weight to the boundary loss)

# Epochs
i_num_epochs = 100000

# Parameters to test external data
i_test_external = True

bh_data = np.loadtxt(
    "tests/support_files/stator_bh_curve.csv", delimiter=",", skiprows=1
)
b = bh_data[:, 1]
h = bh_data[:, 0]
b = reshape(b, (-1, 1))
h = reshape(h, (-1, 1))
mu0 = 1.0
b = convert_to_tensor(b, dtype=i_dtype)
h = convert_to_tensor(h, dtype=i_dtype)

# normalize b and h
mean_b = tf.reduce_mean(b)
std_b = tf.math.reduce_std(b)
mean_h = tf.reduce_mean(h)
std_h = tf.math.reduce_std(h)
b_norm = (b - mean_b) / std_b
h_norm = (h - mean_h) / std_h

magnetisation = MagnetisationModel(b, h, dtype=i_dtype)

# train the bh network
for epoch in range(50000):
    loss = magnetisation.train_step()
    if (epoch + 1) % 100 == 0:
        training_loss = loss["loss"].numpy()
        print(f"Epoch: {epoch+1}, Loss: {training_loss}")
        if training_loss < 1e-6:
            print("Converged")
            break


## Setting up boundary conditions
def inner_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    r = 0.02
    return np.ones_like(x) * r


def outer_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    r = 0.0
    return np.ones_like(x) * r


def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: outer_boundary, 1001: inner_boundary}


def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet"}


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    f_temp = 0.0

    return np.ones_like(x) * f_temp


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    r = 0.0

    return np.ones_like(x) * r


def stator_max_radius():
    """
    This function will return the maximum radius of the stator
    """
    return 46.25


def stator_diameter():
    """
    This function will return the diameter of the stator
    """
    return 92.5


def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    mu0 = 0.0
    return {"mu0": mu0}


def get_dirichlet_and_test_data_external(filename, dtype):
    """
    Function to read external dirichlet data on the inner boundary

    """
    d = np.loadtxt(filename)

    # x, y inner boundary coords
    x = d[:, 0]
    y = d[:, 1]

    # A boundary cond
    a = d[:, 2]

    x_bc = x.flatten()
    y_bc = y.flatten()

    a_bc = a.flatten()

    # input X bc
    input_ext_dirichlet = np.hstack((x_bc[:, None], y_bc[:, None]))

    # input A bc
    output_ext_dirichlet = a_bc[:, None]

    input_ext_dirichlet = tf.constant(input_ext_dirichlet, dtype=dtype)
    output_ext_dirichlet = tf.constant(output_ext_dirichlet, dtype=dtype)

    return input_ext_dirichlet, output_ext_dirichlet


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


# # load the mesh
cells, boundary_points = domain.read_mesh(
    i_mesh_file_name,
    i_boundary_refinement_level,
    i_boundary_sampling_method,
    refinement_level=1,
)
# save cells as pickle file
# print(boundary_points.keys())
# cells = domain.load_cell_points(Path(i_output_path) / "cells.pkl")
# boundary_points = domain.load_boundary_points(Path(i_output_path) / "boundary_points.pkl")

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
    generate_mesh_plot=False,
)


# instantiate data handler
datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

params_dict = {}
params_dict["n_cells"] = fespace.n_cells


# get the input data for the PDE
# train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

# get the dirichlet input data from external file
train_dirichlet_input, train_dirichlet_output = get_dirichlet_and_test_data_external(
    i_external_dirichlet_data, i_dtype
)

print("train_dirichlet_input:\n", train_dirichlet_input.shape)


print("train_dirichlet_output:\n", train_dirichlet_output.shape)

# get bilinear parameters
# this function will obtain the values of the bilinear parameters from the model
# and convert them into tensors of desired dtype
bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(
    get_bilinear_params_dict
)

model = DenseModel(
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict,
    params_dict=params_dict,
    loss_function=pde_loss_magnetostatics,
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
    tensor_dtype=i_dtype,
    activation=i_activation,
    use_adaptive=False,
    use_polynomial=i_use_polynomial,
    polynomial_coeffs=i_polynomial_coeffs,
    trained_magnetisation_model=magnetisation,
)

loss_array = []  # total loss
residual_loss_array = []  # residual loss
dirichlet_loss_array = []  # Dirichlet loss
time_array = []  # time taken for each epoch

# Assuming 'folder' is already defined and concatenated with 'model'
output_folder = folder / "results"

# Create the output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# predict the values for the test points
test_points = domain.get_test_points()
print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

for epoch in tqdm(range(i_num_epochs)):
    # Train the model
    batch_start_time = time.time()
    loss = model.train_step(beta=i_beta, bilinear_params_dict=bilinear_params_dict)
    elapsed = time.time() - batch_start_time

    # print(elapsed)
    time_array.append(elapsed)

    loss_array.append(loss["loss"])
    residual_loss_array.append(loss["loss_pde"])
    dirichlet_loss_array.append(loss["loss_dirichlet"])

    loss_pde = float(loss["loss_pde"].numpy())
    loss_dirichlet = float(loss["loss_dirichlet"].numpy())
    total_loss = float(loss["loss"].numpy())

    if (epoch + 1) % 10000 == 0:
        y_test_pred = model(test_points).numpy().reshape(-1)

        error = y_test_pred - y_exact
        l2_error = np.sqrt(np.mean(error**2))
        l1_error = np.mean(np.abs(error))
        l_inf_error = np.max(np.abs(error))
        print(
            f"loss: {total_loss:.3e}, l2 Error: {l2_error}. l1 Error: {l1_error} linf : {l_inf_error}"
        )
        print(
            f"Variational Losses || Pde : {loss_pde:.3e} Dirichlet : {loss_dirichlet:.3e} Total : {total_loss:.3e}"
        )
        solution_array = np.c_[y_test_pred, y_exact, np.abs(y_exact - y_test_pred)]
        domain.write_vtk(
            solution_array,
            output_path=i_output_path,
            filename=f"prediction_{epoch+1}.vtk",
            data_names=["Sol", "Exact", "Error"],
        )

        b_pred = model.inference(test_points)
        bx = b_pred["Bx"].numpy().reshape(-1)
        by = b_pred["By"].numpy().reshape(-1)
        mag_b = b_pred["B"].numpy().reshape(-1)
        b_solution_array = np.c_[bx, by, mag_b]
        domain.write_vtk(
            b_solution_array,
            output_path=i_output_path,
            filename=f"b_field_{epoch+1}.vtk",
            data_names=["Bx", "By", "B"],
        )

        plt.figure(figsize=(6.4, 4.8), dpi=300)
        plt.plot(loss_array, label="Total Loss")
        plt.plot(dirichlet_loss_array, label="Dirichlet Loss")
        plt.plot(residual_loss_array, label="Residual Loss")
        plt.title("Loss Components Plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(output_folder / "loss_components_plot.png"))
        plt.close()  # Close the figure to free memory

# Get predicted values from the model
y_pred = model(test_points).numpy()
y_pred = y_pred.reshape(-1)

if i_test_external:
    test_points_solution = np.loadtxt(
        "/home/saibhargav/Projects/scirex/SciREX/tests/support_files/Case2/Case2_inference.txt"
    )
    test_points_external = np.hstack(
        (test_points_solution[:, 0][:, None], test_points_solution[:, 1][:, None])
    )

    # predicted solution
    y_test_pred = model(test_points_external).numpy().reshape(-1)

    # exact solution
    y_exact_external = test_points_solution[:, 2].reshape(-1)

    error = y_test_pred - y_exact_external
    l2_error = np.sqrt(np.mean(error**2))
    l1_error = np.mean(np.abs(error))
    l_inf_error = np.max(np.abs(error))

    print(
        f"Inference Metrices A:\n l2 Error: {l2_error}. l1 Error: {l1_error} linf : {l_inf_error}"
    )

    # solution_array = np.c_[y_test_pred, y_exact_external, np.abs(y_exact_external - y_test_pred)]
    # domain.write_vtk(
    #     solution_array,
    #     output_path=i_output_path,
    #     filename=f"prediction_{epoch+1}.vtk",
    #     data_names=["Sol", "Exact", "Abs_Error"],
    # )

    b_pred = model.inference(test_points_external)

    # b_exact = test_points_solution[:, 3].reshape(-1)

    bx = b_pred["Bx"].numpy().reshape(-1)
    by = b_pred["By"].numpy().reshape(-1)
    mag_b = b_pred["B"].numpy().reshape(-1)

    # b_error = mag_b - b_exact
    # b_l2_error = np.sqrt(np.mean(b_error**2))
    # b_l1_error = np.mean(np.abs(b_error))
    # b_l_inf_error = np.max(np.abs(b_error))

    # print(
    #     f"Inference Metrices B:\n l2 Error: {b_l2_error}. l1 Error: {b_l1_error} linf : {b_l_inf_error}"
    # )

    solution_array = np.c_[
        y_test_pred,
        y_exact_external,
        np.abs(y_exact_external - y_test_pred),
        bx,
        by,
        mag_b,
    ]  # , b_exact, np.abs(b_exact - mag_b)]

    domain.write_vtk(
        solution_array,
        output_path=i_output_path,
        filename=f"prediction_external.vtk",
        data_names=[
            "Sol_A",
            "Exact_A",
            "Abs_Error_A",
            "Bx",
            "By",
            "B",
        ],  # , "B_exact", "B_error"],
    )

    # write the inference outputs to a txt file

    np.savetxt(output_folder / "inference_results.txt", solution_array)

    print("Saved Inference results!")


# compute the error
error = np.abs(y_exact - y_pred)
model.save_weights(str(Path(i_output_path) / "model_weights"))
solution_array = np.c_[y_pred, y_exact, np.abs(y_exact - y_pred)]
domain.write_vtk(
    solution_array,
    output_path=i_output_path,
    filename=f"prediction_final.vtk",
    data_names=["Sol", "Exact", "Error"],
)

b_pred = model.inference(test_points)
bx = b_pred["Bx"].numpy().reshape(-1)
by = b_pred["By"].numpy().reshape(-1)
mag_b = b_pred["B"].numpy().reshape(-1)
b_solution_array = np.c_[bx, by, mag_b]
domain.write_vtk(
    b_solution_array,
    output_path=i_output_path,
    filename=f"b_field.vtk",
    data_names=["Bx", "By", "B"],
)

## Figure Plots
# 1. Total Loss Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array, label="Total Loss")
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(str(output_folder / "loss_plot.png"))
plt.close()  # Close the figure to free memory

# 2. Loss Components
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array, label="Total Loss")
plt.plot(dirichlet_loss_array, label="Dirichlet Loss")
plt.plot(residual_loss_array, label="Residual Loss")
plt.title("Loss Components Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(str(output_folder / "loss_components_plot.png"))
plt.close()  # Close the figure to free memory

# 3. Exact Solution Contour Plot
# plt.figure(figsize=(6.4, 4.8), dpi=300)
# contour_exact = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_exact, 100)
# plt.title("Exact Solution")
# plt.xlabel("x")
# plt.ylabel("y")
# cbar = plt.colorbar(contour_exact)
# plt.tight_layout()
# plt.savefig(str(output_folder / "exact_solution.png"))
# plt.close()

# # 4. Predicted Solution Contour Plot
# plt.figure(figsize=(6.4, 4.8), dpi=300)
# contour_pred = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_pred, 100)
# plt.title("Predicted Solution")
# plt.xlabel("x")
# plt.ylabel("y")
# cbar = plt.colorbar(contour_pred)
# plt.tight_layout()
# plt.savefig(str(output_folder / "predicted_solution.png"))
# plt.close()

# # 5. Error Contour Plot
# plt.figure(figsize=(6.4, 4.8), dpi=300)
# contour_error = plt.tricontourf(test_points[:, 0], test_points[:, 1], error, 100)
# plt.title("Error")
# plt.xlabel("x")
# plt.ylabel("y")
# cbar = plt.colorbar(contour_error)
# plt.tight_layout()
# plt.savefig(str(output_folder / "error_plot.png"))
# plt.close()


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


# Create the output folder with subfolder 'model'
output_folder = folder / "model"
output_folder.mkdir(
    parents=True, exist_ok=True
)  # Create the directory if it doesn't exist

# Full path to save weights with a proper filename (e.g., 'model_weights.h5')
weights_file_path = output_folder / "model_magnetostatics_stator.h5"

# save the model weights to the folder
model.save_weights(str(weights_file_path))  # Save the model in the SavedModel
