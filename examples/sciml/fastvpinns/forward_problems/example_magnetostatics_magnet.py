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
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
import tensorflow as tf
import time
from tqdm import tqdm
import sys
import os

# Fastvpinns Modules
from scirex.core.sciml.geometry.geometry_2d import Geometry_2D
from scirex.core.sciml.fe.fespace2d import Fespace2D
from scirex.core.sciml.fastvpinns.data.datahandler2d import DataHandler2D
from scirex.core.dl.tf_backend.datautils import convert_to_tensor, reshape

from scirex.core.sciml.fastvpinns.model.model_magnetostatics_magnet import (
    DenseModel,
    MagnetisationModel,
)
from scirex.core.sciml.fastvpinns.physics.magnetostatics import (
    pde_loss_magnetostatics,
)
from scirex.core.sciml.fastvpinns.physics.magnetostatics_magnet import (
    pde_loss_magnetostatics_magnet,
)

i_mesh_type = "quadrilateral"  # "quadrilateral"
i_mesh_generation_method = "external"  # "internal" or "external"

i_mesh_file_name_air = (
    "tests/support_files/Case_magnet/Airgap.mesh"  # should be a .mesh file
)
i_mesh_file_name_stator = "tests/support_files/Case_magnet/Stator.mesh"

i_boundary_refinement_level = 4
i_boundary_sampling_method = "uniform"  # "uniform"
i_x_min = -1  # minimum x value
i_x_max = 1  # maximum x value
i_y_min = -1  # minimum y value
i_y_max = 1  # maximum y value
i_n_cells_x = 4  # Number of cells in the x direction
i_n_cells_y = 4  # Number of cells in the y direction
i_n_boundary_points = 400  # Number of points on the boundary
# i_output_path = "output/Case1/magnetostatics_stator_w_permeab_soft_polyorder2"  # Output path

i_n_test_points_x = 100  # Number of test points in the x direction
i_n_test_points_y = 100  # Number of test points in the y direction

# fe Variables
i_fe_order = 4  # Order of the finite element space
i_fe_type = "legendre"
i_quad_order = 4  # 4 points in 1D, so 16 points in 2D for one cell
i_quad_type = "gauss-jacobi"

# Neural Network Variables
i_learning_rate_dict_airgap = {
    "initial_learning_rate": 0.001,  # Initial learning rate
    "use_lr_scheduler": True,  # Use learning rate scheduler
    "decay_steps": 10000,  # Decay steps
    "decay_rate": 0.9,  # Decay rate
    "staircase": True,  # Staircase Decay
}

i_learning_rate_dict_stator = {
    "initial_learning_rate": 0.001,  # Initial learning rate
    "use_lr_scheduler": True,  # Use learning rate scheduler
    "decay_steps": 5000,  # Decay steps
    "decay_rate": 0.9,  # Decay rate
    "staircase": True,  # Staircase Decay
}

i_dtype = tf.float64
i_activation = "tanh"
i_use_adaptive = False
i_use_polynomial = True
i_polynomial_coeffs = [0.1, 0.5, 0.3, 0.25, 0.2, 0.15]

i_alpha_stator = 1e0
i_beta_stator = 1e0  # Boundary Loss Penalty ( Adds more weight to the boundary loss)
i_gamma_stator = 1e0

i_alpha_air = 1e0
i_beta_air = 1e6
i_gamma_air = 1e0

# Script for hyperparameter search
# Check if an argument is passed
# if len(sys.argv) > 1:
#     i_beta = float(sys.argv[1])
#     i_alpha = float(sys.argv[2])

print(f"--------------------------------------------------------------------")
print(f"Stator")
print(
    f"Experiment running with beta value= {i_beta_stator}, alpha value  {i_alpha_stator}"
)
print(f"Air")
print(f"Experiment running with beta value= {i_beta_air}, alpha value  {i_alpha_air}")
print(f"--------------------------------------------------------------------")

# i_output_path = f"output/Case_magnet/Hyperparameter/Magnetostatics_poly_{i_alpha}_beta_{i_beta}"  # Output path

i_output_path = f"output/Case_magnet/Perm_check/perm_v4_poly"  # Output path
# i_external_dirichlet_data = "tests/support_files/Case_magnet/20250415_magnet_edge_Az.txt"

i_external_dirichlet_data = (
    "tests/support_files/Case_magnet/magnet_external_dirichlet.txt"
)
i_external_interface_data = "tests/support_files/Case_magnet/interface_points.txt"

# i_use_adaptive_loss_weights= True

# Epochs
i_num_epochs = 100000

# Parameters to test external data
i_test_external = True
i_test_external_path_stator = "/home/saibhargav/Projects/SciREX/tests/support_files/Case_magnet/Case2_inference_v2.txt"
i_test_external_path_airgap = "/home/saibhargav/Projects/SciREX/tests/support_files/Case_magnet/Case2_inference_airgap_v2.txt"
i_test_external_path_magB_stator = "/home/saibhargav/Projects/SciREX/tests/support_files/Case_magnet/Ref_MagB_stator.txt"
i_test_external_path__magB_airgap = "/home/saibhargav/Projects/SciREX/tests/support_files/Case_magnet/Ref_MagB_airgap.txt"


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
        if training_loss < 1e-8:
            print("Converged")
            break


# b_test = np.linspace(0, 3.2, 100)
# b_test_norm = (b_test - mean_b) / std_b
# b_test_norm = convert_to_tensor(reshape(b_test_norm, (-1, 1)), dtype=i_dtype)
# h_test = magnetisation(b_test_norm)
# h_test = h_test * std_h + mean_h
# plt.plot(h_test.numpy(),b_test, "k--", label="Predicted")
# plt.plot(h, b, "o", label="Data")
# plt.xlabel("H")
# plt.ylabel("B")
# plt.legend()
# plt.savefig("h_b_curve_test.png")

# exit(0)


## Setting up boundary conditions for stator


def inner_boundary_stator(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    r = 0.0
    return np.ones_like(x) * r


def outer_boundary_stator(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    r = 0.0
    return np.ones_like(x) * r


def get_boundary_function_dict_stator():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: outer_boundary_stator, 1001: inner_boundary_stator}


def get_bound_cond_dict_stator():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet"}


def rhs_stator(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    f_temp = 0.0

    return np.ones_like(x) * f_temp


def exact_solution_stator(x, y):
    """
    This function will return the exact solution at a given point
    """
    r = 0.0

    return np.ones_like(x) * r


def get_bilinear_params_dict_stator():
    """
    This function will return a dictionary of bilinear parameters
    """
    mu0 = 4.0 * np.pi * 1e-7
    return {"mu0": mu0}


## Setting up boundary conditions for Airgap


def inner_boundary_air(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    r = 0.02
    return np.ones_like(x) * r


def outer_boundary_air(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    r = 0.0
    return np.ones_like(x) * r


def get_boundary_function_dict_air():
    """
    This function will return a dictionary of boundary functions
    """
    return {1001: outer_boundary_air, 1002: inner_boundary_air}


def get_bound_cond_dict_air():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1001: "dirichlet", 1002: "dirichlet"}


def rhs_air(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    f_temp = 0.0

    return np.ones_like(x) * f_temp


def exact_solution_air(x, y):
    """
    This function will return the exact solution at a given point
    """
    r = 0.0

    return np.ones_like(x) * r


def get_bilinear_params_dict_air():
    """
    This function will return a dictionary of bilinear parameters
    """
    mu0 = 1.0  # / (4.0 * np.pi * 1e-7)
    return {"mu0": mu0}


def get_dirichlet_and_test_data_external(filename, dtype):
    """
    Function to read external dirichlet data on the inner boundary of the magnet

    """
    d = np.loadtxt(filename, skiprows=1)

    # x, y inner boundary coords
    x = d[:, 0]
    y = d[:, 1]

    # A boundary cond
    a = d[:, 3]

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


def get_interface_data(filename, dtype):
    """
    Function to read interface coords

    """
    d = np.loadtxt(filename, skiprows=1)

    # x, y inner boundary coords
    x = d[:, 0]
    y = d[:, 1]

    x_int = x.flatten()
    y_int = y.flatten()

    # input X bc
    input_interface = np.hstack((x_int[:, None], y_int[:, None]))

    input_interface = tf.constant(input_interface, dtype=dtype)

    return input_interface


## CREATE OUTPUT FOLDER
# use pathlib to create the folder,if it does not exist
folder = Path(i_output_path)
# create the folder if it does not exist
if not folder.exists():
    folder.mkdir(parents=True, exist_ok=True)


# get the boundary function dictionary for stator
bound_function_dict_stator, bound_condition_dict_stator = (
    get_boundary_function_dict_stator(),
    get_bound_cond_dict_stator(),
)

# Initiate a Geometry_2D object
domain_air = Geometry_2D(
    i_mesh_type,
    i_mesh_generation_method,
    i_n_test_points_x,
    i_n_test_points_y,
    i_output_path,
    vtk_filename="airgap.vtk",
)


# # load the mesh
cells_air, boundary_points_air = domain_air.read_mesh(
    i_mesh_file_name_air,
    i_boundary_refinement_level,
    i_boundary_sampling_method,
    refinement_level=1,
)

# Initiate a Geometry_2D object
domain_stator = Geometry_2D(
    i_mesh_type,
    i_mesh_generation_method,
    i_n_test_points_x,
    i_n_test_points_y,
    i_output_path,
    vtk_filename="stator.vtk",
)

# # load the mesh
cells_stator, boundary_points_stator = domain_stator.read_mesh(
    i_mesh_file_name_stator,
    i_boundary_refinement_level,
    i_boundary_sampling_method,
    refinement_level=1,
)

# get the boundary function dictionary for air
bound_function_dict_air, bound_condition_dict_air = (
    get_boundary_function_dict_air(),
    get_bound_cond_dict_air(),
)

# save cells as pickle file
# print(boundary_points_stator.keys())
# print("boundary_points_stator[1001]:\n", boundary_points_stator[1001])
# plt.scatter(boundary_points_stator[1001][:, 0], boundary_points_stator[1001][:, 1])
# plt.savefig("tests/support_files/Case_magnet/internal_boundary.png")
# plt.close()
# np.savetxt("tests/support_files/Case_magnet/internal_boundary_stator.txt", boundary_points_stator[1001])

# print("boundary_points_air[1000]:\n", boundary_points[1000])
# plt.scatter(boundary_points[1000][:, 0], boundary_points[1000][:, 1])
# plt.savefig("tests/support_files/external_boundary.png")
# plt.close()
# np.savetxt("tests/support_files/Case2/external_boundary_v2.txt", boundary_points[1000])
# exit(0)
# cells = domain.load_cell_points(Path(i_output_path) / "cells.pkl")
# boundary_points = domain.load_boundary_points(Path(i_output_path) / "boundary_points.pkl")

# fe Space for air
fespace_air = Fespace2D(
    mesh=domain_air.mesh,
    cells=cells_air,
    boundary_points=boundary_points_air,
    cell_type=domain_air.mesh_type,
    fe_order=i_fe_order,
    fe_type=i_fe_type,
    quad_order=i_quad_order,
    quad_type=i_quad_type,
    fe_transformation_type="bilinear",
    bound_function_dict=bound_function_dict_air,
    bound_condition_dict=bound_condition_dict_air,
    forcing_function=rhs_air,
    output_path=i_output_path,
    generate_mesh_plot=False,
)

# fe Space for stator
fespace_stator = Fespace2D(
    mesh=domain_stator.mesh,
    cells=cells_stator,
    boundary_points=boundary_points_stator,
    cell_type=domain_stator.mesh_type,
    fe_order=i_fe_order,
    fe_type=i_fe_type,
    quad_order=i_quad_order,
    quad_type=i_quad_type,
    fe_transformation_type="bilinear",
    bound_function_dict=bound_function_dict_stator,
    bound_condition_dict=bound_condition_dict_stator,
    forcing_function=rhs_stator,
    output_path=i_output_path,
    generate_mesh_plot=False,
)


# instantiate data handler

datahandler_air = DataHandler2D(fespace_air, domain_air, dtype=i_dtype)

datahandler_stator = DataHandler2D(fespace_stator, domain_stator, dtype=i_dtype)

params_dict_air = {}
params_dict_air["n_cells"] = fespace_air.n_cells

params_dict_stator = {}
params_dict_stator["n_cells"] = fespace_stator.n_cells

# get the input data for the airgap
train_dirichlet_input_air, train_dirichlet_output_air = (
    get_dirichlet_and_test_data_external(i_external_dirichlet_data, i_dtype)
)

# get the input data for the stator
train_dirichlet_input_stator, train_dirichlet_output_stator = (
    datahandler_stator.get_dirichlet_input()
)


# print(train_dirichlet_input_stator, "shape:\n", train_dirichlet_input_stator.shape)
# plt.scatter(train_dirichlet_input_stator[:5056, 0], train_dirichlet_input_stator[:5056, 1])

# print(train_dirichlet_input, "break\n",train_dirichlet_output)

# get bilinear parameters
# this function will obtain the values of the bilinear parameters from the model
# and convert them into tensors of desired dtype

get_bilinear_params_dict_air = datahandler_air.get_bilinear_params_dict_as_tensors(
    get_bilinear_params_dict_air
)

get_bilinear_params_dict_stator = (
    datahandler_stator.get_bilinear_params_dict_as_tensors(
        get_bilinear_params_dict_stator
    )
)

train_interface_input = get_interface_data(i_external_interface_data, i_dtype)

model_air = DenseModel(
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict_airgap,
    params_dict=params_dict_air,
    loss_function=pde_loss_magnetostatics_magnet,
    input_tensors_list=[
        datahandler_air.x_pde_list,
        train_dirichlet_input_air,
        train_dirichlet_output_air,
        train_interface_input,
    ],
    orig_factor_matrices=[
        datahandler_air.shape_val_mat_list,
        datahandler_air.grad_x_mat_list,
        datahandler_air.grad_y_mat_list,
    ],
    force_function_list=datahandler_air.forcing_function_list,
    tensor_dtype=i_dtype,
    activation=i_activation,
    use_adaptive=i_use_adaptive,
    use_polynomial=i_use_polynomial,
    polynomial_coeffs=i_polynomial_coeffs,
    # use_adaptive_loss_weights=i_use_adaptive_loss_weights
)

model_stator = DenseModel(
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict_stator,
    params_dict=params_dict_stator,
    loss_function=pde_loss_magnetostatics,
    input_tensors_list=[
        datahandler_stator.x_pde_list,
        train_dirichlet_input_stator[:5056, :],
        train_dirichlet_output_stator[:5056, :],
        train_interface_input,
    ],
    orig_factor_matrices=[
        datahandler_stator.shape_val_mat_list,
        datahandler_stator.grad_x_mat_list,
        datahandler_stator.grad_y_mat_list,
    ],
    force_function_list=datahandler_stator.forcing_function_list,
    tensor_dtype=i_dtype,
    activation=i_activation,
    use_adaptive=i_use_adaptive,
    use_polynomial=i_use_polynomial,
    polynomial_coeffs=i_polynomial_coeffs,
    trained_magnetisation_model=magnetisation,
    # use_adaptive_loss_weights=i_use_adaptive_loss_weights
)

# air loss components
loss_array_air = []  # total loss
residual_loss_array_air = []  # residual loss
dirichlet_loss_array_air = []  # Dirichlet loss
interface_loss_array_air = []  # Interface loss
time_array_air = []  # time taken for each epoch

# Stator loss components
loss_array_stator = []  # total loss
residual_loss_array_stator = []  # residual loss
dirichlet_loss_array_stator = []  # Dirichlet loss
interface_loss_array_stator = []  # Interface loss
time_array_stator = []  # time taken for each epoch

output_folder = folder / "results"

# Create the output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# predict the values for the test points for stator

test_points_air = domain_air.get_test_points()
print(f"[bold]Number of Test Points Air= [/bold] {test_points_air.shape[0]}")
y_exact_air = exact_solution_air(test_points_air[:, 0], test_points_air[:, 1])

test_points_stator = domain_stator.get_test_points()
print(f"[bold]Number of Test Points Stator= [/bold] {test_points_stator.shape[0]}")
y_exact_stator = exact_solution_stator(
    test_points_stator[:, 0], test_points_stator[:, 1]
)

# print("Initial coefficients:\n", model.get_polynomial_coefficients())

for epoch in tqdm(range(i_num_epochs)):

    # Train the model for air zone
    batch_start_time = time.time()
    loss_air = model_air.train_step(
        # epoch=epoch,
        alpha=i_alpha_air,
        beta=i_beta_air,
        gamma=i_gamma_air,
        bilinear_params_dict=get_bilinear_params_dict_air,
        trained_conjugate_model=model_stator,
    )

    # print(f"Current weights Air: PDE={loss_air['pde_weight']}, Boundary={loss_air['boundary_weight']}")

    elapsed_air = time.time() - batch_start_time

    # Air zone
    time_array_air.append(elapsed_air)

    loss_array_air.append(loss_air["loss"])
    residual_loss_array_air.append(loss_air["loss_pde"])
    dirichlet_loss_array_air.append(loss_air["loss_dirichlet"])
    interface_loss_array_air.append(loss_air["loss_interface"])

    loss_air_pde = float(loss_air["loss_pde"].numpy())
    loss_air_dirichlet = float(loss_air["loss_dirichlet"].numpy())
    loss_air_interface = float(loss_air["loss_interface"].numpy())
    total_loss_air = float(loss_air["loss"].numpy())

    # Train the model for stator zone
    batch_start_time = time.time()
    loss_stator = model_stator.train_step(
        # epoch=epoch,
        alpha=i_alpha_stator,
        beta=i_beta_stator,
        gamma=i_gamma_stator,
        bilinear_params_dict=get_bilinear_params_dict_stator,
        trained_conjugate_model=model_air,
    )
    elapsed_stator = time.time() - batch_start_time

    # print(f"Current weights Stator: PDE={loss_stator['pde_weight']}, Boundary={loss_stator['boundary_weight']}")

    # print(elapsed)
    time_array_stator.append(elapsed_stator)

    loss_array_stator.append(loss_stator["loss"])
    residual_loss_array_stator.append(loss_stator["loss_pde"])
    dirichlet_loss_array_stator.append(loss_stator["loss_dirichlet"])
    interface_loss_array_stator.append(loss_stator["loss_interface"])

    loss_stator_pde = float(loss_stator["loss_pde"].numpy())
    loss_stator_dirichlet = float(loss_stator["loss_dirichlet"].numpy())
    loss_stator_interface = float(loss_stator["loss_interface"].numpy())
    total_loss_stator = float(loss_stator["loss"].numpy())

    if (epoch + 1) % 5000 == 0:

        # Airgap

        y_test_pred_air = model_air(test_points_air)[:, 0].numpy().reshape(-1)

        error_air = y_test_pred_air - y_exact_air
        l2_error_air = np.sqrt(np.mean(error_air**2))
        l1_error_air = np.mean(np.abs(error_air))
        l_inf_error_air = np.max(np.abs(error_air))
        print(
            f"Airgap"
            f"loss: {total_loss_air}, l2 Error: {l2_error_air}. l1 Error: {l1_error_air} linf : {l_inf_error_air}"
        )
        print(
            f"Variational Losses || Pde : {loss_air_pde:.3e} Dirichlet : {loss_air_dirichlet:.3e} Interface : {loss_air_interface:.3e} Total : {total_loss_air:.3e}"
        )

        solution_array_air = np.c_[
            y_test_pred_air, y_exact_air, np.abs(y_exact_air - y_test_pred_air)
        ]
        domain_air.write_vtk(
            solution_array_air,
            output_path=i_output_path,
            filename=f"prediction_airgap_{epoch+1}.vtk",
            data_names=["Sol", "Exact", "Error"],
        )

        b_pred_air = model_air.inference(test_points_air)
        bx_air = b_pred_air["Bx"].numpy().reshape(-1)
        by_air = b_pred_air["By"].numpy().reshape(-1)
        mag_b_air = b_pred_air["B"].numpy().reshape(-1)
        b_solution_array_air = np.c_[bx_air, by_air, mag_b_air]
        domain_air.write_vtk(
            b_solution_array_air,
            output_path=i_output_path,
            filename=f"b_field_airgap_{epoch+1}.vtk",
            data_names=["Bx", "By", "B"],
        )

        plt.figure(figsize=(6.4, 4.8), dpi=300)
        plt.plot(loss_array_air, label="Total Loss")
        plt.plot(dirichlet_loss_array_air, label="Dirichlet Loss")
        plt.plot(residual_loss_array_air, label="Residual Loss")
        plt.plot(interface_loss_array_air, label="Interface Loss")
        plt.title("Loss Components Plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(output_folder / "loss_components_air_plot.png"))
        plt.close()  # Close the figure to free memory

        # Stator
        y_test_pred_stator = model_stator(test_points_stator)[:, 0].numpy().reshape(-1)

        error_stator = y_test_pred_stator - y_exact_stator
        l2_error_stator = np.sqrt(np.mean(error_stator**2))
        l1_error_stator = np.mean(np.abs(error_stator))
        l_inf_error_stator = np.max(np.abs(error_stator))
        print(
            f"Stator"
            f"loss: {total_loss_stator}, l2 Error: {l2_error_stator}. l1 Error: {l1_error_stator} linf : {l_inf_error_stator}"
        )
        print(
            f"Variational Losses || Pde : {loss_stator_pde:.3e} Dirichlet : {loss_stator_dirichlet:.3e} Interface : {loss_stator_interface:.3e} Total : {total_loss_stator:.3e}"
        )

        solution_array_stator = np.c_[
            y_test_pred_stator,
            y_exact_stator,
            np.abs(y_exact_stator - y_test_pred_stator),
        ]
        domain_stator.write_vtk(
            solution_array_stator,
            output_path=i_output_path,
            filename=f"prediction_stator_{epoch+1}.vtk",
            data_names=["Sol", "Exact", "Error"],
        )

        b_pred_stator = model_stator.inference(test_points_stator)
        bx_stator = b_pred_stator["Bx"].numpy().reshape(-1)
        by_stator = b_pred_stator["By"].numpy().reshape(-1)
        mag_b_stator = b_pred_stator["B"].numpy().reshape(-1)
        b_solution_array_stator = np.c_[bx_stator, by_stator, mag_b_stator]
        domain_stator.write_vtk(
            b_solution_array_stator,
            output_path=i_output_path,
            filename=f"b_field_stator_{epoch+1}.vtk",
            data_names=["Bx", "By", "B"],
        )

        plt.figure(figsize=(6.4, 4.8), dpi=300)
        plt.plot(loss_array_stator, label="Total Loss")
        plt.plot(dirichlet_loss_array_stator, label="Dirichlet Loss")
        plt.plot(residual_loss_array_stator, label="Residual Loss")
        plt.plot(interface_loss_array_stator, label="Interface Loss")
        plt.title("Loss Components Plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(output_folder / "loss_components_stator_plot.png"))
        plt.close()  # Close the figure to free memory

# Get predicted values from the model for air
y_test_pred_air = model_air(test_points_air)[:, 0].numpy()
y_pred_air = y_test_pred_air.reshape(-1)


# Get predicted values from the model for stator
y_test_pred_stator = model_stator(test_points_stator)[:, 0].numpy()
y_pred_stator = y_test_pred_stator.reshape(-1)


# def plot_inference(predicted_path, fig_path):
#     """
#     Plotting code for external inference data
#     """
#     inference_data = np.loadtxt(i_test_external_path)

#     # Extract x and y coordinates
#     x, y = inference_data[:, 0], inference_data[:, 1]

#     prediction_data = np.loadtxt(predicted_path)

#     A_pred = prediction_data[:, 0]
#     A_exact = prediction_data[:, 1]
#     A_err = prediction_data[:, 2]

#     # Create figure with three subplots sharing the y-axis
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

#     # Create a common colorbar normalization for the first two plots
#     vmin = min(np.min(A_pred), np.min(A_exact))
#     vmax = max(np.max(A_pred), np.max(A_exact))
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     # Error plot needs its own normalization
#     error_norm = Normalize(vmin=np.min(A_err), vmax=np.max(A_err))

#     # Create scatter plots with the points
#     scatter1 = axes[0].scatter(x, y, c=A_pred, cmap=cm.viridis, s=5, norm=norm)
#     scatter2 = axes[1].scatter(x, y, c=A_exact, cmap=cm.viridis, s=5, norm=norm)
#     scatter3 = axes[2].scatter(x, y, c=A_err, cmap=cm.magma, s=5, norm=error_norm)

#     # Add colorbars
#     cbar1 = fig.colorbar(scatter1, ax=axes[0])
#     cbar1.set_label(r'$A$')
#     cbar2 = fig.colorbar(scatter2, ax=axes[1])
#     cbar2.set_label(r'$A$')
#     cbar3 = fig.colorbar(scatter3, ax=axes[2])
#     cbar3.set_label('Error')

#     # Set titles for each subplot
#     axes[0].set_title('Predicted Solution')
#     axes[1].set_title('Exact Solution')
#     axes[2].set_title('Error')

#     # Set x-labels for each subplot
#     axes[0].set_xlabel('X')
#     axes[1].set_xlabel('X')
#     axes[2].set_xlabel('X')

#     # Set a common y-label for all subplots
#     fig.text(0.01, 0.5, 'Y', va='center', rotation='vertical', fontsize=12)
#     fig.suptitle("Case 1 - Mag A, Poly Activation function (ord = 2)")
#     # Set uniform aspect ratio for all subplots
#     for ax in axes:
#         ax.set_aspect('equal')

#     # Adjust layout for better spacing
#     plt.tight_layout(rect=[0.03, 0, 1, 1])  # Adjust left margin for y-label

#     # Save figure if needed
#     plt.savefig(fig_path / 'FieldA_Comp.png', dpi=300, bbox_inches='tight')

#     plt.close()

#     B_pred = prediction_data[:, 5]
#     B_exact = prediction_data[:, 6]
#     B_err = prediction_data[:, 7]

#     # Create figure with three subplots sharing the y-axis
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

#     # Create a common colorbar normalization for the first two plots
#     vmin = min(np.min(B_pred), np.min(B_exact))
#     vmax = max(np.max(B_pred), np.max(B_exact))
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     # Error plot needs its own normalization
#     error_norm = Normalize(vmin=np.min(B_err), vmax=np.max(B_err))

#     # Create scatter plots with the points
#     scatter1 = axes[0].scatter(x, y, c=B_pred, cmap=cm.viridis, s=5)
#     scatter2 = axes[1].scatter(x, y, c=B_exact, cmap=cm.viridis, s=5, norm=norm)
#     scatter3 = axes[2].scatter(x, y, c=B_err, cmap=cm.magma, s=5, norm=error_norm)

#     # Add colorbars
#     cbar1 = fig.colorbar(scatter1, ax=axes[0])
#     cbar1.set_label(r'$B$')
#     cbar2 = fig.colorbar(scatter2, ax=axes[1])
#     cbar2.set_label(r'$B$')
#     cbar3 = fig.colorbar(scatter3, ax=axes[2])
#     cbar3.set_label('Error')

#     # Set titles for each subplot
#     axes[0].set_title('Predicted Solution')
#     axes[1].set_title('Exact Solution')
#     axes[2].set_title('Error')

#     # Set x-labels for each subplot
#     axes[0].set_xlabel('X')
#     axes[1].set_xlabel('X')
#     axes[2].set_xlabel('X')

#     # Set a common y-label for all subplots
#     fig.text(0.01, 0.5, 'Y', va='center', rotation='vertical', fontsize=12)
#     fig.suptitle("Case 1 - Mag B, Poly Activation function (ord = 2)")

#     # Set uniform aspect ratio for all subplots
#     for ax in axes:
#         ax.set_aspect('equal')

#     # Adjust layout for better spacing
#     plt.tight_layout(rect=[0.03, 0, 1, 1])  # Adjust left margin for y-label

#     # Save figure if needed
#     plt.savefig(fig_path / 'FieldB_Comp.png', dpi=300, bbox_inches='tight')

#     plt.close()


def plot_inference(predicted_path, fig_path):
    """
    Plotting code for external inference data
    """
    inference_data = np.loadtxt(i_test_external_path_stator)

    # Extract x and y coordinates
    x, y = inference_data[:, 0], inference_data[:, 1]

    prediction_data = np.loadtxt(predicted_path)

    A_pred = prediction_data[:, 0]
    A_exact = prediction_data[:, 1]
    A_err = prediction_data[:, 2]

    # Create figure with three subplots sharing the y-axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Create a common colorbar normalization for the first two plots
    vmin = min(np.min(A_pred), np.min(A_exact))
    vmax = max(np.max(A_pred), np.max(A_exact))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Error plot needs its own normalization
    error_norm = Normalize(vmin=np.min(A_err), vmax=np.max(A_err))

    # Create scatter plots with the points
    scatter1 = axes[0].scatter(x, y, c=A_pred, cmap=cm.viridis, s=5, norm=norm)
    scatter2 = axes[1].scatter(x, y, c=A_exact, cmap=cm.viridis, s=5, norm=norm)
    scatter3 = axes[2].scatter(x, y, c=A_err, cmap=cm.magma, s=5, norm=error_norm)

    # Add colorbars
    cbar1 = fig.colorbar(scatter1, ax=axes[0])
    cbar1.set_label(r"$A$")
    cbar2 = fig.colorbar(scatter2, ax=axes[1])
    cbar2.set_label(r"$A$")
    cbar3 = fig.colorbar(scatter3, ax=axes[2])
    cbar3.set_label("Error")

    # Set titles for each subplot
    axes[0].set_title("Predicted Solution")
    axes[1].set_title("Exact Solution")
    axes[2].set_title("Error")

    # Set x-labels for each subplot
    axes[0].set_xlabel("X")
    axes[1].set_xlabel("X")
    axes[2].set_xlabel("X")

    # Set a common y-label for all subplots
    fig.text(0.01, 0.5, "Y", va="center", rotation="vertical", fontsize=12)
    fig.suptitle("Case 1 - Mag A, Poly Activation function (ord = 2)")
    # Set uniform aspect ratio for all subplots
    for ax in axes:
        ax.set_aspect("equal")

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0.03, 0, 1, 1])  # Adjust left margin for y-label

    # Save figure if needed
    plt.savefig(fig_path / "FieldA_Comp.png", dpi=300, bbox_inches="tight")

    plt.close()


def plot_inference_bothzones(predicted_path_stator, predicted_path_airgap, fig_path):
    """
    Plotting code for external inference data for both stator and airgap
    """
    inference_data = np.loadtxt(i_test_external_path_stator)

    inference_data_airgap = np.loadtxt(i_test_external_path_airgap)

    # Extract x and y coordinates stator
    x, y = inference_data[:, 0], inference_data[:, 1]
    x_airgap, y_airgap = inference_data_airgap[:, 0], inference_data_airgap[:, 1]

    prediction_data_stator = np.loadtxt(predicted_path_stator)
    prediction_data_airgap = np.loadtxt(predicted_path_airgap)

    # stator
    A_pred_stator = prediction_data_stator[:, 0]
    A_exact_stator = prediction_data_stator[:, 1]
    A_err_stator = prediction_data_stator[:, 2]

    # airgap
    A_pred_airgap = prediction_data_airgap[:, 0]
    A_exact_airgap = prediction_data_airgap[:, 1]
    A_err_airgap = prediction_data_airgap[:, 2]

    # Create figure with three subplots sharing the y-axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Create a common colorbar normalization for the first two plots
    vmin = min(
        np.min(A_pred_stator),
        np.min(A_exact_stator),
        np.min(A_pred_airgap),
        np.min(A_exact_airgap),
    )
    vmax = max(
        np.max(A_pred_stator),
        np.max(A_exact_stator),
        np.max(A_pred_airgap),
        np.max(A_exact_airgap),
    )
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Error plot needs its own normalization
    error_norm = Normalize(
        vmin=min(np.min(A_err_stator), np.min(A_err_airgap)),
        vmax=max(np.max(A_err_stator), np.max(A_err_airgap)),
    )

    # Create scatter plots with the points
    scatter1 = axes[0].scatter(x, y, c=A_pred_stator, cmap=cm.viridis, s=5, norm=norm)
    scatter1_air = axes[0].scatter(
        x_airgap, y_airgap, c=A_pred_airgap, cmap=cm.viridis, s=6, norm=norm
    )
    scatter2 = axes[1].scatter(x, y, c=A_exact_stator, cmap=cm.viridis, s=5, norm=norm)
    scatter2_air = axes[1].scatter(
        x_airgap, y_airgap, c=A_exact_airgap, cmap=cm.viridis, s=6, norm=norm
    )
    scatter3 = axes[2].scatter(
        x, y, c=A_err_stator, cmap=cm.magma, s=5, norm=error_norm
    )
    scatter3_air = axes[2].scatter(
        x_airgap, y_airgap, c=A_err_airgap, cmap=cm.magma, s=6, norm=error_norm
    )

    # Add colorbars
    cbar1 = fig.colorbar(scatter1, ax=axes[0])
    cbar1.set_label(r"$A$")
    cbar2 = fig.colorbar(scatter2, ax=axes[1])
    cbar2.set_label(r"$A$")
    cbar3 = fig.colorbar(scatter3, ax=axes[2])
    cbar3.set_label("Error")

    # Set titles for each subplot
    axes[0].set_title("Predicted Solution")
    axes[1].set_title("Exact Solution")
    axes[2].set_title("Error")

    # Set x-labels for each subplot
    axes[0].set_xlabel("X")
    axes[1].set_xlabel("X")
    axes[2].set_xlabel("X")

    # Set a common y-label for all subplots
    fig.text(0.01, 0.5, "Y", va="center", rotation="vertical", fontsize=12)
    fig.suptitle("Case 1 - Mag A, Poly Activation function (ord = 2)")
    # Set uniform aspect ratio for all subplots
    for ax in axes:
        ax.set_aspect("equal")

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0.03, 0, 1, 1])  # Adjust left margin for y-label

    # Save figure if needed
    plt.savefig(fig_path / "FieldA_Comp.png", dpi=300, bbox_inches="tight")

    plt.close()

    # mag B plots

    inference_data_magB_stator = np.loadtxt(
        i_test_external_path_magB_stator, skiprows=1
    )

    inference_data_magB_airgap = np.loadtxt(
        i_test_external_path__magB_airgap, skiprows=1
    )

    magB_stator_pred = prediction_data_stator[:, -1]
    magB_stator_exact = inference_data_magB_stator[:, -1]

    magB_airgap_pred = prediction_data_airgap[:, -1]
    magB_airgap_exact = inference_data_magB_airgap[:, -1]

    magB_err_stator = np.abs(magB_stator_pred - magB_stator_exact)
    magB_err_airgap = np.abs(magB_airgap_pred - magB_airgap_exact)

    # Create figure with three subplots sharing the y-axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Create a common colorbar normalization for the first two plots
    vmin_magB = min(
        np.min(magB_stator_pred),
        np.min(magB_stator_exact),
        np.min(magB_airgap_pred),
        np.min(magB_stator_exact),
    )
    vmax_magB = max(
        np.max(magB_stator_pred),
        np.max(magB_stator_exact),
        np.max(magB_airgap_pred),
        np.max(magB_stator_exact),
    )
    norm = Normalize(vmin=vmin_magB, vmax=vmax_magB)

    # Error plot needs its own normalization
    error_norm_magB = Normalize(
        vmin=min(np.min(magB_err_stator), np.min(magB_err_airgap)),
        vmax=max(np.max(magB_err_stator), np.max(magB_err_airgap)),
    )

    # Create scatter plots with the points
    scatter1 = axes[0].scatter(
        x, y, c=magB_stator_pred, cmap=cm.viridis, s=5, norm=norm
    )
    scatter1_air = axes[0].scatter(
        x_airgap, y_airgap, c=magB_airgap_pred, cmap=cm.viridis, s=6, norm=norm
    )
    scatter2 = axes[1].scatter(
        x, y, c=magB_stator_exact, cmap=cm.viridis, s=5, norm=norm
    )
    scatter2_air = axes[1].scatter(
        x_airgap, y_airgap, c=magB_airgap_exact, cmap=cm.viridis, s=6, norm=norm
    )
    scatter3 = axes[2].scatter(
        x, y, c=magB_err_stator, cmap=cm.magma, s=5, norm=error_norm_magB
    )
    scatter3_air = axes[2].scatter(
        x_airgap, y_airgap, c=magB_err_airgap, cmap=cm.magma, s=6, norm=error_norm_magB
    )

    # Add colorbars
    cbar1 = fig.colorbar(scatter1, ax=axes[0])
    cbar1.set_label(r"$A$")
    cbar2 = fig.colorbar(scatter2, ax=axes[1])
    cbar2.set_label(r"$A$")
    cbar3 = fig.colorbar(scatter3, ax=axes[2])
    cbar3.set_label("Error")

    # Set titles for each subplot
    axes[0].set_title("Predicted Solution")
    axes[1].set_title("Exact Solution")
    axes[2].set_title("Error")

    # Set x-labels for each subplot
    axes[0].set_xlabel("X")
    axes[1].set_xlabel("X")
    axes[2].set_xlabel("X")

    # Set a common y-label for all subplots
    fig.text(0.01, 0.5, "Y", va="center", rotation="vertical", fontsize=12)
    fig.suptitle(
        f"Case 3 - Mag B {'poly order ' + str(len(i_polynomial_coeffs) - 1) if i_use_polynomial else 'tanh'}"
    )

    for ax in axes:
        ax.set_aspect("equal")

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0.03, 0, 1, 1])  # Adjust left margin for y-label

    # Save figure if needed
    plt.savefig(fig_path / "FieldB_Comp.png", dpi=300, bbox_inches="tight")

    plt.close()


if i_test_external:
    test_points_solution = np.loadtxt(
        "/home/saibhargav/Projects/SciREX/tests/support_files/Case_magnet/Case2_inference_v2.txt"
    )

    # airgap solution
    test_points_solution_airgap = np.loadtxt(
        "/home/saibhargav/Projects/SciREX/tests/support_files/Case_magnet/Case2_inference_airgap_v2.txt"
    )

    test_points_external = np.hstack(
        (test_points_solution[:, 0][:, None], test_points_solution[:, 1][:, None])
    )

    # airgap testpoints
    test_points_external_airgap = np.hstack(
        (
            test_points_solution_airgap[:, 0][:, None],
            test_points_solution_airgap[:, 1][:, None],
        )
    )

    # predicted solution
    y_test_pred = model_stator(test_points_external)[:, 0].numpy().reshape(-1)
    y_test_pred_airgap = (
        model_air(test_points_external_airgap)[:, 0].numpy().reshape(-1)
    )

    # exact solution
    y_exact_external = test_points_solution[:, 2].reshape(-1)
    y_exact_external_airgap = test_points_solution_airgap[:, 2].reshape(-1)

    # error stator
    error = y_test_pred - y_exact_external
    l2_error = np.sqrt(np.mean(error**2))
    l1_error = np.mean(np.abs(error))
    l_inf_error = np.max(np.abs(error))

    print(
        f"Inference Metrices Stator A:\n l2 Error: {l2_error}. l1 Error: {l1_error} linf : {l_inf_error}"
    )

    # error airgap
    error_airgap = y_test_pred_airgap - y_exact_external_airgap
    l2_error_airgap = np.sqrt(np.mean(error_airgap**2))
    l1_error_airgap = np.mean(np.abs(error_airgap))
    l_inf_error_airgap = np.max(np.abs(error_airgap))

    print(
        f"Inference Metrices Airgap A:\n l2 Error: {l2_error_airgap}. l1 Error: {l1_error_airgap} linf : {l_inf_error_airgap}"
    )

    b_pred = model_stator.inference(test_points_external)

    bx = b_pred["Bx"].numpy().reshape(-1)
    by = b_pred["By"].numpy().reshape(-1)
    mag_b = b_pred["B"].numpy().reshape(-1)

    solution_array = np.c_[
        y_test_pred,
        y_exact_external,
        np.abs(y_exact_external - y_test_pred),
        bx,
        by,
        mag_b,
    ]

    b_pred_airgap = model_air.inference(test_points_external_airgap)

    bx_airgap = b_pred_airgap["Bx"].numpy().reshape(-1)
    by_airgap = b_pred_airgap["By"].numpy().reshape(-1)
    mag_b_airgap = b_pred_airgap["B"].numpy().reshape(-1)

    solution_array_airgap = np.c_[
        y_test_pred_airgap,
        y_exact_external_airgap,
        np.abs(y_exact_external_airgap - y_test_pred_airgap),
        bx_airgap,
        by_airgap,
        mag_b_airgap,
    ]

    # write the inference outputs to a txt file

    np.savetxt(output_folder / "inference_results.txt", solution_array)

    np.savetxt(output_folder / "inference_results_airgap.txt", solution_array_airgap)

    print("Saved Inference results!")

    plot_inference_bothzones(
        output_folder / "inference_results.txt",
        output_folder / "inference_results_airgap.txt",
        output_folder,
    )
    os.system(
        f"cp scirex/core/sciml/fastvpinns/model/model_magnetostatics_magnet.py {output_folder}"
    )
    os.system(
        f"cp examples/sciml/fastvpinns/forward_problems/example_magnetostatics_magnet.py {output_folder}"
    )

    print("Saved inference plots!")


# compute the error
error_stator = np.abs(y_exact_stator - y_pred_stator)
model_stator.save_weights(str(Path(i_output_path) / "model_weights"))
solution_array_final = np.c_[
    y_pred_stator, y_exact_stator, np.abs(y_exact_stator - y_pred_stator)
]
domain_stator.write_vtk(
    solution_array_final,
    output_path=i_output_path,
    filename=f"prediction_stator_final.vtk",
    data_names=["Sol", "Exact", "Error"],
)

b_pred_final = model_stator.inference(test_points_stator)
bx_final = b_pred_final["Bx"].numpy().reshape(-1)
by_final = b_pred_final["By"].numpy().reshape(-1)
mag_b_final = b_pred_final["B"].numpy().reshape(-1)
b_solution_array_final = np.c_[bx_final, by_final, mag_b_final]
domain_stator.write_vtk(
    b_solution_array_final,
    output_path=i_output_path,
    filename=f"prediction_stator_Bfield_final.vtk",
    data_names=["Bx", "By", "B"],
)

## Figure Plots for stator
# 1. Loss Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array_stator)
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.savefig(str(output_folder / "loss_plot_stator.png"))
plt.close()  # Close the figure to free memory

# 2. Loss Components
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array_stator, label="Total Loss")
plt.plot(dirichlet_loss_array_stator, label="Dirichlet Loss")
plt.plot(interface_loss_array_stator, label="Interface Loss")
plt.plot(residual_loss_array_stator, label="Residual Loss")
plt.title("Loss Components Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(str(output_folder / "loss_components_stator_plot.png"))
plt.close()  # Close the figure to free memory


## Figure Plots for air
# 1. Loss Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array_air)
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.savefig(str(output_folder / "loss_plot_air.png"))
plt.close()  # Close the figure to free memory

# 2. Loss Components
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array_air, label="Total Loss")
plt.plot(dirichlet_loss_array_air, label="Dirichlet Loss")
plt.plot(interface_loss_array_air, label="Interface Loss")
plt.plot(residual_loss_array_air, label="Residual Loss")
plt.title("Loss Components Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(str(output_folder / "loss_components_air_plot.png"))
plt.close()  # Close the figure to free memory

# 2. Exact Solution Contour Plot
# plt.figure(figsize=(6.4, 4.8), dpi=300)
# contour_exact = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_exact, 100)
# plt.title("Exact Solution")
# plt.xlabel("x")
# plt.ylabel("y")
# cbar = plt.colorbar(contour_exact)
# plt.tight_layout()
# plt.savefig(str(output_folder / "exact_solution.png"))
# plt.close()

# # 3. Predicted Solution Contour Plot
# plt.figure(figsize=(6.4, 4.8), dpi=300)
# contour_pred = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_pred, 100)
# plt.title("Predicted Solution")
# plt.xlabel("x")
# plt.ylabel("y")
# cbar = plt.colorbar(contour_pred)
# plt.tight_layout()
# plt.savefig(str(output_folder / "predicted_solution.png"))
# plt.close()

# # 4. Error Contour Plot
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
l2_error = np.sqrt(np.mean(error_stator**2))
l1_error = np.mean(np.abs(error_stator))
l_inf_error = np.max(np.abs(error_stator))
rel_l2_error = l2_error / np.sqrt(np.mean(y_exact_stator**2))
rel_l1_error = l1_error / np.mean(np.abs(y_exact_stator))
rel_l_inf_error = l_inf_error / np.max(np.abs(y_exact_stator))

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
weights_file_path_stator = output_folder / "model_magnetostatics_stator.h5"

# save the model weights to the folder
model_stator.save_weights(
    str(weights_file_path_stator)
)  # Save the model in the SavedModel

weights_file_path_air = output_folder / "model_magnetostatics_air.h5"

# save the model weights to the folder
model_stator.save_weights(
    str(weights_file_path_air)
)  # Save the model in the SavedModel
