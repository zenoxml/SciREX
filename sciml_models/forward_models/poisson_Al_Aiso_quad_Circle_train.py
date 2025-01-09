"""
Example script for solving a 2D Poisson equation using FastvPINNs.

Author: Thivin Anandh (https://thivinanandh.github.io/)

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

i_mesh_generation_method = "external"  # "internal" or "external"
i_mesh_type = "quadrilateral"  # "triangular" or "quadrilateral"
i_mesh_file_name = "tests/support_files/circle_quad.mesh"  # Mesh file name
i_boundary_refinement_level = 4  # Boundary refinement level
i_boundary_sampling_method = "lhs"
i_generate_mesh_plot = True  # Generate mesh plot
i_x_min = -1  # minimum x value
i_x_max = 1  # maximum x value
i_y_min = -1  # minimum y value
i_y_max = 1  # maximum y value
i_output_path = "output/poisson_Al_Aiso_Quad_Circle_train"  # Output path

i_exact_solution_generatinon = True  # Generate exact solution


i_n_test_points_x = 100  # Number of test points in the x direction
i_n_test_points_y = 100  # Number of test points in the y direction

# fe Variables
i_fe_order = 4  # Order of the finite element space
i_fe_type = "legendre"
i_quad_order = 5  # 10 points in 1D, so 100 points in 2D for one cell
i_quad_type = "gauss-jacobi"

# Neural Network Variables
i_learning_rate_dict = {
    "initial_learning_rate": 0.002,  # Initial learning rate
    "use_lr_scheduler": True,  # Use learning rate scheduler
    "decay_steps": 5000,  # Decay steps
    "decay_rate": 0.99,  # Decay rate
    "staircase": True,  # Staircase Decay
}

i_dtype = tf.float32
i_activation = "tanh"
i_beta = 10  # Boundary Loss Penalty ( Adds more weight to the boundary loss)

# Epochs
i_num_epochs = 10000


## Setting up boundary conditions
def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return np.cos(x) * np.sin(y**2 + x)


def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return np.cos(x) * np.sin(y**2 + x)


def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return np.cos(x) * np.sin(y**2 + x)


def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return np.cos(x) * np.sin(y**2 + x)


def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    omegaX = 2.0 * np.pi
    omegaY = 2.0 * np.pi
    f_temp = -2.0 * (omegaX**2) * (np.sin(omegaX * x) * np.sin(omegaY * y))

    return (
        4 * y**2 * (2 * x - 3 * y**2) * np.sin(x + y**2) * np.cos(x)
        + 12 * y**2 * np.cos(x) * np.cos(x + y**2)
        + 2 * (-2 * x + 3 * y**2) * np.cos(x) * np.cos(x + y**2)
        + 2 * (2 * x - 3 * y**2) * np.sin(2 * x + y**2)
        + 2 * np.sin(x) * np.sin(x + y**2)
        - 2 * np.cos(x) * np.cos(x + y**2)
    )


def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    # If the exact Solution does not have an analytical expression, leave the value as 0(zero)
    # it can be set using `np.ones_like(x) * 0.0` and then ignore the errors and the error plots generated.

    omegaX = 2.0 * np.pi
    omegaY = 2.0 * np.pi
    val = -1.0 * np.sin(omegaX * x) * np.sin(omegaY * y)

    return np.cos(x) * np.sin(y**2 + x)


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


def get_bilinear_params_dict(x, y):
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = 2 * x - 3 * (y * y)

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
cells, boundary_points = domain.read_mesh(
    i_mesh_file_name,
    i_boundary_refinement_level,
    i_boundary_sampling_method,
    refinement_level=1,
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

from scirex.core.sciml.fastvpinns.model.model_anisotropic import DenseModelAnisotropic
from scirex.core.sciml.fastvpinns.physics.poisson2d_anisotropic import (
    pde_loss_poisson_anisotropic,
)

params_dict = {}
params_dict["n_cells"] = fespace.n_cells

# get the input data for the PDE
train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

# get bilinear parameters
# this function will obtain the values of the bilinear parameters from the model
# and convert them into tensors of desired dtype
bilinear_params_dict = get_bilinear_params_dict(
    datahandler.x_pde_list[:, 0:1], datahandler.x_pde_list[:, 1:2]
)

# convert all the tensors to the desired dtype
for key in bilinear_params_dict.keys():
    bilinear_params_dict[key] = tf.convert_to_tensor(
        bilinear_params_dict[key], dtype=i_dtype
    )

model = DenseModelAnisotropic(
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict,
    params_dict=params_dict,
    loss_function=pde_loss_poisson_anisotropic,
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
)

loss_array = []  # total loss
time_array = []  # time taken for each epoch


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

    if (epoch + 1) % 1000 == 0:
        y_test_pred = model(test_points).numpy().reshape(-1)
        error = y_test_pred - y_exact
        l2_error = np.sqrt(np.mean(error**2))
        l1_error = np.mean(np.abs(error))
        l_inf_error = np.max(np.abs(error))
        print(
            f"loss: {loss['loss']}, l2 Error: {l2_error}. l1 Error: {l1_error} linf : {l_inf_error}"
        )


# Get predicted values from the model
y_pred = model(test_points).numpy()
y_pred = y_pred.reshape(-1)

# compute the error
error = np.abs(y_exact - y_pred)

## Figure Plots.

# Assuming 'folder' is already defined and concatenated with 'model'
output_folder = folder / "results"

# Create the output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# 1. Loss Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array)
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.savefig(str(output_folder / "loss_plot.png"))
plt.close()  # Close the figure to free memory

# 2. Exact Solution Contour Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
contour_exact = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_exact, 100)
plt.title("Exact Solution")
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar(contour_exact)
plt.tight_layout()
plt.savefig(str(output_folder / "exact_solution.png"))
plt.close()

# 3. Predicted Solution Contour Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
contour_pred = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_pred, 100)
plt.title("Predicted Solution")
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar(contour_pred)
plt.tight_layout()
plt.savefig(str(output_folder / "predicted_solution.png"))
plt.close()

# 4. Error Contour Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
contour_error = plt.tricontourf(test_points[:, 0], test_points[:, 1], error, 100)
plt.title("Error")
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar(contour_error)
plt.tight_layout()
plt.savefig(str(output_folder / "error_plot.png"))
plt.close()


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
weights_file_path = output_folder / "model_poisson_al_aiso_quad_circle_weights.h5"

# save the model weights to the folder
model.save_weights(str(weights_file_path))  # Save the model in the SavedModel

from tensorflow.keras import layers, models


layer_dims = [2, 30, 30, 30, 1]

# Create a Sequential model
model = models.Sequential()

# Add the hidden layers (except the last layer, which will have no activation)
for dim in layer_dims[1:-1]:
    model.add(
        layers.Dense(
            units=dim,
            activation="tanh",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
    )

# Add the output layer with no activation function
model.add(
    layers.Dense(
        units=layer_dims[-1],
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )
)


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")

# Build the model with input shape of (None, 2) (which is equivalent to (?, 2))
model.build(input_shape=(None, 2))

# Print the model summary
model.summary()


# load the saved weights to the model
# Create the output folder with subfolder 'model'
model.load_weights(str(weights_file_path))

# Predict the solution
pred_solution = model(test_points).numpy().reshape(-1)

error = pred_solution - y_exact

# print errors
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


# Assuming 'folder' is already defined and concatenated with 'model'
output_folder = folder / "results_inference"

# Create the output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# 1. Loss Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
plt.plot(loss_array)
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.tight_layout()
plt.savefig(str(output_folder / "loss_plot.png"))
plt.close()  # Close the figure to free memory

# 2. Exact Solution Contour Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
contour_exact = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_exact, 100)
plt.title("Exact Solution")
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar(contour_exact)
plt.tight_layout()
plt.savefig(str(output_folder / "exact_solution.png"))
plt.close()

# 3. Predicted Solution Contour Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
contour_pred = plt.tricontourf(test_points[:, 0], test_points[:, 1], y_pred, 100)
plt.title("Predicted Solution")
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar(contour_pred)
plt.tight_layout()
plt.savefig(str(output_folder / "predicted_solution.png"))
plt.close()

# 4. Error Contour Plot
plt.figure(figsize=(6.4, 4.8), dpi=300)
contour_error = plt.tricontourf(test_points[:, 0], test_points[:, 1], error, 100)
plt.title("Error")
plt.xlabel("x")
plt.ylabel("y")
cbar = plt.colorbar(contour_error)
plt.tight_layout()
plt.savefig(str(output_folder / "error_plot.png"))
plt.close()
