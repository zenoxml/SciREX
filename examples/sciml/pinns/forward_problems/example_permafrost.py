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
Example script for solving a 1+1D permafrost model using PINNs.

Author: Divij Ghose (https://divijghose.github.io/)

Versions:
    - 27-Jan-2025 (Version 0.1): Initial Implementation
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

from scirex.core.sciml.pinns.model.model_scalar_transient import DenseModel
from scirex.core.sciml.pinns.physics.energy import pde_loss_permafrost
from scirex.core.sciml.pinns.optimizers.lbfgs import *

i_mesh_type = "quadrilateral"  # "quadrilateral"
i_mesh_generation_method = "internal"  # "internal" or "external"


i_x_min = 0  # minimum x value
i_x_max = 1  # maximum x value

i_t_min = 0  # minimum time value
i_t_max = 1  # maximum time value

i_n_collocation_points = 10000  # Number of collocation points
i_n_boundary_points = 1000  # Number of points on the boundary
i_output_path = "output/permafrost_new"  # Output path

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
i_num_epochs = 50000  # Number of epochs

# penalty parameter for boundary loss
i_beta_boundary = 10.0
i_beta_initial = 100


#############################################################################
## Generate the input data (Collacation points) within the domain
#############################################################################

# Generate the collocation points in the domain between x_min and x_max and y_min and y_max
input_points = lhs(2, i_n_collocation_points)

# Rescale the points to the x_min, x_max, y_min and y_max
input_points[:, 0] = input_points[:, 0] * (i_x_max - i_x_min) + i_x_min
input_points[:, 1] = input_points[:, 1] * (i_t_max - i_t_min) + i_t_min


#############################################################################
################# Generate the Initial Condition Data #######################
#############################################################################
def initial_data_w(x):
    """
    Function to generate the initial data for the Poisson 2D problem
    """
    val = np.exp(-2.0)
    return np.ones_like(x) * val


#############################################################################
####################### Generate the Boundary Data ##########################
#############################################################################


def boundary_data_theta_left(t):
    """
    Function to generate the boundary data for the Poisson 2D problem
    """
    val = -2.0
    return np.ones_like(t) * val


def boundary_data_theta_right(t):
    """
    Function to generate the boundary data for the Poisson 2D problem
    """
    val = 1.0
    return np.ones_like(t) * val


#############################################################################
################# Generate the Boundary Point ################################
#############################################################################


# left boundary: x = i_x_min
left_x = np.ones(i_n_boundary_points) * i_x_min
left_t = np.linspace(i_t_min, i_t_max, i_n_boundary_points)
# concatenate the points into 3D array
left_boundary_points = np.vstack((left_x, left_t)).T
left_boundary_values = boundary_data_theta_left(left_t)

# right boundary: x = i_x_max
right_x = np.ones(i_n_boundary_points) * i_x_max
right_t = np.linspace(i_t_min, i_t_max, i_n_boundary_points)
# concatenate the points into 2D array
right_boundary_points = np.vstack((right_x, right_t)).T
right_boundary_values = boundary_data_theta_right(right_t)


# initial condition boundary: t = i_t_min for w
initial_collocation_points = lhs(1, i_n_boundary_points)
initial_x_w = initial_collocation_points[:, 0] * (i_x_max - i_x_min) + i_x_min
initial_t_w = np.ones(i_n_boundary_points) * i_t_min


# concatenate the points into 2D array
initial_boundary_points_w = np.vstack((initial_x_w, initial_t_w)).T
initial_boundary_values_w = initial_data_w(initial_x_w)

# concatenate all the boundary points into a single array
boundary_points = np.vstack(
    (
        left_boundary_points,
        right_boundary_points,
    )
)
boundary_values = np.concatenate(
    (
        left_boundary_values,
        right_boundary_values,
    )
)


#############################################################################
## Fill the RHS of the Poisson Equation
#############################################################################


def rhs(x, t):
    """
    Function to generate the right hand side of the Poisson 2D problem
    """
    val = 0

    return np.ones_like(x) * val


# For all the collocation points, compute the right hand side
rhs_values = rhs(input_points[:, 0:1], input_points[:, 1:2])


#############################################################################
## Setup Test points
#############################################################################

# Generate the test points in the domain between x_min and x_max and y_min and y_max

test_points_x = np.linspace(i_x_min, i_x_max, i_n_test_points_x)
test_points_t = np.linspace(i_t_min, i_t_max, i_n_test_points_t)


test_points = np.array(np.meshgrid(test_points_x, test_points_t)).T.reshape(-1, 2)

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


def exact_solution_theta(x, t):
    pass


def exact_solution_w(x, t):
    pass


#############################################################################
##  Convert the input data into tensors
#############################################################################

# Convert the input points, boundary points, boundary values and rhs values into tensors
input_points = tf.convert_to_tensor(input_points, dtype=tf.float32)
boundary_points = tf.convert_to_tensor(boundary_points, dtype=tf.float32)
boundary_values = tf.reshape(
    tf.convert_to_tensor(boundary_values, dtype=tf.float32), (-1, 1)
)
initial_boundary_points_w = tf.convert_to_tensor(
    initial_boundary_points_w, dtype=tf.float32
)
initial_boundary_values_w = tf.reshape(
    tf.convert_to_tensor(initial_boundary_values_w, dtype=tf.float32), (-1, 1)
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
    layer_dims=[2, 30, 30, 30, 1],
    learning_rate_dict=i_learning_rate_dict,
    loss_function=pde_loss_permafrost,
    input_tensors_list=[
        input_points,
        boundary_points,
        boundary_values,
        initial_boundary_points_w,
        initial_boundary_values_w,
    ],
    force_function_values=rhs_values,
    tensor_dtype=i_dtype,
    activation=i_activation,
)


loss_array = []  # total loss
time_array = []  # time taken for each epoch


def semismooth(theta):
    return np.where(np.less(theta, 0.0), np.exp(2.0 * theta), theta + 1)


def exact_naren(model):
    x = np.genfromtxt("data/xcc_final")
    actual_theta = np.genfromtxt("data/theta_final")
    actual_w = np.genfromtxt("data/w_final")
    t = np.ones_like(x) * i_t_max

    # get the predicted Solution
    test_points = np.vstack((x, t)).T
    test_points = tf.convert_to_tensor(test_points, dtype=i_dtype)
    model_sol = model(test_points)
    sol_theta = model_sol[:, 0:1].numpy().reshape(-1)
    sol_w = semismooth(sol_theta)

    # calculate l1, l2 and l_inf error
    error_theta = np.abs(actual_theta - sol_theta)
    l_inf_error_theta = np.max(np.abs(error_theta))
    l2_error_theta = np.sqrt(np.mean(error_theta**2))

    error_w = np.abs(actual_w - sol_w)
    l_inf_error_w = np.max(np.abs(error_w))
    l2_error_w = np.sqrt(np.mean(error_w**2))

    print(
        f"L2 Error Theta: {l2_error_theta:.4e}, L_inf Error Theta: {l_inf_error_theta:.4e}",
        f"L2 Error W: {l2_error_w:.4e}, L_inf Error W: {l_inf_error_w:.4e}",
    )

    # Create subplots in a single figure
    plt.figure(figsize=(12, 4.8), dpi=200)

    # First subplot for Theta
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(x, actual_theta, label="Actual-Naren")
    plt.plot(x, sol_theta, label="Predicted")
    plt.xlabel("x")
    plt.ylabel("Theta")
    plt.legend()
    plt.title("Theta")

    # Second subplot for W
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(x, actual_w, label="Actual-Naren")
    plt.plot(x, sol_w, label="Predicted")
    plt.xlabel("x")
    plt.ylabel("W")
    plt.legend()
    plt.title("W")

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{i_output_path}/theta_w_comparison.png")


for epoch in tqdm(range(i_num_epochs)):
    # Train the model
    batch_start_time = time.time()
    loss = model.train_step(
        beta_boundary=i_beta_boundary,
        beta_initial=i_beta_initial,
        bilinear_params_dict=i_bilinear_params_dict,
    )
    loss_array.append(loss["loss"])
    if epoch % 2000 == 0:
        pde_loss = loss["loss_pde"]
        boundary_loss = loss["loss_dirichlet"]
        initial_loss = loss["loss_initial"]
        print(
            f"Epoch: {epoch}, Loss: {loss['loss'] :.4e}",
            f"PDE Loss: {pde_loss:.4e}, Boundary Loss: {boundary_loss:.4e}",
            f"Inital Loss: {initial_loss:.4e}",
        )

    elapsed = time.time() - batch_start_time
    # print(elapsed)
    time_array.append(elapsed)

    if epoch % 10000 == 0:
        model_sol = model(test_points)
        sol_theta = model_sol[:, 0:1].numpy().reshape(-1)
        sol_w = semismooth(sol_theta)
        # Plot contours for sol_theta and sol_w
        plt.figure(figsize=(12, 6))

        # Contour plot for sol_theta
        plt.subplot(1, 2, 1)
        plt.tricontourf(test_points[:, 0], test_points[:, 1], sol_theta, 100)
        plt.title("Contour of sol_theta")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()

        # Contour plot for sol_w
        plt.subplot(1, 2, 2)
        plt.tricontourf(test_points[:, 0], test_points[:, 1], sol_w, 100)
        plt.title("Contour of sol_w")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{i_output_path}/contours_epoch_{epoch}.png")
        plt.close()

        exact_naren(model)

    #     sol_Ez = model_sol[:, 0:1].numpy().reshape(-1)
    #     error_Ez = np.abs(exact_values_Ez - sol_Ez)
    #     l_inf_error_Ez = np.max(np.abs(error_Ez))
    #     l2_error_Ez = np.sqrt(np.mean(error_Ez**2))

    #     sol_Hx = model_sol[:, 1:2].numpy().reshape(-1)
    #     error_Hx = np.abs(exact_values_Hx - sol_Hx)
    #     l_inf_error_Hx = np.max(np.abs(error_Hx))
    #     l2_error_Hx = np.sqrt(np.mean(error_Hx**2))

    #     sol_Hy = model_sol[:, 2:3].numpy().reshape(-1)
    #     error_Hy = np.abs(exact_values_Hy - sol_Hy)
    #     l_inf_error_Hy = np.max(np.abs(error_Hy))
    #     l2_error_Hy = np.sqrt(np.mean(error_Hy**2))

    #     print(
    #         f"L2 Error Ez: {l2_error_Ez:.4e}, L_inf Error Ez: {l_inf_error_Ez:.4e}",
    #         f"L2 Error Hx: {l2_error_Hx:.4e}, L_inf Error Hx: {l_inf_error_Hx:.4e}",
    #         f"L2 Error Hy: {l2_error_Hy:.4e}, L_inf Error Hy: {l_inf_error_Hy:.4e}",
    #     )


# Get predicted values of Ez at x = 0.75 and y = 0.75 for 200 time points
def plot_line_time(model, t=0.5, num_points=1000):
    test_points_x = np.linspace(i_x_min, i_x_max, num_points)
    test_points_t = np.ones_like(test_points_x) * t
    test_points = np.vstack((test_points_x, test_points_t)).T
    test_points = tf.convert_to_tensor(test_points, dtype=i_dtype)
    pred_vals = model(test_points)
    sol_theta = pred_vals[:, 0:1].numpy()
    sol_w = semismooth(sol_theta)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(test_points_x, sol_theta, label="Theta")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Theta")
    ax[0].set_title(f"Theta at {t}")

    ax[1].plot(test_points_x, sol_w, label="W")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("W")
    ax[1].set_title(f"W at {t}")

    plt.legend()
    plt.savefig(f"{i_output_path}/line_plot_t{t}.png")
    plt.close()


def movie_line_time(model, num_points=1000):
    t_ = np.linspace(i_t_min, i_t_max, 20)
    sol_theta_A = []
    sol_w_a = []
    for t in t_:
        test_points_x = np.linspace(i_x_min, i_x_max, num_points)
        test_points_t = np.ones_like(test_points_x) * t
        test_points = np.vstack((test_points_x, test_points_t)).T
        test_points = tf.convert_to_tensor(test_points, dtype=i_dtype)
        pred_vals = model(test_points)
        sol_theta = pred_vals[:, 0:1].numpy()
        sol_w = semismooth(sol_theta)
        sol_theta_A.append(sol_theta)
        sol_w_a.append(sol_w)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    for i, t_ in enumerate(t_):
        ax[0].plot(test_points_x, sol_theta_A[i], label=f"Theta_{i}")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("Theta")
        ax[0].set_title(f"Theta at {t}")

        ax[1].plot(test_points_x, sol_w_a[i], label=f"W_{i}")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("W")
        ax[1].set_title(f"W at {t}")

    plt.legend()
    plt.savefig(f"{i_output_path}/movie_line_plot_t{t}.png")
    plt.show()
    plt.close()


def movie_line_time_animated(model, num_points=1000, fps=3):
    """
    Creates an animated visualization of the model predictions over time

    Args:
        model: The trained model
        num_points: Number of spatial points to evaluate
        fps: Frames per second for the animation
    """
    import matplotlib.animation as animation

    # Generate time points
    t_ = np.linspace(i_t_min, i_t_max, 20)
    test_points_x = np.linspace(i_x_min, i_x_max, num_points)

    # Pre-compute all solutions
    solutions = []
    for t in t_:
        test_points_t = np.ones_like(test_points_x) * t
        test_points = np.vstack((test_points_x, test_points_t)).T
        test_points = tf.convert_to_tensor(test_points, dtype=i_dtype)
        pred_vals = model(test_points)
        sol_theta = pred_vals[:, 0:1].numpy()
        sol_w = semismooth(sol_theta)
        solutions.append((sol_theta, sol_w))

    # Create figure and axis objects
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Initialize lines
    (line_theta,) = ax1.plot([], [], label="Theta")
    (line_w,) = ax2.plot([], [], label="W")

    # Set axis limits
    ax1.set_xlim(i_x_min, i_x_max)
    ax2.set_xlim(i_x_min, i_x_max)
    ax1.set_ylim(
        np.min([s[0] for s in solutions]) - 0.1, np.max([s[0] for s in solutions]) + 0.1
    )
    ax2.set_ylim(
        np.min([s[1] for s in solutions]) - 0.1, np.max([s[1] for s in solutions]) + 0.1
    )

    # Set labels and titles
    ax1.set_xlabel("x")
    ax1.set_ylabel("Theta")
    ax2.set_xlabel("x")
    ax2.set_ylabel("W")
    ax1.set_title("Theta Evolution")
    ax2.set_title("W Evolution")

    # Add time text
    time_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes)

    def init():
        """Initialize animation"""
        line_theta.set_data([], [])
        line_w.set_data([], [])
        time_text.set_text("")
        return line_theta, line_w, time_text

    def animate(i):
        """Animation function called for each frame"""
        sol_theta, sol_w = solutions[i]
        line_theta.set_data(test_points_x, sol_theta)
        line_w.set_data(test_points_x, sol_w)
        time_text.set_text(f"t = {t_[i]:.2f}")
        return line_theta, line_w, time_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(t_), interval=1000 / fps, blit=True
    )

    # Save as MP4
    writer_mp4 = animation.FFMpegWriter(fps=fps)
    anim.save(f"{i_output_path}/animation.mp4", writer=writer_mp4)

    # Save as GIF
    writer_gif = animation.PillowWriter(fps=fps)
    anim.save(f"{i_output_path}/animation.gif", writer=writer_gif)

    plt.close()

    return anim  # Return animation object in case needed for display in notebooks


plot_line_time(model=model, t=0.8)
movie_line_time(model=model)
movie_line_time_animated(model)

exact_naren(model)


exit()
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
