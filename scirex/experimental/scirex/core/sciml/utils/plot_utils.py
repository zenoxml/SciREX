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
File: plot_utils.py
Purpose: This file contains utility functions to plot the loss function and other parameters.

Functions:
    - plot_loss_function: Plot the loss function
    - plot_array: Plot the array
    - plot_multiple_loss_function: Plot multiple loss functions
    - plot_inverse_test_loss_function: Plot the test loss function of the inverse parameter
    - plot_test_loss_function: Plot the test loss function
    - plot_test_time_loss_function: Plot the test loss as a function of time in seconds
    - plot_contour: Plot the contour plot
    - plot_inverse_param_function: Plot the predicted inverse parameter

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version Info:
    27/Dec/2024: Initial version - Thivin Anandh D
"""

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np


plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20

plt.rcParams["legend.fontsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "darkblue",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#bcbd22",
        "#8c564b",
        "#17becf",
        "#9467bd",
        "#e377c2",
        "#7f7f7f",
    ]
)


# plot the loss function
def plot_loss_function(loss_function: np.ndarray, output_path: str) -> None:
    """
    This function will plot the loss function.

    Args:
        loss_function: list of loss values
        output_path: path to save the plot

    Returns:
        None
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    plt.grid()
    plt.savefig(output_path + "/loss_function.png", dpi=300)

    plt.close()


def plot_array(
    array: list,
    output_path: str,
    filename: str,
    title: str,
    x_label="Epochs",
    y_label="Loss",
) -> None:
    """
    This function will plot the loss function.

    Args:
        array: list of loss values
        output_path: path to save the plot
        filename: filename to save the plot
        title: title of the plot
        x_label: x-axis label, defaults to "Epochs"
        y_label: y-axis label, defaults to "Loss"

    Returns:
        None
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(array)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.savefig(output_path + f"/{filename}.png", dpi=300)
    plt.close()


# general utility to plot multiple parameters
def plot_multiple_loss_function(
    loss_function_list: list,
    output_path: str,
    filename: str,
    legend_labels: str,
    y_label: str,
    title: str,
    x_label="Epochs",
):
    """
    This function will plot the loss function in log scale for multiple parameters.

    Args:
        loss_function_list: list of loss values
        output_path: path to save the plot
        filename: filename to save the plot
        legend_labels: legend labels
        y_label: y-axis label
        title: title of the plot
        x_label: x-axis label, defaults to "Epochs"

    Returns:
        None
    """

    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    for loss_function, label in zip(loss_function_list, legend_labels):
        plt.plot(loss_function, label=label)

    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(output_path + f"/{filename}.png", dpi=300)
    plt.close()


# plot the loss function
def plot_inverse_test_loss_function(loss_function: list, output_path: str) -> None:
    """
    This function will plot the test loss function of the inverse parameter.

    Args:
        loss_function: list of loss values
        output_path: path to save the plot

    Returns:
        None
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    plt.savefig(output_path + "/test_inverse_loss_function.png", dpi=300)
    plt.close()


def plot_test_loss_function(
    loss_function: np.ndarray, output_path: str, fileprefix=""
) -> None:
    """
    This function will plot the test loss function.

    Args:
        loss_function: list of loss values
        output_path: path to save the plot
        fileprefix: prefix for the filename

    Returns:
        None
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    if fileprefix == "":
        plt.savefig(output_path + "/test_loss_function.png", dpi=300)
    else:
        plt.savefig(output_path + "/" + fileprefix + "_test_loss_function.png", dpi=300)
    plt.close()


def plot_test_time_loss_function(
    time_array: np.ndarray, loss_function: np.ndarray, output_path: str
) -> None:
    """
    This function will plot the test loss as a function of time in seconds.

    Args:
        time_array: time array
        loss_function: list of loss values
        output_path: path to save the plot

    Returns:
        None
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(time_array, loss_function)
    # plot y axis in log scale
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Time [s]")
    plt.ylabel("MAE Loss")
    plt.title("Loss Function")
    plt.tight_layout()
    plt.savefig(output_path + "/test_time_loss_function.png", dpi=300)
    plt.close()


def plot_contour(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    output_path: str,
    filename: str,
    title: str,
) -> None:
    """
    This function will plot the contour plot.

    Args:
        x(np.ndarray): x values
        y(np.ndarray): y values
        z(np.ndarray): z values
        output_path: path to save the plot
        filename: filename to save the plot
        title: title of the plot

    Returns:
        None
    """

    plt.figure(figsize=(6.4, 4.8))
    plt.contourf(x, y, z, levels=100, cmap="jet")
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path + "/" + filename + ".png", dpi=300)

    plt.close()


# plot the Inverse parameter prediction
def plot_inverse_param_function(
    inverse_predicted: list,
    inverse_param_name: str,
    actual_value: float,
    output_path: str,
    file_prefix: str,
) -> None:
    """
    This function will plot the predicted inverse parameter.

    Args:
        inverse_predicted(list): list of predicted inverse parameter values
        inverse_param_name(str): name of the inverse parameter
        actual_value(float): actual value of the inverse parameter
        output_path(str): path to save the plot
        file_prefix(str): prefix for the filename

    Returns:
        None
    """
    # plot the loss function
    plt.figure(figsize=(6.4, 4.8), dpi=300)
    plt.plot(inverse_predicted, label="Predicted " + inverse_param_name)

    # draw a horizontal dotted line at the actual value
    plt.hlines(
        actual_value,
        0,
        len(inverse_predicted),
        colors="k",
        linestyles="dashed",
        label="Actual " + inverse_param_name,
    )

    # plot y axis in log scale
    # plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel(inverse_param_name)

    # plt.title("Loss Function")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(output_path + f"/{file_prefix}.png", dpi=300)
    plt.close()

    # plot the loss of inverse parameter
    plt.figure(figsize=(6.4, 4.8), dpi=300)
    actual_val_array = np.ones_like(inverse_predicted) * actual_value
    plt.plot(abs(actual_val_array - inverse_predicted))
    plt.xlabel("Epochs")
    plt.ylabel("Absolute Error")
    plt.yscale("log")
    plt.title("Absolute Error of " + inverse_param_name)
    plt.tight_layout()
    plt.savefig(output_path + f"/{file_prefix}_absolute_error.png", dpi=300)
    plt.close()
