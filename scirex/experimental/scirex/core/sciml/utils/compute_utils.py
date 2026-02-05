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
File: compute_utils.py

Purpose: This file contains utility functions to compute errors between the exact and approximate solutions.

Functions:
    - compute_l2_error: Compute the L2 error between the exact and approximate solutions
    - compute_l1_error: Compute the L1 error between the exact and approximate solutions
    - compute_linf_error: Compute the L_inf error between the exact and approximate solutions
    - compute_l2_error_relative: Compute the relative L2 error between the exact and approximate solutions
    - compute_linf_error_relative: Compute the relative L_inf error between the exact and approximate solutions
    - compute_l1_error_relative: Compute the relative L1 error between the exact and approximate solutions
    - compute_errors_combined: Compute the L1, L2 and L_inf absolute and relative errors

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version Info:
    27/Dec/2024: Initial version - Thivin Anandh D
"""

# Importing the required libraries
import numpy as np


def compute_l2_error(u_exact: np.ndarray, u_approx: np.ndarray) -> np.ndarray:
    """
    This function will compute the L2 error between the exact solution and the approximate solution.

    Args:
        u_exact: numpy array containing the exact solution
        u_approx: numpy array containing the approximate solution

    Returns:
        L2 error between the exact and approximate solutions
    """

    # The L2 error is defined as:

    #   ..math::
    #        \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (u_{exact} - u_{approx})^2}

    # Flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l2_error = np.sqrt(np.mean(np.square(u_exact - u_approx)))
    return l2_error


def compute_l1_error(u_exact: np.ndarray, u_approx: np.ndarray) -> np.ndarray:
    """
    This function will compute the L1 error between the exact solution and the approximate solution.

    Args:
        u_exact: numpy array containing the exact solution
        u_approx: numpy array containing the approximate solution

    Returns:
        L1 error between the exact and approximate solutions
    """

    # The L1 error is defined as:

    #    ..math::
    #        \\frac{1}{N} \\sum_{i=1}^{N} |u_{exact} - u_{approx}|

    # Flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()
    # compute the L1 error
    l1_error = np.mean(np.abs(u_exact - u_approx))
    return l1_error


def compute_linf_error(u_exact: np.ndarray, u_approx: np.ndarray) -> np.ndarray:
    """
    This function will compute the L_inf error between the exact solution and the approximate solution.

    Args:
        u_exact: numpy array containing the exact solution
        u_approx: numpy array containing the approximate solution

    Returns:
        L_inf error between the exact and approximate solutions

    """

    # The L_inf error is defined as

    #    ..math::
    #        \\max_{i=1}^{N} |u_{exact} - u_{approx}|

    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L_inf error
    linf_error = np.max(np.abs(u_exact - u_approx))
    return linf_error


def compute_l2_error_relative(u_exact: np.ndarray, u_approx: np.ndarray) -> np.ndarray:
    """
    This function will compute the relative L2 error between the exact solution and the approximate solution.

    Args:
        u_exact: numpy array containing the exact solution
        u_approx: numpy array containing the approximate solution

    Returns:
        relative L2 error between the exact and approximate solutions
    """

    # The relative L2 error is defined as:

    #    ..math::
    #        \\frac{\\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (u_{exact} - u_{approx})^2}}{\\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} u_{exact}^2}}

    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l2_error = compute_l2_error(u_exact, u_approx)
    # compute the relative L2 error
    l2_error_relative = l2_error / np.sqrt(np.mean(np.square(u_exact)))
    return l2_error_relative


def compute_linf_error_relative(
    u_exact: np.ndarray, u_approx: np.ndarray
) -> np.ndarray:
    """
    This function will compute the relative L_inf error between the exact solution and the approximate solution.

    Args:
        u_exact: numpy array containing the exact solution
        u_approx: numpy array containing the approximate solution

    Returns:
        relative L_inf error between the exact and approximate solutions
    """

    # The relative L_inf error is defined as:

    #    ..math::
    #        \\frac{\\max_{i=1}^{N} |u_{exact} - u_{approx}|}{\\max_{i=1}^{N} |u_{exact}|}

    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L_inf error
    linf_error = compute_linf_error(u_exact, u_approx)
    # compute the relative L_inf error
    linf_error_relative = linf_error / np.max(np.abs(u_exact))
    return linf_error_relative


def compute_l1_error_relative(u_exact: np.ndarray, u_approx: np.ndarray) -> np.ndarray:
    """
    This function will compute the relative L1 error between the exact solution and the approximate solution.

    Args:
        u_exact: numpy array containing the exact solution
        u_approx: numpy array containing the approximate solution

    Returns:
        relative L1 error between the exact and approximate solutions
    """

    # The relative L1 error is defined as:

    #    ..math::
    #        \\frac{\\frac{1}{N} \\sum_{i=1}^{N} |u_{exact} - u_{approx}|}{\\frac{1}{N} \\sum_{i=1}^{N} |u_{exact}|}

    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l1_error = compute_l1_error(u_exact, u_approx)
    # compute the relative l1 error
    l1_error_relative = l1_error / np.mean(np.abs(u_exact))
    return l1_error_relative


def compute_errors_combined(u_exact: np.ndarray, u_approx: np.ndarray) -> np.ndarray:
    """
    This function will compute the L1, L2 and L_inf absolute and relative errors.

    Args:
        u_exact: numpy array containing the exact solution
        u_approx: numpy array containing the approximate solution

    Returns:
        tuple: The L1, L2 and L_inf absolute and relative errors
    """
    # flatten the arrays
    u_exact = u_exact.flatten()
    u_approx = u_approx.flatten()

    # compute the L2 error
    l2_error = compute_l2_error(u_exact, u_approx)
    # compute the L_inf error
    linf_error = compute_linf_error(u_exact, u_approx)
    # compute the relative L2 error
    l2_error_relative = compute_l2_error_relative(u_exact, u_approx)
    # compute the relative L_inf error
    linf_error_relative = compute_linf_error_relative(u_exact, u_approx)

    # compute L1 Error
    l1_error = compute_l1_error(u_exact, u_approx)

    # compute the relative L1 error
    l1_error_relative = compute_l1_error_relative(u_exact, u_approx)

    return (
        l2_error,
        linf_error,
        l2_error_relative,
        linf_error_relative,
        l1_error,
        l1_error_relative,
    )
