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
# Author: Thivin Anandh D

# Importing the required libraries
import numpy as np


def compute_l2_error(u_exact, u_approx):
    """
    This function will compute the L2 error between the exact solution and the approximate solution.

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray

    :return: L2 error between the exact and approximate solutions
    :rtype: float
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


def compute_l1_error(u_exact, u_approx):
    """
    This function will compute the L1 error between the exact solution and the approximate solution.

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray

    :return: L1 error between the exact and approximate solutions
    :rtype: float
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


def compute_linf_error(u_exact, u_approx):
    """
    This function will compute the L_inf error between the exact solution and the approximate solution.

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray

    :return: L_inf error between the exact and approximate solutions
    :rtype: float
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


def compute_l2_error_relative(u_exact, u_approx):
    """
    This function will compute the relative L2 error between the exact solution and the approximate solution.

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray

    :return: relative L2 error between the exact and approximate solutions
    :rtype: float
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


def compute_linf_error_relative(u_exact, u_approx):
    """
    This function will compute the relative L_inf error between the exact solution and the approximate solution.

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray

    :return: relative L_inf error between the exact and approximate solutions
    :rtype: float
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


def compute_l1_error_relative(u_exact, u_approx):
    """
    This function will compute the relative L1 error between the exact solution and the approximate solution.

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray

    :return: relative L1 error between the exact and approximate solutions
    :rtype: float
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


def compute_errors_combined(u_exact, u_approx):
    """
    This function will compute the L1, L2 and L_inf absolute and relative errors.

    :param u_exact: numpy array containing the exact solution
    :type u_exact: numpy.ndarray
    :param u_approx: numpy array containing the approximate solution
    :type u_approx: numpy.ndarray

    :return: L1, L2 and L_inf absolute and relative errors
    :rtype: tuple

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
