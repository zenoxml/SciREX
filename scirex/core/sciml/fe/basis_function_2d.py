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
    Module: basis_function_2d.py

    This module provides the abstract base class for all 2D finite element basis functions 
    used in the FE2D code. It defines the interface for computing basis functions and their 
    derivatives in reference coordinates.

    Classes:
        BasisFunction2D: Abstract base class for 2D finite element basis functions

    Dependencies:
        - abc: For abstract base class functionality

    Key Features:
        - Abstract interface for 2D basis function evaluation
        - Support for first and second order derivatives
        - Reference coordinate system (xi, eta) implementation
        - Unified interface for different polynomial bases
        - Common structure for Legendre, Jacobi, and Chebyshev implementations

    Authors:
        Thivin Anandh (http://thivinanandh.github.io/) 

    Version Info:
        27/Dec/2024: Initial version: Thivin Anandh D
       

    References:
        None
"""

from abc import abstractmethod
import numpy as np


class BasisFunction2D:
    """
    An abstract base class defining the interface for two-dimensional finite element basis functions.

    This class serves as a template for implementing various types of 2D basis functions
    (Legendre, Jacobi, Chebyshev, etc.) used in finite element computations. It defines
    the required methods for function evaluation and derivatives.

    Attributes:
        num_shape_functions (int): Number of shape functions in the element.
            Typically a perfect square for tensor-product bases.

    Methods:
        value(xi, eta): Evaluates basis functions at given reference coordinates
            Args:
                xi (float): First reference coordinate
                eta (float): Second reference coordinate
            Returns:
                float: Values of basis functions at (xi, eta)

        gradx(xi, eta): Computes x-derivatives at reference coordinates
            Args:
                xi (float): First reference coordinate
                eta (float): Second reference coordinate
            Returns:
                float: Values of x-derivatives at (xi, eta)

        grady(xi, eta): Computes y-derivatives at reference coordinates
            Args:
                xi (float): First reference coordinate
                eta (float): Second reference coordinate
            Returns:
                float: Values of y-derivatives at (xi, eta)

        gradxx(xi, eta): Computes second x-derivatives at reference coordinates
            Args:
                xi (float): First reference coordinate
                eta (float): Second reference coordinate
            Returns:
                float: Values of second x-derivatives at (xi, eta)

        gradxy(xi, eta): Computes mixed derivatives at reference coordinates
            Args:
                xi (float): First reference coordinate
                eta (float): Second reference coordinate
            Returns:
                float: Values of mixed derivatives at (xi, eta)

        gradyy(xi, eta): Computes second y-derivatives at reference coordinates
            Args:
                xi (float): First reference coordinate
                eta (float): Second reference coordinate
            Returns:
                float: Values of second y-derivatives at (xi, eta)

    Notes:
        - All coordinate inputs (xi, eta) should be in the reference element range
        - Subclasses must implement all abstract methods
        - Used as base class for specific polynomial implementations:
            - Legendre polynomials (normal and special variants)
            - Jacobi polynomials
            - Chebyshev polynomials
    """

    def __init__(self, num_shape_functions):
        self.num_shape_functions = num_shape_functions

    @abstractmethod
    def value(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Evaluates the basis function at the given xi and eta coordinates.

        Args:
            xi (float): The xi coordinate.
            eta (float): The eta coordinate.

        Returns:
            float: The value of the basis function at ( xi, eta).
        """
        pass

    @abstractmethod
    def gradx(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Computes the partial derivative of the basis function with respect to xi.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The partial derivative of the basis function with respect to xi.
        """
        pass

    @abstractmethod
    def grady(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Computes the partial derivative of the basis function with respect to eta.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The partial derivative of the basis function with respect to eta.
        """
        pass

    @abstractmethod
    def gradxx(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Computes the second partial derivative of the basis function with respect to xi.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The second partial derivative of the basis function with respect to xi.
        """
        pass

    @abstractmethod
    def gradxy(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Computes the mixed partial derivative of the basis function with respect to xi and eta.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The mixed partial derivative of the basis function with respect to xi and eta.
        """
        pass

    @abstractmethod
    def gradyy(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Computes the second partial derivative of the basis function with respect to eta.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The second partial derivative of the basis function with respect to eta.
        """
        pass


# ---------------- Legendre -------------------------- #
from .basis_2d_qn_legendre import *  # Normal Legendre from Jacobi -> J(n) = J(n-1) - J(n+1)
from .basis_2d_qn_legendre_special import *  # L(n) = L(n-1) - L(n+1)

# ---------------- Jacobi -------------------------- #
from .basis_2d_qn_jacobi import *  # Normal Jacobi

# ---------------- Chebyshev -------------------------- #
from .basis_2d_qn_chebyshev_2 import *  # Normal Chebyshev
