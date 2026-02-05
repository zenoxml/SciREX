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
    Module: basis_2d_QN_Legendre_Special.py

    This module implements a specialized basis function class for 2D Quad elements using Legendre polynomials. 
    It provides functionality for computing basis functions and their derivatives in two dimensions, with a 
    special formulation based on differences of consecutive Legendre polynomials.

    Classes:
        Basis2DQNLegendreSpecial: Main class implementing 2D basis functions using special Legendre polynomials

    Dependencies:
        - numpy: For numerical computations
        - scipy.special: For Legendre polynomial calculations
        - matplotlib.pyplot: For visualization support
        - .basis_function_2d: For base class implementation

    Key Features:
        - Implementation of 2D Quad element basis functions using Legendre polynomials
        - Special formulation using differences of consecutive polynomials
        - Computation of function values and derivatives up to second order
        - Tensor product construction of 2D basis functions from 1D components
        - Support for variable number of shape functions

    Authors:
        Thivin Anandh (http://thivinanandh.github.io/) 
    Version Info:
        27/Dec/2024: Initial version: Thivin Anandh D
    References:
        None
    """

import numpy as np

# import the legendre polynomials
from scipy.special import legendre
import matplotlib.pyplot as plt

from .basis_function_2d import BasisFunction2D


class Basis2DQNLegendreSpecial(BasisFunction2D):
    """
    A specialized implementation of two-dimensional basis functions using Legendre polynomials for Q1 elements.

    This class provides a complete implementation for computing basis functions and their derivatives
    in two dimensions. The basis functions are constructed using a special formulation based on
    differences of consecutive Legendre polynomials.

    The class inherits from BasisFunction2D and implements all required methods for computing
    function values and derivatives up to second order.

    Attributes:
        num_shape_functions (int): Total number of shape functions in the 2D element.
            Must be a perfect square as it represents tensor product of 1D functions.

    Methods:
        test_fcn(n_test, x): Computes test functions using Legendre polynomial differences
        test_grad_fcn(n_test, x): Computes first derivatives of test functions
        test_grad_grad_fcn(n_test, x): Computes second derivatives of test functions
        value(xi, eta): Computes values of all basis functions
        gradx(xi, eta): Computes x-derivatives of all basis functions
        grady(xi, eta): Computes y-derivatives of all basis functions
        gradxx(xi, eta): Computes second x-derivatives of all basis functions
        gradyy(xi, eta): Computes second y-derivatives of all basis functions
        gradxy(xi, eta): Computes mixed xy-derivatives of all basis functions

    Implementation Details:
        - Basis functions are constructed using differences of consecutive Legendre polynomials
        - Test functions are created using Pn+1(x) - Pn-1(x) where Pn is the nth Legendre polynomial
        - All computations maintain numerical precision using numpy arrays
        - Efficient vectorized operations for multiple point evaluations
        - Tensor product construction for 2D basis functions

    Example:
        ```python
        basis = Basis2DQNLegendreSpecial(num_shape_functions=16)  # Creates 4x4 basis functions
        xi = np.linspace(-1, 1, 100)
        eta = np.linspace(-1, 1, 100)
        values = basis.value(xi, eta)
        x_derivatives = basis.gradx(xi, eta)
        ```
    """

    def __init__(self, num_shape_functions: int):
        super().__init__(num_shape_functions)

    def test_fcn(self, n_test: int, x: np.ndarray) -> np.ndarray:
        """
        Calculate the test function values for a given number of tests and input values.

        Args:
            n_test (int): The number of test functions to calculate.
            x (np.ndarray): The input values at which to evaluate the test functions.

        Returns:
            np.ndarray: An array containing the results of the test functions at the given input values.
        """
        test_total = []
        for n in range(1, n_test + 1):
            obj1 = legendre(n + 1)
            obj2 = legendre(n - 1)
            test = obj1(x) - obj2(x)
            test_total.append(test)
        return np.asarray(test_total)

    def test_grad_fcn(self, n_test: int, x: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the test function at a given point.

        Args:
            n_test (int): The number of test cases to evaluate.
            x (np.ndarray): The input value at which to evaluate the function.

        Returns:
            np.ndarray: An array containing the results of the test cases.
        """
        test_total = []
        for n in range(1, n_test + 1):
            obj1 = legendre(n + 1).deriv()
            obj2 = legendre(n - 1).deriv()
            test = obj1(x) - obj2(x)
            test_total.append(test)
        return np.asarray(test_total)

    def test_grad_grad_fcn(self, n_test: int, x: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the second derivative of a function using Legendre polynomials.

        Args:
            n_test (int): The number of test cases to evaluate.
            x (np.ndarray): The input value at which to evaluate the function.

        Returns:
            np.ndarray: An array containing the results of the test cases.
        """
        test_total = []
        for n in range(1, n_test + 1):
            obj1 = legendre(n + 1).deriv(2)
            obj2 = legendre(n - 1).deriv(2)
            test = obj1(x) - obj2(x)

            test_total.append(test)
        return np.asarray(test_total)

    def value(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): The xi coordinates.
            eta (np.ndarray): The eta coordinates.

        Returns:
            np.ndarray: The values of the basis functions.
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_x = self.test_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.test_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_x[i, :] * test_function_y
            )

        return values

    def gradx(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the x-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): The xi coordinates.
            eta (np.ndarray): The eta coordinates.

        Returns:
            np.ndarray: The x-derivatives of the basis functions.
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_grad_x = self.test_grad_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.test_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_grad_x[i, :] * test_function_y
            )

        return values

    def grady(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the y-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): The xi coordinates.
            eta (np.ndarray): The eta coordinates.

        Returns:
            np.ndarray: The y-derivatives of the basis functions.
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_x = self.test_fcn(num_shape_func_in_1d, xi)
        test_function_grad_y = self.test_grad_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_x[i, :] * test_function_grad_y
            )

        return values

    def gradxx(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the xx-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The xx-derivatives of the basis functions.
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_grad_grad_x = self.test_grad_grad_fcn(num_shape_func_in_1d, xi)
        test_function_y = self.test_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_grad_grad_x[i, :] * test_function_y
            )

        return values

    def gradxy(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the xy-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The xy-derivatives of the basis functions.
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_grad_x = self.test_grad_fcn(num_shape_func_in_1d, xi)
        test_function_grad_y = self.test_grad_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_grad_x[i, :] * test_function_grad_y
            )

        return values

    def gradyy(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the yy-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): The xi coordinates.
            eta (np.ndarray): The eta coordinates.

        Returns:
            np.ndarray: The yy-derivatives of the basis functions.
        """
        values = np.zeros((self.num_shape_functions, len(xi)))

        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))

        test_function_x = self.test_fcn(num_shape_func_in_1d, xi)
        test_function_grad_grad_y = self.test_grad_grad_fcn(num_shape_func_in_1d, eta)

        # Generate an outer product of the test functions to generate the basis functions
        for i in range(num_shape_func_in_1d):
            values[i * num_shape_func_in_1d : (i + 1) * num_shape_func_in_1d, :] = (
                test_function_x[i, :] * test_function_grad_grad_y
            )

        return values
