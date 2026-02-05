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
    Module: basis_2d_QN_Legendre.py

    This module implements a specialized basis function class for 2D Quad elements using Legendre polynomials. 
    It provides functionality for computing basis functions and their derivatives in two dimensions, primarily used 
    in variational physics-informed neural networks (VPINNs) with domain decomposition.

    Classes:
        Basis2DQNLegendre: Main class implementing 2D basis functions using Legendre polynomials

    Dependencies:
        - numpy: For numerical computations
        - scipy.special: For Jacobi polynomial calculations
        - .basis_function_2d: For base class implementation

    Key Features:
        - Implementation of 2D Q1 element basis functions using Legendre polynomials
        - Computation of function values and derivatives up to second order
        - Tensor product construction of 2D basis functions from 1D components
        - Specialized handling of Jacobi polynomials for test functions
        - Support for variable number of shape functions

    Authors:
        Thivin Anandh (http://thivinanandh.github.io/) 
    Version Info:
        27/Dec/2024: Initial version: Thivin Anandh D

    References:
        - hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition: https://github.com/ehsankharazmi/hp-VPINNs/
    """

import numpy as np

# import the legendre polynomials
from scipy.special import jacobi

from .basis_function_2d import BasisFunction2D


class Basis2DQNLegendre(BasisFunction2D):
    """
    A specialized implementation of two-dimensional basis functions using Legendre polynomials for Q1 elements.

    This class provides a complete implementation for computing basis functions and their derivatives
    in two dimensions, specifically designed for use in variational physics-informed neural networks
    (VPINNs) with domain decomposition. The basis functions are constructed using Legendre polynomials
    implemented through Jacobi polynomial representations with parameters (0,0).

    The class inherits from BasisFunction2D and implements all required methods for computing
    function values and derivatives up to second order.

    Attributes:
        num_shape_functions (int): Total number of shape functions in the 2D element.
            Must be a perfect square as it represents tensor product of 1D functions.

    Methods:
        jacobi_wrapper(n, a, b, x): Evaluates Jacobi polynomial at given points
        test_fcnx(n_test, x): Computes x-component test functions
        test_fcny(n_test, y): Computes y-component test functions
        dtest_fcn(n_test, x): Computes first and second derivatives of test functions
        value(xi, eta): Computes values of all basis functions
        gradx(xi, eta): Computes x-derivatives of all basis functions
        grady(xi, eta): Computes y-derivatives of all basis functions
        gradxx(xi, eta): Computes second x-derivatives of all basis functions
        gradyy(xi, eta): Computes second y-derivatives of all basis functions
        gradxy(xi, eta): Computes mixed xy-derivatives of all basis functions

    Implementation Details:
        - Basis functions are constructed as tensor products of 1D test functions
        - Test functions use Legendre polynomials via Jacobi polynomials with (0,0) parameters
        - Special cases handled for n=1,2 in derivative calculations
        - All computations maintain double precision (float64)
        - Efficient vectorized operations using numpy arrays

    Example:
        ```python
        basis = Basis2DQNLegendre(num_shape_functions=16)  # Creates 4x4 basis functions
        xi = np.linspace(-1, 1, 100)
        eta = np.linspace(-1, 1, 100)
        values = basis.value(xi, eta)
        x_derivatives = basis.gradx(xi, eta)
        ```
    """

    def __init__(self, num_shape_functions: int):
        super().__init__(num_shape_functions)

    def jacobi_wrapper(self, n: int, a: int, b: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Jacobi polynomial of degree n with parameters a and b at the given points x.

        Args:
            n (int): Degree of the Jacobi polynomial.
            a (int): First parameter of the Jacobi polynomial.
            b (int): Second parameter of the Jacobi polynomial.
            x (np.ndarray): Points at which to evaluate the Jacobi polynomial.

        Returns:
            np.ndarray: Values of the Jacobi polynomial at the given points.
        """
        x = np.array(x, dtype=np.float64)
        return jacobi(n, a, b)(x)

    ## Helper Function
    def test_fcnx(self, n_test: int, x: np.ndarray) -> np.ndarray:
        """
        Compute the x-component of the test functions for a given number of test functions and x-coordinates.

        Args:
            n_test (int): Number of test functions.
            x (np.ndarray): x-coordinates at which to evaluate the test functions.

        Returns:
            np.ndarray: Values of the x-component of the test functions.
        """
        test_total = []
        for n in range(1, n_test + 1):
            test = self.jacobi_wrapper(n + 1, 0, 0, x) - self.jacobi_wrapper(
                n - 1, 0, 0, x
            )
            test_total.append(test)
        return np.asarray(test_total, np.float64)

    def test_fcny(self, n_test: int, y: np.ndarray) -> np.ndarray:
        """
        Compute the y-component of the test functions for a given number of test functions and y-coordinates.

        Args:
            n_test (int): Number of test functions.
            y (np.ndarray): y-coordinates at which to evaluate the test functions.

        Returns:
            np.ndarray: Values of the y-component of the test functions.
        """
        test_total = []
        for n in range(1, n_test + 1):
            test = self.jacobi_wrapper(n + 1, 0, 0, y) - self.jacobi_wrapper(
                n - 1, 0, 0, y
            )
            test_total.append(test)
        return np.asarray(test_total, np.float64)

    def dtest_fcn(self, n_test: int, x: np.ndarray) -> np.ndarray:
        """
        Compute the x-derivatives of the test functions for a given number of test functions and x-coordinates.

        Args:
            n_test (int): Number of test functions.
            x (np.ndarray): x-coordinates at which to evaluate the test functions.

        Returns:
            np.ndarray: Values of the x-derivatives of the test functions.
        """
        d1test_total = []
        d2test_total = []
        for n in range(1, n_test + 1):
            if n == 1:
                d1test = ((n + 2) / 2) * self.jacobi_wrapper(n, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * self.jacobi_wrapper(
                    n - 1, 2, 2, x
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n == 2:
                d1test = ((n + 2) / 2) * self.jacobi_wrapper(n, 1, 1, x) - (
                    (n) / 2
                ) * self.jacobi_wrapper(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * self.jacobi_wrapper(
                    n - 1, 2, 2, x
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            else:
                d1test = ((n + 2) / 2) * self.jacobi_wrapper(n, 1, 1, x) - (
                    (n) / 2
                ) * self.jacobi_wrapper(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) / (2 * 2)) * self.jacobi_wrapper(
                    n - 1, 2, 2, x
                ) - ((n) * (n + 1) / (2 * 2)) * self.jacobi_wrapper(n - 3, 2, 2, x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
        return np.asarray(d1test_total), np.asarray(d2test_total)

    def value(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the values of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): x-coordinates at which to evaluate the basis functions.
            eta (np.ndarray): y-coordinates at which to evaluate the basis functions.

        Returns:
            np.ndarray: Values of the basis functions.
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        test_x = self.test_fcnx(num_shape_func_in_1d, xi)
        test_y = self.test_fcny(num_shape_func_in_1d, eta)
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                test_x[i, :] * test_y
            )

        return values

    def gradx(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the x-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): x-coordinates at which to evaluate the basis functions.
            eta (np.ndarray): y-coordinates at which to evaluate the basis functions.

        Returns:
            np.ndarray: Values of the x-derivatives of the basis functions.
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        grad_test_x = self.dtest_fcn(num_shape_func_in_1d, xi)[0]
        test_y = self.test_fcny(num_shape_func_in_1d, eta)
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                grad_test_x[i, :] * test_y
            )

        return values

    def grady(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the y-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): x-coordinates at which to evaluate the basis functions.
            eta (np.ndarray): y-coordinates at which to evaluate the basis functions.

        Returns:
            np.ndarray: Values of the y-derivatives of the basis functions.
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        test_x = self.test_fcnx(num_shape_func_in_1d, xi)
        grad_test_y = self.dtest_fcn(num_shape_func_in_1d, eta)[0]
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                test_x[i, :] * grad_test_y
            )

        return values

    def gradxx(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the xx-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): x-coordinates at which to evaluate the basis functions.
            eta (np.ndarray): y-coordinates at which to evaluate the basis functions.

        Returns:
            np.ndarray: Values of the xx-derivatives of the basis functions.
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        grad_grad_x = self.dtest_fcn(num_shape_func_in_1d, xi)[1]
        test_y = self.test_fcny(num_shape_func_in_1d, eta)
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                grad_grad_x[i, :] * test_y
            )

        return values

    def gradxy(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the xy-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): x-coordinates at which to evaluate the basis functions.
            eta (np.ndarray): y-coordinates at which to evaluate the basis functions.

        Returns:
            np.ndarray: Values of the xy-derivatives of the basis functions.
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        grad_test_x = self.dtest_fcn(num_shape_func_in_1d, xi)[0]
        grad_test_y = self.dtest_fcn(num_shape_func_in_1d, eta)[0]
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                grad_test_x[i, :] * grad_test_y
            )

        return values

    def gradyy(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the yy-derivatives of the basis functions at the given (xi, eta) coordinates.

        Args:
            xi (np.ndarray): x-coordinates at which to evaluate the basis functions.
            eta (np.ndarray): y-coordinates at which to evaluate the basis functions.

        Returns:
            np.ndarray: Values of the yy-derivatives of the basis functions.
        """
        num_shape_func_in_1d = int(np.sqrt(self.num_shape_functions))
        test_x = self.test_fcnx(num_shape_func_in_1d, xi)
        grad_grad_y = self.dtest_fcn(num_shape_func_in_1d, eta)[1]
        values = np.zeros((self.num_shape_functions, len(xi)), dtype=np.float64)

        for i in range(num_shape_func_in_1d):
            values[num_shape_func_in_1d * i : num_shape_func_in_1d * (i + 1), :] = (
                test_x[i, :] * grad_grad_y
            )

        return values
