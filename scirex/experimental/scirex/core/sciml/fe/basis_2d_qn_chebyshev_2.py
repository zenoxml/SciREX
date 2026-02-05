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
    Module: basis_2d_qn_chebyshev_2.py

    This module implements a specialized basis function class for 2D finite elements using Chebyshev polynomials. 
    It provides functionality for computing basis functions and their derivatives in two dimensions, primarily used 
    in variational physics-informed neural networks (VPINNs) with domain decomposition.

    Classes:
        Basis2DQNChebyshev2: Main class implementing 2D basis functions using Chebyshev polynomials

    Dependencies:
        - numpy: For numerical computations and array operations
        - scipy.special: For Jacobi polynomial calculations and evaluations
        - .basis_function_2d: For base class BasisFunction2D implementation

    Key Features:
        - Implementation of 2D element basis functions using Chebyshev polynomials
        - Computation of function values and derivatives up to second order
        - Tensor product construction of 2D basis functions from 1D components
        - Specialized handling of Jacobi polynomials for test functions
        - Support for variable number of shape functions through initialization

    Authors:
        Thivin Anandh (http://thivinanandh.github.io/) 

    Version Info:
        27/Dec/2024: Initial version: Thivin Anandh D

    References:
        - hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition: https://github.com/ehsankharazmi/hp-VPINNs/
    """

# import the legendre polynomials
from scipy.special import jacobi

import numpy as np
from .basis_function_2d import BasisFunction2D


class Basis2DQNChebyshev2(BasisFunction2D):
    """A specialized implementation of two-dimensional basis functions using Chebyshev polynomials for Q1 elements.

    This class provides a complete implementation for computing basis functions and their derivatives
    in two dimensions, specifically designed for use in variational physics-informed neural networks
    (VPINNs) with domain decomposition. The basis functions are constructed using Chebyshev polynomials
    through Jacobi polynomial representations.

    The class inherits from BasisFunction2D and implements all required methods for computing
    function values and derivatives. The implementation follows the methodology described in
    hp-VPINNs research by Ehsan Kharazmi et al.

    Attributes:
        num_shape_functions (int): Total number of shape functions in the 2D element.
            Must be a perfect square as it represents tensor product of 1D functions.

    Methods:
        value(xi, eta): Computes values of all basis functions at given points
        gradx(xi, eta): Computes x-derivatives of all basis functions
        grady(xi, eta): Computes y-derivatives of all basis functions
        gradxx(xi, eta): Computes second x-derivatives of all basis functions
        gradyy(xi, eta): Computes second y-derivatives of all basis functions
        gradxy(xi, eta): Computes mixed xy-derivatives of all basis functions

    Implementation Details:
        - Basis functions are constructed as tensor products of 1D test functions
        - Test functions are derived from normalized Jacobi polynomials
        - Special cases are handled for first few polynomial degrees in derivatives
        - All computations maintain double precision (float64)
        - Efficient vectorized operations using numpy arrays

    Example:
        ```python
        basis = Basis2DQNChebyshev2(num_shape_functions=16)  # Creates 4x4 basis functions
        xi = np.linspace(-1, 1, 100)
        eta = np.linspace(-1, 1, 100)
        values = basis.value(xi, eta)
        x_derivatives = basis.gradx(xi, eta)
        ```

    Notes:
        - num_shape_functions must be a perfect square
        - All coordinate inputs (xi, eta) should be in the range [-1, 1]
        - Implementation optimized for vectorized operations on multiple points
        - Based on hp-VPINNs methodology: https://github.com/ehsankharazmi/hp-VPINNs/

    References:
        Kharazmi, E., et al. "hp-VPINNs: Variational Physics-Informed Neural Networks
        With Domain Decomposition"
    """

    def __init__(self, num_shape_functions: int):
        super().__init__(num_shape_functions)

    def jacobi_wrapper(self, n: int, a: int, b: int, x: np.ndarray) -> np.ndarray:
        """Evaluates Jacobi polynomial at specified points.

        Computes values of nth degree Jacobi polynomial with parameters (a,b)
        at given points x.

        Args:
            n: Degree of Jacobi polynomial. Must be non-negative integer.
            a: First parameter of Jacobi polynomial
            b: Second parameter of Jacobi polynomial
            x: Points at which to evaluate polynomial
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of Jacobi polynomial at input points
                Shape: Same as input x

        Notes:
            Wrapper around scipy.special.jacobi that ensures float64 precision
            and proper array handling.
        """
        x = np.array(x, dtype=np.float64)
        return jacobi(n, a, b)(x)

    ## Helper Function
    def test_fcnx(self, n_test: int, x: np.ndarray) -> np.ndarray:
        """Computes x-component test functions.

        Evaluates the x-direction test functions constructed as differences
        of normalized Jacobi polynomials.

        Args:
            n_test: Number of test functions to compute
            x: Points at which to evaluate functions
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of test functions at input points
                Shape: (n_test, n_points)

        Notes:
            Test functions are constructed as differences of normalized Jacobi
            polynomials following hp-VPINNs methodology.
        """
        test_total = []
        for n in range(1, n_test + 1):
            test = self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, x) / self.jacobi_wrapper(
                n + 1, -1 / 2, -1 / 2, 1
            ) - self.jacobi_wrapper(n - 1, -1 / 2, -1 / 2, x) / self.jacobi_wrapper(
                n - 1, -1 / 2, -1 / 2, 1
            )
            test_total.append(test)
        return np.asarray(test_total, np.float64)

    def test_fcny(self, n_test: int, y: np.ndarray) -> np.ndarray:
        """Computes y-component test functions.

        Evaluates the y-direction test functions constructed as differences
        of normalized Jacobi polynomials.

        Args:
            n_test: Number of test functions to compute
            y: Points at which to evaluate functions
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of test functions at input points
                Shape: (n_test, n_points)

        Notes:
            Test functions are constructed as differences of normalized Jacobi
            polynomials following hp-VPINNs methodology.
        """
        test_total = []
        for n in range(1, n_test + 1):
            test = self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, y) / self.jacobi_wrapper(
                n + 1, -1 / 2, -1 / 2, 1
            ) - self.jacobi_wrapper(n - 1, -1 / 2, -1 / 2, y) / self.jacobi_wrapper(
                n - 1, -1 / 2, -1 / 2, 1
            )
            test_total.append(test)
        return np.asarray(test_total, np.float64)

    def dtest_fcn(self, n_test: int, x: np.ndarray) -> np.ndarray:
        """Computes first and second derivatives of test functions.

        Calculates derivatives of test functions constructed from Jacobi
        polynomials, handling special cases for n=1,2 separately.

        Args:
            n_test: Number of test functions
            x: Points at which to evaluate derivatives
                Shape: (n_points,)

        Returns:
            tuple(np.ndarray, np.ndarray): First and second derivatives
                First element: First derivatives, shape (n_test, n_points)
                Second element: Second derivatives, shape (n_test, n_points)

        Notes:
            Special cases for n=1,2 ensure proper derivative calculations
            following hp-VPINNs methodology.
        """
        d1test_total = []
        d2test_total = []
        for n in range(1, n_test + 1):
            if n == 1:
                d1test = (
                    ((n + 1) / 2)
                    * self.jacobi_wrapper(n, 1 / 2, 1 / 2, x)
                    / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1)
                )
                d2test = (
                    ((n + 2) * (n + 1) / (2 * 2))
                    * self.jacobi_wrapper(n - 1, 3 / 2, 3 / 2, x)
                    / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1)
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n == 2:
                d1test = ((n + 1) / 2) * self.jacobi_wrapper(
                    n, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1) - (
                    (n - 1) / 2
                ) * self.jacobi_wrapper(
                    n - 2, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(
                    n - 1, -1 / 2, -1 / 2, 1
                )
                d2test = (
                    ((n + 2) * (n + 1) / (2 * 2))
                    * self.jacobi_wrapper(n - 1, 3 / 2, 3 / 2, x)
                    / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1)
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            else:
                d1test = ((n + 1) / 2) * self.jacobi_wrapper(
                    n, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1) - (
                    (n - 1) / 2
                ) * self.jacobi_wrapper(
                    n - 2, 1 / 2, 1 / 2, x
                ) / self.jacobi_wrapper(
                    n - 1, -1 / 2, -1 / 2, 1
                )
                d2test = ((n + 2) * (n + 1) / (2 * 2)) * self.jacobi_wrapper(
                    n - 1, 3 / 2, 3 / 2, x
                ) / self.jacobi_wrapper(n + 1, -1 / 2, -1 / 2, 1) - (
                    (n) * (n - 1) / (2 * 2)
                ) * self.jacobi_wrapper(
                    n - 3, 3 / 2, 3 / 2, x
                ) / self.jacobi_wrapper(
                    n - 1, -1 / 2, -1 / 2, 1
                )
                d1test_total.append(d1test)
                d2test_total.append(d2test)
        return np.asarray(d1test_total), np.asarray(d2test_total)

    def value(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Evaluates basis functions at given coordinates.

        Computes values of all basis functions at specified (xi,eta) points
        using tensor product of 1D test functions.

        Args:
            xi: x-coordinates at which to evaluate functions
                Shape: (n_points,)
            eta: y-coordinates at which to evaluate functions
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of all basis functions
                Shape: (num_shape_functions, n_points)

        Notes:
            Basis functions are constructed as products of 1D test functions
            in x and y directions.
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
        """Computes x-derivatives of basis functions.

        Evaluates partial derivatives with respect to x of all basis
        functions at given coordinates.

        Args:
            xi: x-coordinates at which to evaluate derivatives
                Shape: (n_points,)
            eta: y-coordinates at which to evaluate derivatives
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of x-derivatives
                Shape: (num_shape_functions, n_points)

        Notes:
            Uses product rule with x-derivatives of test functions in
            x-direction and values in y-direction.
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
        """Computes y-derivatives of basis functions.

        Evaluates partial derivatives with respect to y of all basis
        functions at given coordinates.

        Args:
            xi: x-coordinates at which to evaluate derivatives
                Shape: (n_points,)
            eta: y-coordinates at which to evaluate derivatives
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of y-derivatives
                Shape: (num_shape_functions, n_points)

        Notes:
            Uses product rule with values in x-direction and y-derivatives
            of test functions in y-direction.
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
        """Computes second x-derivatives of basis functions.

        Evaluates second partial derivatives with respect to x of all basis
        functions at given coordinates.

        Args:
            xi: x-coordinates at which to evaluate derivatives
                Shape: (n_points,)
            eta: y-coordinates at which to evaluate derivatives
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of second x-derivatives
                Shape: (num_shape_functions, n_points)

        Notes:
            Uses product rule with second x-derivatives of test functions in
            x-direction and values in y-direction.
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
        """Computes second x-derivatives of basis functions.

        Evaluates second partial derivatives with respect to x of all basis
        functions at given coordinates.

        Args:
            xi: x-coordinates at which to evaluate derivatives
                Shape: (n_points,)
            eta: y-coordinates at which to evaluate derivatives
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of second x-derivatives
                Shape: (num_shape_functions, n_points)

        Notes:
            Uses product rule with second x-derivatives of test functions in
            x-direction and y derivative values in y-direction.
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
        """Computes second x-derivatives of basis functions.

        Evaluates second partial derivatives with respect to x of all basis
        functions at given coordinates.

        Args:
            xi: x-coordinates at which to evaluate derivatives
                Shape: (n_points,)
            eta: y-coordinates at which to evaluate derivatives
                Shape: (n_points,)

        Returns:
            np.ndarray: Values of second x-derivatives
                Shape: (num_shape_functions, n_points)

        Notes:
            Uses product rule with second y-derivatives of test functions in
            x-direction and values in y-direction.
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
