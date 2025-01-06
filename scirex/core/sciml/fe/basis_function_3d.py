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
    Module: basis_function_3d.py

    This module provides the abstract base class for all 3D finite element basis functions. 
    It defines the interface for computing basis functions and their derivatives in three-dimensional 
    reference coordinates.

    Classes:
        BasisFunction3D: Abstract base class for 3D finite element basis functions

    Dependencies:
        - abc: For abstract base class functionality

    Key Features:
        - Abstract interface for 3D basis function evaluation
        - Support for first and second order derivatives
        - Reference coordinate system (xi, eta, zeta) implementation
        - Unified interface for different polynomial bases
        - Systematic derivative computation in three dimensions

    Authors:
        Thivin Anandh D (https://thivinanandh.github.io)

    Version Info:
        27/Dec/2024: Initial version: Thivin Anandh D

    References:
        None
"""

from abc import abstractmethod
import numpy as np


class BasisFunction3D:  # pragma: no cover
    """
    An abstract base class defining the interface for three-dimensional finite element basis functions.

    This class serves as a template for implementing various types of 3D basis functions
    used in finite element computations. It defines the required methods for function
    evaluation and derivatives in three dimensions.

    Attributes:
        num_shape_functions (int): The number of shape functions in the 3D element.
            Must be specified during initialization.

    Methods:
        value(xi, eta, zeta): Evaluates the basis function at given coordinates
            Args:
                xi (np.ndarray): First reference coordinate
                eta (np.ndarray): Second reference coordinate
                zeta (np.ndarray): Third reference coordinate
            Returns:
                float: Value of basis function at (xi, eta, zeta)

        gradx(xi, eta, zeta): Computes derivative w.r.t. xi
            Args:
                xi (np.ndarray): First reference coordinate
                eta (np.ndarray): Second reference coordinate
                zeta (np.ndarray): Third reference coordinate
            Returns:
                float: Partial derivative w.r.t. xi

        grady(xi, eta, zeta): Computes derivative w.r.t. eta
            Args:
                xi (np.ndarray): First reference coordinate
                eta (np.ndarray): Second reference coordinate
                zeta (np.ndarray): Third reference coordinate
            Returns:
                float: Partial derivative w.r.t. eta

        gradxx(xi, eta, zeta): Computes second derivative w.r.t. xi
            Args:
                xi (np.ndarray): First reference coordinate
                eta (np.ndarray): Second reference coordinate
                zeta (np.ndarray): Third reference coordinate
            Returns:
                float: Second partial derivative w.r.t. xi

        gradxy(xi, eta, zeta): Computes mixed derivative w.r.t. xi and eta
            Args:
                xi (np.ndarray): First reference coordinate
                eta (np.ndarray): Second reference coordinate
                zeta (np.ndarray): Third reference coordinate
            Returns:
                float: Mixed partial derivative w.r.t. xi and eta

        gradyy(xi, eta, zeta): Computes second derivative w.r.t. eta
            Args:
                xi (np.ndarray): First reference coordinate
                eta (np.ndarray): Second reference coordinate
                zeta (np.ndarray): Third reference coordinate
            Returns:
                float: Second partial derivative w.r.t. eta

    Notes:
        - All coordinate inputs (xi, eta, zeta) should be in the reference element range
        - All methods are abstract and must be implemented by derived classes
        - Implementation should ensure proper handling of 3D tensor-product bases
    """

    def __init__(self, num_shape_functions):
        self.num_shape_functions = num_shape_functions

    @abstractmethod
    def value(self, xi, eta, zeta):
        """
        Evaluates the basis function at the given xi and eta coordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: The value of the basis function at (xi, eta, zeta).
        """

    @abstractmethod
    def gradx(self, xi, eta, zeta):
        """
        Computes the partial derivative of the basis function with respect to xi.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: The partial derivative of the basis function with respect to xi.
        """

    @abstractmethod
    def grady(self, xi, eta, zeta):
        """
        Computes the partial derivative of the basis function with respect to eta.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: The partial derivative of the basis function with respect to eta.
        """

    @abstractmethod
    def gradxx(self, xi, eta, zeta):
        """
        Computes the second partial derivative of the basis function with respect to xi.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: The second partial derivative of the basis function with respect to xi.
        """

    @abstractmethod
    def gradxy(self, xi, eta, zeta):
        """
        Computes the mixed partial derivative of the basis function with respect to xi and eta.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: The mixed partial derivative of the basis function with respect to xi and eta.
        """

    @abstractmethod
    def gradyy(self, xi, eta, zeta):
        """
        Computes the second partial derivative of the basis function with respect to eta.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.
            zeta (np.ndarray): The zeta coordinate.

        Returns:
            np.ndarray: The second partial derivative of the basis function with respect to eta.
        """
