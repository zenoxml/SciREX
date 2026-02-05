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
    Module: FE2D_Cell.py

    This module provides functionality for setting up and managing finite element 
    calculations for individual 2D cells, including basis functions, quadrature 
    rules, and transformations.

    Classes:
        FE2D_Cell: Main class for managing cell-level FE computations

    Dependencies:
        - basis_function_2d: Base classes for 2D basis functions
        - quadratureformulas_quad2d: Quadrature rules for 2D elements
        - fe2d_setup_main: Setup utilities for 2D FE calculations
        - numpy: Numerical computations

    Key Features:
        - Cell-level finite element value storage
        - Basis function evaluation at quadrature points
        - Reference to physical domain transformations
        - Gradient and derivative computations
        - Quadrature rule implementation
        - Forcing function integration
        - Support for different element types and orders

    Authors:
        Thivin Anandh D (https://thivinanandh.github.io)

    Version Info:
        27/Dec/2024: Initial version - Thivin Anandh D

    Notes:
        The implementation includes optimization for gradient calculations
        where grad_x_orig and grad_y_orig store multiplication factors
        for reference gradients to improve computational efficiency.
"""

# Importing the required libraries
import numpy as np

# Importing the required libraries
from .basis_function_2d import *

# import Quadrature rules
from .quadratureformulas_quad2d import *
from .fe2d_setup_main import *


class FE2D_Cell:
    """
    A class for managing finite element computations at the cell level.

    This class handles the storage and computation of finite element values,
    including basis functions, quadrature rules, and transformations for a
    single cell in a 2D mesh.

    Attributes:
        cell_coordinates (np.ndarray): Physical coordinates of the cell vertices
        cell_type (str): Type of the cell (e.g., 'quad', 'triangle')
        fe_order (int): Order of the finite element approximation
        fe_type (str): Type of finite element basis
        quad_order (int): Order of quadrature rule
        quad_type (str): Type of quadrature formula
        fe_transformation (str): Type of geometric transformation
        forcing_function (callable): Source term function
        basis_function (BasisFunction2D): Basis function implementation
        quad_xi (np.ndarray): Xi coordinates of quadrature points
        quad_eta (np.ndarray): Eta coordinates of quadrature points
        quad_weight (np.ndarray): Quadrature weights
        jacobian (np.ndarray): Transformation Jacobian
        basis_at_quad (np.ndarray): Basis values at quadrature points
        basis_gradx_at_quad (np.ndarray): X-derivatives at quadrature points
        basis_grady_at_quad (np.ndarray): Y-derivatives at quadrature points
        quad_actual_coordinates (np.ndarray): Physical quadrature point coordinates

    Example:
        >>> coords = np.array([[0,0], [1,0], [1,1], [0,1]])
        >>> cell = FE2D_Cell(
        ...     cell_coordinates=coords,
        ...     cell_type='quad',
        ...     fe_order=2,
        ...     fe_type='lagrange',
        ...     quad_order=3,
        ...     quad_type='gauss',
        ...     fe_transformation_type='bilinear',
        ...     forcing_function=lambda x, y: x*y
        ... )
        >>> cell.basis_at_quad  # Get basis values at quadrature points

    Notes:
        - All gradient and derivative values are stored in the reference domain
        - Jacobian and quadrature weights are combined for efficiency
        - Forcing function values are typically computed in the fespace class
        - Supports multiple types of transformations and element types
    """

    def __init__(
        self,
        cell_coordinates: np.ndarray,
        cell_type: str,
        fe_order: int,
        fe_type: str,
        quad_order: int,
        quad_type: str,
        fe_transformation_type: str,
        forcing_function,
    ):
        """
        Constructor for the FE2D_Cell class.

        Args:
            cell_coordinates (np.ndarray): Physical coordinates of the cell vertices
            cell_type (str): Type of the cell (e.g., 'quad', 'triangle')
            fe_order (int): Order of the finite element approximation
            fe_type (str): Type of finite element basis
            quad_order (int): Order of quadrature rule
            quad_type (str): Type of quadrature formula
            fe_transformation_type (str): Type of geometric transformation
            forcing_function (callable): Source term function

        Returns:
            None
        """
        self.cell_coordinates = cell_coordinates
        self.cell_type = cell_type
        self.fe_order = fe_order
        self.fe_type = fe_type
        self.quad_order = quad_order
        self.quad_type = quad_type
        self.fe_transformation = fe_transformation_type
        self.forcing_function = forcing_function

        # Basis function Class
        self.basis_function = None

        # Quadrature Values
        self.quad_xi = None
        self.quad_eta = None
        self.quad_weight = None
        self.jacobian = None
        self.mult = None

        # FE Values
        self.basis_at_quad = None
        self.basis_gradx_at_quad = None
        self.basis_grady_at_quad = None
        self.basis_gradxy_at_quad = None
        self.basis_gradxx_at_quad = None
        self.basis_gradyy_at_quad = None

        # Quadrature Coordinates
        self.quad_actual_coordinates = None

        # Forcing function values at the quadrature points
        self.forcing_at_quad = None

        # FE Transformation Class
        self.fetransformation = None

        # get instance of the FE_setup class
        self.fe_setup = FE2DSetupMain(
            cell_type=self.cell_type,
            fe_order=self.fe_order,
            fe_type=self.fe_type,
            quad_order=self.quad_order,
            quad_type=self.quad_type,
        )

        # Call the function to assign the basis function
        self.assign_basis_function()

        # Assign the quadrature points and weights
        self.assign_quadrature()

        # Assign the FE Transformation
        self.assign_fe_transformation()

        # calculate mult -> quadrature weights * Jacobian
        self.assign_quad_weights_and_jacobian()

        # Calculate the basis function values at the quadrature points
        self.assign_basis_values_at_quadrature_points()

        # calculate the actual coordinates of the quadrature points
        self.assign_quadrature_coordinates()

        # Calculate the forcing function values at the actual quadrature points
        # NOTE : The function is just for printing the shape of the force matrix, the
        # actual calculation is performed on the fespace class
        self.assign_forcing_term(self.forcing_function)

        # # print the values
        # print("============================================================================")
        # print("Cell Co-ord : ", self.cell_coordinates)
        # print("Basis function values at the quadrature points: \n", self.basis_at_quad / self.mult)
        # print("Basis function gradx at the quadrature points: \n", self.basis_gradx_at_quad)
        # print("Basis function grady at the quadrature points: \n", self.basis_grady_at_quad)
        # print("Forcing function values at the quadrature points: \n", self.forcing_at_quad)

        # grad_x = np.array([5,6,7,8])
        # grad_y = np.array([1,2,3,4])

        # pde = np.matmul(self.basis_gradx_at_quad, grad_x.reshape(-1,1)) + np.matmul(self.basis_grady_at_quad, grad_y.reshape(-1,1))
        # print("PDE values at the quadrature points: \n", pde)

    def assign_basis_function(self) -> BasisFunction2D:
        """
        Assigns the basis function class based on the cell type and the FE order.

        Args:
            None

        Returns:
            BasisFunction2D: The basis function class for the given cell type and FE order.
        """
        self.basis_function = self.fe_setup.assign_basis_function()

    def assign_quadrature(self) -> None:
        """
        Assigns the quadrature points and weights based on the cell type and the quadrature order.

        Args:
            None

        Returns:
            None
        """
        self.quad_weight, self.quad_xi, self.quad_eta = (
            self.fe_setup.assign_quadrature_rules()
        )

    def assign_fe_transformation(self) -> None:
        """
        Assigns the FE Transformation class based on the cell type and the FE order.

        This method assigns the appropriate FE Transformation class based on the cell type and the FE order.
        It sets the cell coordinates for the FE Transformation and obtains the Jacobian of the transformation.

        Args:
            None

        Returns:
            None
        """
        self.fetransformation = self.fe_setup.assign_fe_transformation(
            self.fe_transformation, self.cell_coordinates
        )
        # Sets cell co-ordinates for the FE Transformation
        self.fetransformation.set_cell()

        # obtains the Jacobian of the transformation
        self.jacobian = self.fetransformation.get_jacobian(
            self.quad_xi, self.quad_eta
        ).reshape(-1, 1)

    def assign_basis_values_at_quadrature_points(self) -> None:
        """
        Assigns the basis function values at the quadrature points.

        This method calculates the values of the basis functions and their gradients at the quadrature points.
        The basis function values are stored in `self.basis_at_quad`, while the gradients are stored in
        `self.basis_gradx_at_quad`, `self.basis_grady_at_quad`, `self.basis_gradxy_at_quad`,
        `self.basis_gradxx_at_quad`, and `self.basis_gradyy_at_quad`.

        Args:
            None

        Returns:
            None
        """
        self.basis_at_quad = []
        self.basis_gradx_at_quad = []
        self.basis_grady_at_quad = []
        self.basis_gradxy_at_quad = []
        self.basis_gradxx_at_quad = []
        self.basis_gradyy_at_quad = []

        self.basis_at_quad = self.basis_function.value(self.quad_xi, self.quad_eta)

        # For Gradients we need to perform a transformation to the original cell
        grad_x_ref = self.basis_function.gradx(self.quad_xi, self.quad_eta)
        grad_y_ref = self.basis_function.grady(self.quad_xi, self.quad_eta)

        grad_x_orig, grad_y_orig = self.fetransformation.get_orig_from_ref_derivative(
            grad_x_ref, grad_y_ref, self.quad_xi, self.quad_eta
        )

        self.basis_gradx_at_quad = grad_x_orig
        self.basis_grady_at_quad = grad_y_orig

        self.basis_gradx_at_quad_ref = grad_x_ref
        self.basis_grady_at_quad_ref = grad_y_ref

        # get the double derivatives of the basis functions ( ref co-ordinates )
        grad_xx_ref = self.basis_function.gradxx(self.quad_xi, self.quad_eta)
        grad_xy_ref = self.basis_function.gradxy(self.quad_xi, self.quad_eta)
        grad_yy_ref = self.basis_function.gradyy(self.quad_xi, self.quad_eta)

        # get the double derivatives of the basis functions ( orig co-ordinates )
        grad_xx_orig, grad_xy_orig, grad_yy_orig = (
            self.fetransformation.get_orig_from_ref_second_derivative(
                grad_xx_ref, grad_xy_ref, grad_yy_ref, self.quad_xi, self.quad_eta
            )
        )

        # = the value
        self.basis_gradxy_at_quad = grad_xy_orig
        self.basis_gradxx_at_quad = grad_xx_orig
        self.basis_gradyy_at_quad = grad_yy_orig

        # Multiply each row with the quadrature weights
        # Basis at Quad - n_test * N_quad
        self.basis_at_quad = self.basis_at_quad * self.mult
        self.basis_gradx_at_quad = self.basis_gradx_at_quad * self.mult
        self.basis_grady_at_quad = self.basis_grady_at_quad * self.mult
        self.basis_gradxy_at_quad = self.basis_gradxy_at_quad * self.mult
        self.basis_gradxx_at_quad = self.basis_gradxx_at_quad * self.mult
        self.basis_gradyy_at_quad = self.basis_gradyy_at_quad * self.mult

    def assign_quad_weights_and_jacobian(self) -> None:
        """
        Assigns the quadrature weights and the Jacobian of the transformation.

        This method calculates and assigns the quadrature weights and the Jacobian of the transformation
        for the current cell. The quadrature weights are multiplied by the flattened Jacobian array
        and stored in the `mult` attribute of the class.

        Args:
            None

        Returns:
            None
        """
        self.mult = self.quad_weight * self.jacobian.flatten()

    def assign_quadrature_coordinates(self) -> None:
        """
        Assigns the actual coordinates of the quadrature points.

        This method calculates the actual coordinates of the quadrature points based on the given Xi and Eta values.
        The Xi and Eta values are obtained from the `quad_xi` and `quad_eta` attributes of the class.
        The calculated coordinates are stored in the `quad_actual_coordinates` attribute as a NumPy array.

        Args:
            None

        Returns:
            None
        """
        actual_co_ord = []
        for xi, eta in zip(self.quad_xi, self.quad_eta):
            actual_co_ord.append(self.fetransformation.get_original_from_ref(xi, eta))

        self.quad_actual_coordinates = np.array(actual_co_ord)

    def assign_forcing_term(self, forcing_function) -> None:
        """
        Assigns the forcing function values at the quadrature points.

        This function computes the values of the forcing function at the quadrature points
        and assigns them to the `forcing_at_quad` attribute of the FE2D_Cell object.

        Args:
            forcing_function (callable): The forcing function to be integrated

        Returns:
            None

        Notes:
            - The final shape of `forcing_at_quad` will be N_shape_functions x 1.
            - This function is for backward compatibility with old code and currently assigns
              the values as zeros. The actual calculation is performed in the fespace class.
        """
        # get number of shape functions
        n_shape_functions = self.basis_function.num_shape_functions

        # Loop over all the basis functions and compute the integral
        f_integral = np.zeros((n_shape_functions, 1), dtype=np.float64)

        # The above code is for backward compatibility with old code. this function will just assign the values as zeros
        # the actual calculation is performed in the fespace class

        # for i in range(n_shape_functions):
        #     val = 0
        #     for q in range(self.basis_at_quad.shape[1]):
        #         x = self.quad_actual_coordinates[q, 0]
        #         y = self.quad_actual_coordinates[q, 1]
        #         # print("f_values[q] = ",f_values[q])

        #         # the JAcobian and the quadrature weights are pre multiplied to the basis functions
        #         val +=  ( self.basis_at_quad[i, q] ) * self.forcing_function(x, y)
        #         # print("val = ", val)

        #     f_integral[i] = val

        self.forcing_at_quad = f_integral
