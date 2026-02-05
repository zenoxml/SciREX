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
    Module: fe2d_setup_main.py

    This module provides the main setup and configuration functionality for 2D 
    finite element analysis, handling basis function assignment, quadrature rules, 
    and transformations.

    Classes:
        FE2DSetupMain: Core class for configuring FE2D computations

    Dependencies:
        - basis_function_2d: Base classes for 2D basis functions
        - quadratureformulas_quad2d: Quadrature rule implementations
        - fe_transformation_2d: Geometric transformation handlers

    Key Features:
        - Flexible basis function selection
        - Multiple polynomial basis options
            * Legendre polynomials
            * Special Legendre polynomials
            * Chebyshev polynomials
            * Jacobi polynomials
        - Configurable quadrature rules
        - Support for different element transformations
        - Automatic validation of parameters
        - Element type specific handling

    Authors:
        Thivin Anandh D (https://thivinanandh.github.io)

    Version Info:
        27/Dec/2024: Initial version - Thivin Anandh D

    Notes:
        Currently supports quadrilateral elements with various polynomial 
        basis options and transformation types. The implementation focuses 
        on flexibility and extensibility for different element types.
"""

# Importing the required libraries
from .basis_function_2d import *

# import Quadrature rules
from .quadratureformulas_quad2d import *


# import base class for fe transformation
from .fe_transformation_2d import *

import numpy as np


class FE2DSetupMain:
    """
    Main configuration class for 2D finite element analysis setup.

    This class handles the configuration and initialization of finite element
    analysis components, including basis functions, quadrature rules, and
    geometric transformations.

    Attributes:
        cell_type (str): Type of finite element ('quadrilateral')
        fe_order (int): Order of finite element approximation (1 < order < 1e3)
        fe_type (str): Type of basis functions
            ('legendre', 'legendre_special', 'chebyshev_2', 'jacobi_plain')
        quad_order (int): Order of quadrature rule (>= 2)
        quad_type (str): Type of quadrature formula
        n_nodes (int): Number of nodes in the element

    Example:
        >>> setup = FE2DSetupMain(
        ...     cell_type='quadrilateral',
        ...     fe_order=2,
        ...     fe_type='legendre',
        ...     quad_order=3,
        ...     quad_type='gauss'
        ... )
        >>> basis = setup.assign_basis_function()
        >>> weights, xi, eta = setup.assign_quadrature_rules()

    Notes:
        - Supports only quadrilateral elements currently
        - Validates all input parameters for correctness
        - Provides different polynomial basis options
        - Handles both affine and bilinear transformations
        - Quadrature order must be >= 3 for accuracy
    """

    def __init__(
        self,
        cell_type: str,
        fe_order: int,
        fe_type: str,
        quad_order: int,
        quad_type: str,
    ):
        """
        Constructor for the FE2DSetupMain class.

        Args:
            cell_type (str): Type of finite element ('quadrilateral')
            fe_order (int): Order of finite element approximation (1 < order < 1e3)
            fe_type (str): Type of basis functions
                ('legendre', 'legendre_special', 'chebyshev_2', 'jacobi_plain')
            quad_order (int): Order of quadrature rule (>= 2)
            quad_type (str): Type of quadrature formula

        Raises:
            None

        Returns:
            None
        """
        self.cell_type = cell_type
        self.fe_order = fe_order
        self.fe_type = fe_type
        self.quad_order = quad_order
        self.quad_type = quad_type

        self.assign_basis_function()

    def assign_basis_function(self) -> BasisFunction2D:
        """
        Assigns the basis function based on the cell type and the fe_order.

        Args:
            None

        Returns:
            BasisFunction2D: The basis function object for the given configuration.

        Raises:
            ValueError: If the fe order is invalid or the cell type is invalid.
        """
        # check for fe order lower bound and higher bound
        if self.fe_order <= 1 or self.fe_order >= 1e3:
            print(
                f"Invalid fe order {self.fe_order} in {self.__class__.__name__} from {__name__}."
            )
            raise ValueError("fe order should be greater than 1 and less than 1e4.")

        if self.cell_type == "quadrilateral":
            self.n_nodes = 4

            # --- LEGENDRE --- #
            if self.fe_type == "legendre" or self.fe_type == "jacobi":
                # jacobi is added for backward compatibility with prev pushes
                # generally, jacobi is referred to as Legendre basis on previous iterations
                return Basis2DQNLegendre(self.fe_order**2)

            elif self.fe_type == "legendre_special":
                return Basis2DQNLegendreSpecial(self.fe_order**2)

            # ----- CHEBYSHEV ---- #
            elif self.fe_type == "chebyshev_2":
                return Basis2DQNChebyshev2(self.fe_order**2)

            # ----- PLain jacobi ---- #
            elif self.fe_type == "jacobi_plain":
                return Basis2DQNJacobi(self.fe_order**2)

            else:
                print(
                    f"Invalid fe order {self.fe_order} in {self.__class__.__name__} from {__name__}."
                )
                raise ValueError(
                    'fe order should be one of the : "legendre" , "jacobi", "legendre_special", "chebyshev_2", "jacobi_plain"'
                )

        print(
            f"Invalid cell type {self.cell_type} in {self.__class__.__name__} from {__name__}."
        )

    def assign_quadrature_rules(self):
        """
        Assigns the quadrature rule based on the quad_order.

        Args:
            None

        Returns:
            tuple: The quadrature weights, xi and eta values in a numpy array format.

        Raises:
            ValueError: If the quad_order is invalid
            ValueError: If the cell type is invalid
            ValueError: If the quad_order is not between 1 and 9999
        """
        if self.cell_type == "quadrilateral":
            if self.quad_order < 3:
                raise ValueError("Quad order should be greater than 2.")
            elif self.quad_order >= 2 and self.quad_order <= 9999:
                weights, xi, eta = Quadratureformulas_Quad2D(
                    self.quad_order, self.quad_type
                ).get_quad_values()
                return weights, xi, eta
            else:
                print(
                    f"Invalid quad order {self.quad_order} in {self.__class__.__name__} from {__name__}."
                )
                raise ValueError("Quad order should be between 1 and 9999.")

        raise ValueError(
            f"Invalid cell type {self.cell_type} in {self.__class__.__name__} from {__name__}."
        )

    def assign_fe_transformation(
        self, fe_transformation_type: str, cell_coordinates: np.ndarray
    ) -> FETransforamtion2D:
        """
        Assigns the fe transformation based on the cell type.

        Args:
            fe_transformation_type (str): Type of fe transformation ('affine', 'bilinear')
            cell_coordinates (np.ndarray): The cell coordinates

        Returns:
            FETransforamtion2D: The fe transformation object for the given configuration.

        Raises:
            ValueError: If the cell type is invalid
            ValueError: If the fe transformation type is invalid
        """
        if self.cell_type == "quadrilateral":
            if fe_transformation_type == "affine":
                return QuadAffin(cell_coordinates)
            elif fe_transformation_type == "bilinear":
                return QuadBilinear(cell_coordinates)
            else:
                raise ValueError(
                    f"Invalid fe transformation type {fe_transformation_type} in {self.__class__.__name__} from {__name__}."
                )

        else:
            raise ValueError(
                f"Invalid cell type {self.cell_type} in {self.__class__.__name__} from {__name__}."
            )
