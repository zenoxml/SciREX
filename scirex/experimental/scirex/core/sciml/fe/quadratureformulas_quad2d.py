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

"""Quadrature Formula Implementation for 2D Quadrilateral Elements.

This module implements numerical integration formulas for 2D quadrilateral elements,
providing both Gauss-Legendre and Gauss-Jacobi quadrature schemes. The implementation
focuses on accurate numerical integration required for finite element computations.

Key functionalities:
    - Gauss-Legendre quadrature for quadrilateral elements
    - Gauss-Jacobi quadrature with Lobatto points
    - Tensor product based 2D quadrature point generation
    - Weight computation for various quadrature orders

The implementation provides:
    - Flexible quadrature order selection
    - Multiple quadrature schemes
    - Efficient tensor product based computations
    - Automated weight and point generation

Key classes:
    - Quadratureformulas_Quad2D: Main class for 2D quadrature computations

Dependencies:
    - numpy: For numerical computations
    - scipy.special: For special function evaluations (roots, weights)
    - scipy.special.orthogonal: For orthogonal polynomial computations

Note:
    The implementation assumes tensor-product based quadrature rules for
    2D elements. Specialized non-tensor product rules are not included.

References:
    [1] Karniadakis, G., & Sherwin, S. (2013). Spectral/hp Element 
        Methods for Computational Fluid Dynamics. Oxford University Press.

    [2] Kharazmi - hp-VPINNs github repository

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version:
    27/Dec/2024: Initial version - Thivin Anandh D

"""

import numpy as np
from scipy.special import roots_legendre, roots_jacobi, jacobi, gamma
from scipy.special import legendre
from scipy.special import eval_legendre, legendre

from .quadratureformulas import Quadratureformulas


class Quadratureformulas_Quad2D(Quadratureformulas):
    """Implements quadrature formulas for 2D quadrilateral elements.

    This class provides methods to compute quadrature points and weights for
    2D quadrilateral elements using either Gauss-Legendre or Gauss-Jacobi
    quadrature schemes. The implementation uses tensor products of 1D rules.

    Attributes:
        quad_order: Order of quadrature rule
        quad_type: Type of quadrature ('gauss-legendre' or 'gauss-jacobi')
        num_quad_points: Total number of quadrature points (quad_order^2)
        xi_quad: x-coordinates of quadrature points in reference element
        eta_quad: y-coordinates of quadrature points in reference element
        quad_weights: Weights for each quadrature point

    Example:
        >>> quad = Quadratureformulas_Quad2D(quad_order=3, quad_type='gauss-legendre')
        >>> weights, xi, eta = quad.get_quad_values()
        >>> n_points = quad.get_num_quad_points()

    Note:
        - Gauss-Legendre points are optimal for polynomial integrands
        - Gauss-Jacobi points include element vertices (useful for certain FEM applications)
        - All computations are performed in the reference element [-1,1]×[-1,1]

    """

    def __init__(self, quad_order: int, quad_type: str):
        """
        Constructor for the Quadratureformulas_Quad2D class.

        Args:
            quad_order: Order of quadrature rule
            quad_type: Type of quadrature ('gauss-legendre' or 'gauss-jacobi')

        Returns:
            None

        Raises:
            ValueError: If the quadrature type is not supported.
        """
        # initialize the super class
        super().__init__(
            quad_order=quad_order,
            quad_type=quad_type,
            num_quad_points=quad_order * quad_order,
        )

        # Calculate the Gauss-Legendre quadrature points and weights for 1D
        # nodes_1d, weights_1d = roots_jacobi(self.quad_order, 1, 1)

        quad_type = self.quad_type

        if quad_type == "gauss-legendre":
            # Commented out by THIVIN -  to Just use legendre quadrature points as it is
            # if quad_order == 2:
            #     nodes_1d = np.array([-1, 1])
            #     weights_1d = np.array([1, 1])
            # else:
            nodes_1d, weights_1d = np.polynomial.legendre.leggauss(
                quad_order
            )  # Interior points
            # nodes_1d = np.concatenate(([-1, 1], nodes_1d))
            # weights_1d = np.concatenate(([1, 1], weights_1d))

            # Generate the tensor outer product of the nodes
            xi_quad, eta_quad = np.meshgrid(nodes_1d, nodes_1d)
            xi_quad = xi_quad.flatten()
            eta_quad = eta_quad.flatten()

            # Multiply the weights accordingly for 2D
            quad_weights = (weights_1d[:, np.newaxis] * weights_1d).flatten()

            # Assign the values
            self.xi_quad = xi_quad
            self.eta_quad = eta_quad
            self.quad_weights = quad_weights

        elif quad_type == "gauss-jacobi":

            def GaussJacobiWeights(Q: int, a, b):
                [X, W] = roots_jacobi(Q, a, b)
                return [X, W]

            def jacobi_wrapper(n, a, b, x):

                x = np.array(x, dtype=np.float64)

                return jacobi(n, a, b)(x)

            # Weight coefficients
            def GaussLobattoJacobiWeights(Q: int, a, b):
                W = []
                X = roots_jacobi(Q - 2, a + 1, b + 1)[0]
                if a == 0 and b == 0:
                    W = 2 / ((Q - 1) * (Q) * (jacobi_wrapper(Q - 1, 0, 0, X) ** 2))
                    Wl = 2 / ((Q - 1) * (Q) * (jacobi_wrapper(Q - 1, 0, 0, -1) ** 2))
                    Wr = 2 / ((Q - 1) * (Q) * (jacobi_wrapper(Q - 1, 0, 0, 1) ** 2))
                else:
                    W = (
                        2 ** (a + b + 1)
                        * gamma(a + Q)
                        * gamma(b + Q)
                        / (
                            (Q - 1)
                            * gamma(Q)
                            * gamma(a + b + Q + 1)
                            * (jacobi_wrapper(Q - 1, a, b, X) ** 2)
                        )
                    )
                    Wl = (
                        (b + 1)
                        * 2 ** (a + b + 1)
                        * gamma(a + Q)
                        * gamma(b + Q)
                        / (
                            (Q - 1)
                            * gamma(Q)
                            * gamma(a + b + Q + 1)
                            * (jacobi_wrapper(Q - 1, a, b, -1) ** 2)
                        )
                    )
                    Wr = (
                        (a + 1)
                        * 2 ** (a + b + 1)
                        * gamma(a + Q)
                        * gamma(b + Q)
                        / (
                            (Q - 1)
                            * gamma(Q)
                            * gamma(a + b + Q + 1)
                            * (jacobi_wrapper(Q - 1, a, b, 1) ** 2)
                        )
                    )
                W = np.append(W, Wr)
                W = np.append(Wl, W)
                X = np.append(X, 1)
                X = np.append(-1, X)
                return [X, W]

            # get quadrature points and weights in 1D
            x, w = GaussLobattoJacobiWeights(self.quad_order, 0, 0)

            # Generate the tensor outer product of the nodes
            xi_quad, eta_quad = np.meshgrid(x, x)
            xi_quad = xi_quad.flatten()
            eta_quad = eta_quad.flatten()

            # Multiply the weights accordingly for 2D
            quad_weights = (w[:, np.newaxis] * w).flatten()

            # Assign the values
            self.xi_quad = xi_quad
            self.eta_quad = eta_quad
            self.quad_weights = quad_weights

        else:
            print("Supported quadrature types are: gauss-legendre, gauss-jacobi")
            print(
                f"Invalid quadrature type {quad_type} in {self.__class__.__name__} from {__name__}."
            )
            raise ValueError("Quadrature type not supported.")

    def get_quad_values(self):
        """
        Returns the quadrature weights, xi and eta values.

        Args:
            None

        Returns:
            tuple: The quadrature weights, xi and eta values in a numpy array format
        """
        return self.quad_weights, self.xi_quad, self.eta_quad

    def get_num_quad_points(self):
        """
        Returns the number of quadrature points.

        Args:
            None

        Returns:
            int: The number of quadrature points
        """
        return self.num_quad_points
