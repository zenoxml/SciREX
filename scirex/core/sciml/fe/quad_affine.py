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

"""Implementation of Affine Transformation for Quadrilateral Elements.

This module provides functionality for affine transformations of quadrilateral elements
in finite element analysis. It implements mapping between reference and physical elements
based on the ParMooN project's methodology.

Key functionalities:
    - Reference to physical coordinate mapping
    - Jacobian computation
    - First-order derivatives transformation
    - Second-order derivatives transformation

The implementation follows standard finite element mapping techniques with
focus on quadrilateral elements. The transformations maintain geometric
consistency and numerical accuracy required for FEM computations.

Key classes:
    - QuadAffin: Main class implementing affine transformation for quads

Note:
    This implementation is specifically referenced from ParMooN project's
    QuadAffine.C file with adaptations for Python and SciREX framework.

References:
    [1] ParMooN Project: ParMooN/FiniteElement/QuadAffine.C

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version:
    27/Dec/2024: Initial version - Thivin Anandh D
"""

import numpy as np
from .fe_transformation_2d import FETransforamtion2D


class QuadAffin(FETransforamtion2D):
    """
    Implements affine transformation for quadrilateral elements.

    This class provides methods to transform between reference and physical
    quadrilateral elements using affine mapping. It handles coordinate
    transformations, Jacobian computations, and derivative mappings.

    Attributes:
        co_ordinates: Array of physical element vertex coordinates
            Shape: (4, 2) for 2D quadrilateral
        x0, x1, x2, x3: x-coordinates of vertices
        y0, y1, y2, y3: y-coordinates of vertices
        xc0, xc1, xc2: x-coordinate transformation coefficients
        yc0, yc1, yc2: y-coordinate transformation coefficients
        detjk: Determinant of the Jacobian
        rec_detjk: Reciprocal of Jacobian determinant

    Example:
        >>> coords = np.array([[0,0], [1,0], [1,1], [0,1]])
        >>> quad = QuadAffin(coords)
        >>> ref_point = np.array([0.5, 0.5])
        >>> physical_point = quad.get_original_from_ref(*ref_point)

    Note:
        The implementation assumes counterclockwise vertex ordering and
        non-degenerate quadrilateral elements.

    References:
        [1] ParMooN Project: QuadAffine.C implementation
    """

    def __init__(self, co_ordinates: np.ndarray) -> None:
        """
        Constructor for the QuadAffin class.

        Args:
            co_ordinates: Array of physical element vertex coordinates
                Shape: (4, 2) for 2D quadrilateral

        Returns:
            None
        """
        self.co_ordinates = co_ordinates
        self.set_cell()
        self.get_jacobian(
            0, 0
        )  # 0,0 is just a dummy value # this sets the jacobian and the inverse of the jacobian

    def set_cell(self):
        """
        Set the cell coordinates, which will be used to calculate the Jacobian and actual values.

        Args:
            None

        Returns:
            None
        """

        self.x0 = self.co_ordinates[0][0]
        self.x1 = self.co_ordinates[1][0]
        self.x2 = self.co_ordinates[2][0]
        self.x3 = self.co_ordinates[3][0]

        # get the y-coordinates of the cell
        self.y0 = self.co_ordinates[0][1]
        self.y1 = self.co_ordinates[1][1]
        self.y2 = self.co_ordinates[2][1]
        self.y3 = self.co_ordinates[3][1]

        self.xc0 = (self.x1 + self.x3) * 0.5
        self.xc1 = (self.x1 - self.x0) * 0.5
        self.xc2 = (self.x3 - self.x0) * 0.5

        self.yc0 = (self.y1 + self.y3) * 0.5
        self.yc1 = (self.y1 - self.y0) * 0.5
        self.yc2 = (self.y3 - self.y0) * 0.5

    def get_original_from_ref(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Returns the original coordinates from the reference coordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The transformed original coordinates from the reference coordinates.
        """
        x = self.xc0 + self.xc1 * xi + self.xc2 * eta
        y = self.yc0 + self.yc1 * xi + self.yc2 * eta

        return np.array([x, y])

    def get_jacobian(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Returns the Jacobian of the transformation.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The Jacobian of the transformation.
        """
        self.detjk = self.xc1 * self.yc2 - self.xc2 * self.yc1
        self.rec_detjk = 1 / self.detjk

        return abs(self.detjk)

    def get_orig_from_ref_derivative(self, ref_gradx, ref_grady, xi, eta):
        """
        Returns the derivatives of the original coordinates with respect to the reference coordinates.

        Args:
            ref_gradx (np.ndarray): The reference gradient in the x-direction.
            ref_grady (np.ndarray): The reference gradient in the y-direction.
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            tuple: The derivatives of the original coordinates with respect to the reference coordinates.
        """
        gradx_orig = np.zeros(ref_gradx.shape)
        grady_orig = np.zeros(ref_grady.shape)

        for i in range(ref_gradx.shape[0]):
            gradx_orig[i] = (
                self.yc2 * ref_gradx[i] - self.yc1 * ref_grady[i]
            ) * self.rec_detjk
            grady_orig[i] = (
                -self.xc2 * ref_gradx[i] + self.xc1 * ref_grady[i]
            ) * self.rec_detjk

        return gradx_orig, grady_orig

    def get_orig_from_ref_second_derivative(
        self, grad_xx_ref, grad_xy_ref, grad_yy_ref, xi, eta
    ):
        """
        Returns the second derivatives (xx, xy, yy) of the original coordinates with respect to the reference coordinates.

        Args:
            grad_xx_ref (np.ndarray): The reference second derivative in the x-direction.
            grad_xy_ref (np.ndarray): The reference second derivative in the xy-direction.
            grad_yy_ref (np.ndarray): The reference second derivative in the y-direction.
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            tuple: The second derivatives (xx, xy, yy) of the original coordinates with respect to the reference coordinates.
        """
        GeoData = np.zeros((3, 3))
        Eye = np.identity(3)

        # Populate GeoData (assuming xc1, xc2, yc1, yc2 are defined)
        GeoData[0, 0] = self.xc1 * self.xc1
        GeoData[0, 1] = 2 * self.xc1 * self.yc1
        GeoData[0, 2] = self.yc1 * self.yc1
        GeoData[1, 0] = self.xc1 * self.xc2
        GeoData[1, 1] = self.yc1 * self.xc2 + self.xc1 * self.yc2
        GeoData[1, 2] = self.yc1 * self.yc2
        GeoData[2, 0] = self.xc2 * self.xc2
        GeoData[2, 1] = 2 * self.xc2 * self.yc2
        GeoData[2, 2] = self.yc2 * self.yc2

        # solve the linear system
        solution = np.linalg.solve(GeoData, Eye)

        # generate empty arrays for the original second derivatives
        grad_xx_orig = np.zeros(grad_xx_ref.shape)
        grad_xy_orig = np.zeros(grad_xy_ref.shape)
        grad_yy_orig = np.zeros(grad_yy_ref.shape)

        for j in range(grad_xx_ref.shape[0]):
            r20 = grad_xx_ref[j]
            r11 = grad_xy_ref[j]
            r02 = grad_yy_ref[j]

            grad_xx_orig[j] = (
                solution[0, 0] * r20 + solution[0, 1] * r11 + solution[0, 2] * r02
            )
            grad_xy_orig[j] = (
                solution[1, 0] * r20 + solution[1, 1] * r11 + solution[1, 2] * r02
            )
            grad_yy_orig[j] = (
                solution[2, 0] * r20 + solution[2, 1] * r11 + solution[2, 2] * r02
            )

        return grad_xx_orig, grad_xy_orig, grad_yy_orig
