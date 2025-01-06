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

"""Implementation of Bilinear Transformation for Quadrilateral Elements.

This module provides functionality for bilinear transformations of quadrilateral 
elements in finite element analysis. It implements mapping between reference and 
physical elements based on the ParMooN project's methodology.

Key functionalities:
    - Reference to physical coordinate mapping using bilinear transformation
    - Jacobian computation for bilinear elements
    - First-order derivatives transformation
    - Limited second-order derivatives transformation

The implementation allows for more general quadrilateral elements compared to
affine transformations, by using bilinear mapping functions. This enables
handling of non-parallelogram quadrilateral elements while maintaining 
geometric consistency.

Key classes:
    - QuadBilinear: Main class implementing bilinear transformation for quads

Note:
    Second derivative calculations are currently not fully implemented.
    This implementation is specifically referenced from ParMooN project's
    QuadBilineare.C file with adaptations for Python and SciREX framework.

References:
    [1] ParMooN Project: ParMooN/FiniteElement/QuadBilinear.C

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version:
    27/Dec/2024: Initial version - Thivin Anandh D
"""

import numpy as np
from .fe_transformation_2d import FETransforamtion2D


class QuadBilinear(FETransforamtion2D):
    """
    Implements bilinear transformation for quadrilateral elements.

    This class provides methods to transform between reference and physical
    quadrilateral elements using bilinear mapping. It handles coordinate
    transformations, Jacobian computations, and derivative mappings for more
    general quadrilateral elements than affine transformations.

    Attributes:
        co_ordinates: Array of physical element vertex coordinates
            Shape: (4, 2) for 2D quadrilateral
        x0, x1, x2, x3: x-coordinates of vertices
        y0, y1, y2, y3: y-coordinates of vertices
        xc0, xc1, xc2, xc3: x-coordinate transformation coefficients
        yc0, yc1, yc2, yc3: y-coordinate transformation coefficients
        detjk: Determinant of the Jacobian matrix

    Example:
        >>> coords = np.array([[0,0], [1,0], [1.2,1], [0.2,1.1]])
        >>> quad = QuadBilinear(coords)
        >>> ref_point = np.array([0.5, 0.5])
        >>> physical_point = quad.get_original_from_ref(*ref_point)

    Note:
        - Implementation assumes counterclockwise vertex ordering
        - Second derivatives computation is not fully implemented
        - Jacobian is computed point-wise due to non-constant nature
        of bilinear transformation

    References:
        [1] ParMooN Project: QuadBilineare.C implementation
    """

    def __init__(self, co_ordinates: np.ndarray) -> None:
        """
        Constructor for the QuadBilinear class.

        Args:
            co_ordinates: Array of physical element vertex coordinates
                Shape: (4, 2) for 2D quadrilateral

        Returns:
            None
        """
        self.co_ordinates = co_ordinates
        self.set_cell()
        self.detjk = None  # Jacobian of the transformation

    def set_cell(self):
        """
        Set the cell coordinates, which will be used as intermediate values to calculate the Jacobian and actual values.

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

        self.xc0 = (self.x0 + self.x1 + self.x2 + self.x3) * 0.25
        self.xc1 = (-self.x0 + self.x1 + self.x2 - self.x3) * 0.25
        self.xc2 = (-self.x0 - self.x1 + self.x2 + self.x3) * 0.25
        self.xc3 = (self.x0 - self.x1 + self.x2 - self.x3) * 0.25

        self.yc0 = (self.y0 + self.y1 + self.y2 + self.y3) * 0.25
        self.yc1 = (-self.y0 + self.y1 + self.y2 - self.y3) * 0.25
        self.yc2 = (-self.y0 - self.y1 + self.y2 + self.y3) * 0.25
        self.yc3 = (self.y0 - self.y1 + self.y2 - self.y3) * 0.25

    def get_original_from_ref(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the original coordinates from the reference coordinates.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: Returns the transformed original coordinates from the reference coordinates.
        """
        x = self.xc0 + self.xc1 * xi + self.xc2 * eta + self.xc3 * xi * eta
        y = self.yc0 + self.yc1 * xi + self.yc2 * eta + self.yc3 * xi * eta

        return np.array([x, y], dtype=np.float64)

    def get_jacobian(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        This method returns the Jacobian of the transformation.

        Args:
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: Returns the Jacobian of the transformation.
        """
        self.detjk = abs(
            (self.xc1 + self.xc3 * eta) * (self.yc2 + self.yc3 * xi)
            - (self.xc2 + self.xc3 * xi) * (self.yc1 + self.yc3 * eta)
        )
        return self.detjk

    def get_orig_from_ref_derivative(
        self,
        ref_gradx: np.ndarray,
        ref_grady: np.ndarray,
        xi: np.ndarray,
        eta: np.ndarray,
    ) -> np.ndarray:
        """
        This method returns the derivatives of the original coordinates with respect to the reference coordinates.

        Args:
            ref_gradx (np.ndarray): The derivative of the xi coordinate in the reference element.
            ref_grady (np.ndarray): The derivative of the eta coordinate in the reference element.
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Returns:
            np.ndarray: The derivatives of the original coordinates [x, y] with respect to the reference coordinates.

        """
        n_test = ref_gradx.shape[0]
        gradx_orig = np.zeros(ref_gradx.shape, dtype=np.float64)
        grady_orig = np.zeros(ref_grady.shape, dtype=np.float64)

        for j in range(n_test):
            Xi = xi
            Eta = eta
            rec_detjk = 1 / (
                (self.xc1 + self.xc3 * Eta) * (self.yc2 + self.yc3 * Xi)
                - (self.xc2 + self.xc3 * Xi) * (self.yc1 + self.yc3 * Eta)
            )
            gradx_orig[j] = (
                (self.yc2 + self.yc3 * Xi) * ref_gradx[j]
                - (self.yc1 + self.yc3 * Eta) * ref_grady[j]
            ) * rec_detjk
            grady_orig[j] = (
                -(self.xc2 + self.xc3 * Xi) * ref_gradx[j]
                + (self.xc1 + self.xc3 * Eta) * ref_grady[j]
            ) * rec_detjk

        return gradx_orig, grady_orig

    def get_orig_from_ref_second_derivative(
        self,
        grad_xx_ref: np.ndarray,
        grad_xy_ref: np.ndarray,
        grad_yy_ref: np.ndarray,
        xi: np.ndarray,
        eta: np.ndarray,
    ):
        """
        This method returns the second derivatives of the original coordinates with respect to the reference coordinates.

        Args:
            grad_xx_ref (np.ndarray): The second derivative of the xi coordinate in the reference element.
            grad_xy_ref (np.ndarray): The second derivative of the xi and eta coordinates in the reference element.
            grad_yy_ref (np.ndarray): The second derivative of the eta coordinate in the reference element.
            xi (np.ndarray): The xi coordinate.
            eta (np.ndarray): The eta coordinate.

        Note:
            Second derivative calculations are not fully implemented in this method. Needs further development.
        """
        # print(" Error : Second Derivative not implemented -- Ignore this error, if second derivative is not required ")
        return grad_xx_ref, grad_xy_ref, grad_yy_ref
