# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform).
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
The file `quadratureformulas.py` contains class for all quadrature formulas

Author: Thivin Anandh D

Changelog: Not specified

Known issues: None

Dependencies: numpy, scipy
"""

from abc import abstractmethod


class Quadratureformulas:
    """
    Defines the Quadrature Formulas for the 2D Quadrilateral elements.

    :param quad_order: The order of the quadrature.
    :type quad_order: int
    :param quad_type: The type of the quadrature.
    :type quad_type: str
    """

    def __init__(self, quad_order: int, quad_type: str, num_quad_points: int):
        """
        Constructor for the Quadratureformulas_Quad2D class.

        :param quad_order: The order of the quadrature.
        :type quad_order: int
        :param quad_type: The type of the quadrature.
        :type quad_type: str
        """
        self.quad_order = quad_order
        self.quad_type = quad_type
        self.num_quad_points = num_quad_points

    @abstractmethod
    def get_quad_values(self):
        """
        Returns the quadrature weights, xi and eta values.

        :return: A tuple containing the quadrature weights, xi values, and eta values.
        :rtype: tuple
        """

    @abstractmethod
    def get_num_quad_points(self):
        """
        Returns the number of quadrature points.

        :return: The number of quadrature points.
        :rtype: int
        """
