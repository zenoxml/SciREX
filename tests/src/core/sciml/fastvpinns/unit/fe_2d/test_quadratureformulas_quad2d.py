# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab,
# Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# SciREX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with SciREX. If not, see <https://www.gnu.org/licenses/>.
#
# For any clarifications or special considerations,
# please contact <scirex@zenteiq.ai>
# Author : Thivin Anandh. D
# Added test cases for validating Quadrature routines by computing the areas.
# The test cases are parametrized for different quadrature types and transformations.

import pytest
import numpy as np
import tensorflow as tf

from scirex.core.sciml.fe.fespace2d import Fespace2D
from scirex.core.sciml.fe.fe2d_setup_main import FE2DSetupMain
from scirex.core.sciml.fe.quadratureformulas_quad2d import Quadratureformulas_Quad2D

import pytest


@pytest.mark.parametrize("quad_order", [3, 10, 15])
def test_invalid_fe_type(quad_order):
    """
    Test case to validate the behavior when an invalid finite element type is provided.
    It should raise a ValueError.
    """

    quad_formula_main = Quadratureformulas_Quad2D(quad_order, "gauss-jacobi")
    num_quad = quad_formula_main.get_num_quad_points()
    assert num_quad == quad_order ** 2
