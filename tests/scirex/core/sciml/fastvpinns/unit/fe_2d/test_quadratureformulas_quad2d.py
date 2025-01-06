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
    assert num_quad == quad_order**2
