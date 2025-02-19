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
    File: unstacked.py
    Description: Implements the unstacked DeepONet architecture.

    Authors: Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 03/02/2025: Initial version
"""
from typing import List, Tuple, Optional, Union, Callable
from scirex.core.dl.tf_backend.networks.fcnn import FullyConnectedNetwork
from scirex.core.dl.tf_backend.mathutils import *
from scirex.core.dl.tf_backend.datautils import *
from scirex.core.sciml.deeponet.base import DeepONet


class UnstackedDeepONet(DeepONet):
    """
    Unstacked DeepONet architecture.
    """

    def __init__(
        self,
        num_sensors: int,
        num_branches: int,
        trunk_architecture: List[int],
        branch_architecture: List[int],
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """
        Initialize the unstacked DeepONet architecture.

        Args:
            num_sensors: Number of sensors
            num_branches: Number of branches
            trunk_architecture: List of integers defining the trunk architecture
            branch_architecture: List of integers defining the branch architecture
        """
        num_branch_networks = 1
        super().__init__(
            num_sensors,
            num_branches,
            trunk_architecture,
            branch_architecture,
            num_branch_networks,
            dtype,
        )
        assert (
            branch_architecture[0] == num_sensors
        ), "Input layer of branch architecture must match number of sensors"
        assert (
            branch_architecture[-1] == num_branches
        ), "Output layer of branch architecture must match number of branches"
