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
    File: base.py
    Description: Implements the base DeepONet architecture.

    Authors: Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 03/02/2025: Initial version
"""
from typing import List, Optional, Union, Callable, Tuple
from scirex.core.dl.tf_backend.networks.fcnn import FullyConnectedNetwork
from scirex.core.dl.tf_backend.mathutils import *
from scirex.core.dl.tf_backend.datautils import *


class DeepONet:
    """
    Base DeepONet architecture.
    """

    def __init__(
        self,
        num_sensors: int,
        num_branches: int,
        trunk_architecture: List[int],
        branch_architecture: List[int],
        num_branch_networks: int,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """
        Initialize the DeepONet architecture.

        Args:
            num_sensors: Number of sensors
            num_branches: Number of branches
            trunk_architecture: List of integers defining the trunk architecture
            branch_architecture: List of integers defining the branch architecture
        """
        self.num_sensors = num_sensors
        self.num_branches = num_branches
        assert self.num_sensors > 0, "Number of sensors must be greater than 0"
        assert self.num_branches > 0, "Number of branches must be greater than 0"

        self.trunk_architecture = trunk_architecture
        assert (
            trunk_architecture[-1] == num_branches
        ), "Trunk output neurons must match num_branches"
        self.branch_architecture = branch_architecture

        self.num_branch_networks = num_branch_networks

        self.dtype = dtype

        self.trunk = FullyConnectedNetwork(
            architecture=self.trunk_architecture, dtype=self.dtype
        )
        self.branches = [
            FullyConnectedNetwork(
                architecture=self.branch_architecture, dtype=self.dtype
            )
            for _ in range(self.num_branch_networks)
        ]

    def __call__(
        self,
        u: tf.Tensor,
        y: tf.Tensor,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Forward pass through the DeepONet.

        Args:
            x: Input tensor
            sensor_id: Sensor ID

        Returns:
            Tuple containing the output of the trunk and a list of outputs from the branches
        """
        trunk_output = self.trunk(u)
        branch_outputs = [branch(y) for branch in self.branches]

        trunk_output = reshape(trunk_output, [-1, 1])

        if self.num_branch_networks == 1:
            # branch_outputs = reshape(branch_outputs, [-1, 1])
            return dot(trunk_output, branch_outputs)
        elif self.num_branch_networks > 1:
            branch_outputs = concat(branch_outputs, axis=1)
            branch_outputs = reshape(branch_outputs, [-1, 1])
            return dot(trunk_output, branch_outputs)

    def get_config(self):
        """
        Get the configuration of the DeepONet.

        Returns:
            Dictionary containing the configuration of the DeepONet
        """
        return {
            "num_sensors": self.num_sensors,
            "num_branches": self.num_branches,
            "trunk_architecture": self.trunk_architecture,
            "branch_architecture": self.branch_architecture,
        }

    def get_weights(self):
        """
        Get the weights of the DeepONet.

        Returns:
            List of weights for the trunk and branches
        """
        weights = [self.trunk.get_weights()]
        weights.extend([branch.get_weights() for branch in self.branches])

        return weights

    def train(self, x: tf.Tensor, y: tf.Tensor, epochs: int, batch_size: int):
        """
        Train the DeepONet.

        Args:
            x: Input tensor
            y: Output tensor
            epochs: Number of epochs
            batch_size: Batch size
        """
        return

    def print_summary(self):
        """
        Print a summary of the DeepONet architecture.
        """
        print("DeepONet architecture:")
        print("-" * 50)
        print("Trunk architecture:")
        print("-" * 50)
        print(self.trunk.summary())
        print("\nBranch architecture:\n")
        print("-" * 50)
        for i, branch in enumerate(self.branches):
            print(f"Branch {i+1}:")
            print("-" * 50)
            print(branch.summary())
            print("-" * 50)
            print("\n")
        print("\n")


if __name__ == "__main__":
    # Test DeepONet
    num_sensors = 3
    num_branches = 4
    trunk_architecture = [num_sensors, 10, 10, num_branches]
    branch_architecture = [num_sensors, 10, 10, num_branches]
    num_branch_networks = 4

    deeponet = DeepONet(
        num_sensors=num_sensors,
        num_branches=num_branches,
        trunk_architecture=trunk_architecture,
        branch_architecture=branch_architecture,
        num_branch_networks=num_branch_networks,
    )

    deeponet.print_summary()
