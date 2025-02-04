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
    File: optimizers.py
    Description: This module lets the user choose one of the various optimizers in tensorflow.

    Authors: Divij Ghose (divijghose@{iisc.ac.in})

    Version Info:
        - 03/02/2025: Initial version
"""
from typing import List, Optional, Union, Callable
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, schedules, Optimizer


def get_optimizer(
    optimizer_name: str = "Adam", learning_rate_dict: dict = {}
) -> Optimizer:
    """
    Get an optimizer object given the name and learning rate.

    Args:
        optimizer_name: Name of the optimizer
        learning_rate: Learning rate for the optimizer

    Returns:
        Optimizer object

    Notes:
          - Adam, SGD, and RMSprop are the optimizers available in this function.
          - L-BFGS will be implemented separately.
          - Only ExponentialDecay learning rate scheduler is supported as of now.


    """
    if learning_rate_dict["use_lr_scheduler"]:
        try:
            decay_rate = learning_rate_dict["decay_rate"]
            decay_steps = learning_rate_dict["decay_steps"]
            staircase = learning_rate_dict["staircase"]
        except KeyError:
            print("Learning rate scheduler parameters not found. Using default values.")
            print("-" * 50)
            print("Default learning rate scheduler parameters:")
            print("decay_rate = 0.9")
            print("decay_steps = 1000")
            print("staircase = False")
            print("-" * 50)
            decay_rate = 0.9
            decay_steps = 1000
            staircase = False
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_dict["initial_learning_rate"],
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )

        optimizer_name = optimizer_name.lower()

        if optimizer_name == "adam":
            return Adam(learning_rate=lr_schedule)
        elif optimizer_name == "sgd":
            return SGD(learning_rate=lr_schedule)
        elif optimizer_name == "rmsprop":
            return RMSprop(learning_rate=lr_schedule)
        else:
            raise ValueError("Invalid optimizer name")

    else:
        if optimizer_name == "adam":
            return Adam(learning_rate=learning_rate_dict["initial_learning_rate"])
        elif optimizer_name == "sgd":
            return SGD(learning_rate=learning_rate_dict["initial_learning_rate"])
        elif optimizer_name == "rmsprop":
            return RMSprop(learning_rate=learning_rate_dict["initial_learning_rate"])
        else:
            raise ValueError("Invalid optimizer name")


def lbfgs_optimizer():
    pass
