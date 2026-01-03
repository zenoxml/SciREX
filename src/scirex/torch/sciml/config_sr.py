# Copyright (c) 2025 Zenteiq Aitech Innovations Private Limited and
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

# Author: Diya
# Version Info: 29/Dec/2025

from dataclasses import dataclass

@dataclass
class FNOConfig:
    """Configuration for FNO model"""
    modes: int = 12
    width: int = 32
    n_layers: int = 4
    in_channels: int = 2
    out_channels: int = 1


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 200
    learning_rate: float = 5e-4
    batch_size: int = 20
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    lr_patience: int = 10
    lr_factor: float = 0.5


@dataclass
class DataConfig:
    """Configuration for data generation"""
    nx: int = 256
    nt: int = 100
    L: float = 1.0
    n_train: int = 200
    n_test: int = 40
