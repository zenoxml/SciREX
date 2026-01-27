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
Optimizer Configuration Module

This module contains configuration classes for optimizers, schedulers,
and loss functions used in training neural operator models.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers.
    
    Attributes:
        scheduler_type: Type of scheduler to use.
            Options: "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", 
                     "ExponentialLR", "OneCycleLR", "None"
        step_size: Period of learning rate decay (for StepLR).
        gamma: Multiplicative factor of learning rate decay.
        patience: Number of epochs with no improvement for ReduceLROnPlateau.
        min_lr: Minimum learning rate.
        T_max: Maximum number of iterations for CosineAnnealingLR.
        warmup_epochs: Number of warmup epochs.
        warmup_start_lr: Starting learning rate for warmup.
    """
    scheduler_type: Literal[
        "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", 
        "ExponentialLR", "OneCycleLR", "None"
    ] = "ReduceLROnPlateau"
    step_size: int = 60
    gamma: float = 0.5
    patience: int = 100
    min_lr: float = 1e-8
    T_max: Optional[int] = None
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-7


@dataclass
class LossConfig:
    """Configuration for loss functions.
    
    Attributes:
        training_loss: Loss function for training.
            Options: "mse", "l2", "l1", "relative_l2", "h1"
        testing_loss: Loss function for evaluation.
        loss_weights: Optional weights for multi-objective losses.
        lambda_mse: Weight for MSE component in combined losses.
        lambda_nce: Weight for NCE/contrastive component in combined losses.
        regularization: Type of regularization ("l1", "l2", "none").
        reg_weight: Weight for regularization term.
    """
    training_loss: Literal["mse", "l2", "l1", "relative_l2", "h1"] = "l2"
    testing_loss: Literal["mse", "l2", "l1", "relative_l2", "h1"] = "l2"
    loss_weights: Optional[List[float]] = None
    lambda_mse: float = 1.0
    lambda_nce: float = 0.0
    regularization: Literal["l1", "l2", "none"] = "none"
    reg_weight: float = 1e-5


@dataclass
class OptimizationConfig:
    """Configuration for training optimization.
    
    Attributes:
        optimizer: Type of optimizer to use.
            Options: "adam", "adamw", "sgd", "rmsprop", "lamb"
        learning_rate: Initial learning rate.
        weight_decay: Weight decay (L2 penalty).
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        eval_interval: Interval for evaluation (in epochs).
        gradient_clip: Maximum gradient norm for clipping (None to disable).
        mixed_precision: Whether to use mixed precision training.
        accumulation_steps: Gradient accumulation steps.
        early_stopping: Whether to use early stopping.
        early_stopping_patience: Patience for early stopping.
        seed: Random seed for reproducibility.
        scheduler: Learning rate scheduler configuration.
        loss: Loss function configuration.
    """
    # Optimizer settings
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop", "lamb"] = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Training settings
    n_epochs: int = 500
    batch_size: int = 32
    eval_interval: int = 1
    gradient_clip: Optional[float] = 1.0
    mixed_precision: bool = False
    accumulation_steps: int = 1
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 50
    
    # Reproducibility
    seed: Optional[int] = 42
    
    # Nested configs
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
