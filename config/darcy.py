# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited.
# Licensed under the Apache License, Version 2.0.

"""Darcy Flow Configuration for training neural operators."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO2DConfig


@dataclass
class DarcyDataConfig:
    """Configuration for Darcy Flow data."""
    data_path: Optional[str] = None
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 16
    nx: int = 64
    ny: int = 64
    domain: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))
    permeability_type: str = "log_normal"  # "log_normal", "channelized", "binary"
    normalize: bool = True
    include_mesh: bool = True


@dataclass
class DarcyFlowConfig:
    """Configuration for Darcy Flow: -∇·(a(x)∇u) = f.
    
    Input: Permeability field a(x) + mesh coordinates
    Output: Pressure field u(x)
    """
    name: str = "darcy_flow"
    description: str = "Darcy Flow: -∇·(a∇u) = f"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=3, out_channels=1, hidden_channels=64,
        n_layers=4, n_modes=(12, 12), activation="gelu", use_channel_mlp=True
    ))
    data: DarcyDataConfig = field(default_factory=DarcyDataConfig)
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3, n_epochs=500, batch_size=16,
        scheduler=SchedulerConfig(scheduler_type="StepLR", step_size=100, gamma=0.5),
        loss=LossConfig(training_loss="relative_l2")
    ))


@dataclass
class DarcyHighResConfig:
    """High resolution Darcy Flow configuration."""
    name: str = "darcy_high_res"
    description: str = "High resolution Darcy Flow"
    
    model: FNO2DConfig = field(default_factory=lambda: FNO2DConfig(
        in_channels=3, out_channels=1, hidden_channels=128,
        n_layers=5, n_modes=(24, 24), activation="gelu", use_channel_mlp=True
    ))
    data: DarcyDataConfig = field(default_factory=lambda: DarcyDataConfig(
        nx=128, ny=128, batch_size=8
    ))
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=5e-4, n_epochs=800, batch_size=8
    ))


DarcyConfig = DarcyFlowConfig
