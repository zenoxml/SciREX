from dataclasses import dataclass, field
from typing import Tuple, Literal
from .opt import OptimizationConfig, SchedulerConfig, LossConfig
from .models import FNO1DConfig

@dataclass
class HeatDataConfig:
    """Configuration for Heat equation data."""
    n_samples: int = 1200
    n_train: int = 1000
    n_test: int = 200
    nx: int = 64
    nt: int = 100
    L: float = 6.28318530718  # 2*pi
    T: float = 1.0
    D: float = 0.1

@dataclass
class Heat1DConfig:
    """Configuration for 1D Heat Equation Experiment."""
    name: str = "heat_1d"
    description: str = "1D Heat equation: u_t = D * u_xx"
    
    model: FNO1DConfig = field(default_factory=lambda: FNO1DConfig(
        in_channels=2,        # u0(x) + x
        out_channels=1,       # u(x, T)
        hidden_channels=64,   # Width
        n_layers=4,
        n_modes=(16,),
        activation="gelu",
        use_channel_mlp=True,
        norm="group_norm"
    ))
    
    data: HeatDataConfig = field(default_factory=HeatDataConfig)
    
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        learning_rate=1e-3,
        n_epochs=50,
        batch_size=50,
        scheduler=SchedulerConfig(
            scheduler_type="StepLR",
            step_size=10,
            gamma=0.5
        ),
        loss=LossConfig(
            training_loss="mse",
            testing_loss="mse"
        )
    ))


# Convenience alias
HeatConfig = Heat1DConfig

