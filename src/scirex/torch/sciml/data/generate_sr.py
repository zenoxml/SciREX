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

import torch
import numpy as np
from typing import Tuple
from ..config_sr import DataConfig
from ..builders import WaveSolver, HeatSolver, AdvectionSolver

class DataGenerator:
    SOLVERS = {
        'wave': WaveSolver,
        'heat': HeatSolver,
        'advection': AdvectionSolver
    }
    
    PARAM_RANGES = {
        'wave': {'param_range': (0.3, 1.2), 'T': 0.3},
        'heat': {'param_range': (0.005, 0.015), 'T': 0.3},
        'advection': {'param_range': (0.3, 1.2), 'T': 0.2}
    }
    
    @classmethod
    def generate_dataset(cls, pde_type: str, n_samples: int, 
                        config: DataConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate dataset for a specific PDE type
        
        Args:
            pde_type: Type of PDE ('wave', 'heat', or 'advection')
            n_samples: Number of samples to generate
            config: Data configuration
            
        Returns:
            Tuple of (input_data, output_data) tensors
        """
        if pde_type not in cls.SOLVERS:
            raise ValueError(f"Unknown PDE type: {pde_type}")
        
        solver = cls.SOLVERS[pde_type]
        param_config = cls.PARAM_RANGES[pde_type]
        param_min, param_max = param_config['param_range']
        T = param_config['T']
        
        data_x = []
        data_y = []
        
        for i in range(n_samples):
            success = False
            
            for attempt in range(5):
                try:
                    # Generate random parameter
                    param = np.random.uniform(param_min, param_max)
                    
                    # Solve PDE
                    if pde_type in ['wave', 'advection']:
                        x, t, u = solver.solve(nx=config.nx, nt=config.nt, c=param, 
                                              L=config.L, T=T)
                    else:  # heat
                        x, t, u = solver.solve(nx=config.nx, nt=config.nt, alpha=param, 
                                              L=config.L, T=T)
                    
                    # Normalize parameter to [0, 1]
                    param_norm = (param - param_min) / (param_max - param_min)
                    
                    # Create input/output pairs
                    input_data = np.stack([u[0], np.ones_like(u[0]) * param_norm], axis=-1)
                    output_data = u[-1]
                    
                    # Verify data is finite
                    if (np.all(np.isfinite(input_data)) and 
                        np.all(np.isfinite(output_data)) and
                        np.abs(output_data).max() < 100):
                        data_x.append(input_data)
                        data_y.append(output_data)
                        success = True
                        break
                        
                except Exception as e:
                    if attempt == 4:
                        print(f"  Warning: Failed to generate sample {i+1}: {e}")
            
            if not success:
                # Use conservative parameter as fallback
                param = (param_min + param_max) / 2
                if pde_type in ['wave', 'advection']:
                    x, t, u = solver.solve(nx=config.nx, nt=config.nt, c=param, 
                                          L=config.L, T=T*0.7)
                else:
                    x, t, u = solver.solve(nx=config.nx, nt=config.nt, alpha=param, 
                                          L=config.L, T=T*0.7)
                
                param_norm = (param - param_min) / (param_max - param_min)
                input_data = np.stack([u[0], np.ones_like(u[0]) * param_norm], axis=-1)
                output_data = u[-1]
                data_x.append(input_data)
                data_y.append(output_data)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{n_samples} samples")
        
        return torch.FloatTensor(np.array(data_x)), torch.FloatTensor(np.array(data_y))