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
import torch.nn as nn
import torch.nn.functional as F
from ...config_sr import FNOConfig
from .fno_block_sr import FNOBlock1d

class FNO1d(nn.Module): 
    def __init__(self, config: FNOConfig):
        """
        Args:
            config: FNOConfig object containing model configuration parameters

        Returns:
            Output tensor of shape (batch, out_channels)
        """
        super().__init__()
        self.config = config
        
        # Lift to higher dimension
        self.fc0 = nn.Linear(config.in_channels, config.width)
        
        self.blocks = nn.ModuleList([
            FNOBlock1d(config.width, config.modes) 
            for _ in range(config.n_layers)
        ])
        
        # Project back to solution
        self.fc1 = nn.Linear(config.width, 128)
        self.fc2 = nn.Linear(128, config.out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, nx, in_channels)
        Returns:    
            Output tensor of shape (batch, out_channels)
        """
        # Lifting layer
        x = self.fc0(x)
        
        # Transpose for conv
        x = x.permute(0, 2, 1)
        
        # blocks
        for block in self.blocks:
            x = block(x)
        
        # Transpose back
        x = x.permute(0, 2, 1)
        
        # Project to output
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)
    
    def count_parameters(self) -> int:
        """
        Args:
            None
        Returns:
            Total number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters())