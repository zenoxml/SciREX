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

import numpy as np
from typing import Tuple
from base import PDESolver

class AdvectionSolver(PDESolver):
    """Solver for advection equation: u_t + c * u_x = 0"""
    
    @staticmethod
    def solve(nx: int = 256, nt: int = 100, c: float = 1.0, 
              L: float = 1.0, T: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve advection equation with periodic boundary conditions
        
        Args:
            nx: Number of spatial points
            nt: Number of time steps
            c: Advection speed
            L: Domain length
            T: Final time
            
        Returns:
            Tuple of (x, t, u) arrays
        """
        x = np.linspace(0, L, nx)
        dx = x[1] - x[0]
        
        # CFL condition
        dt_max = 0.8 * dx / abs(c)
        dt = min(T / nt, dt_max)
        nt_actual = int(np.ceil(T / dt)) + 1
        t = np.linspace(0, T, nt_actual)
        dt = t[1] - t[0]
        
        # Initial condition: smooth step function
        u0 = 0.5 * (np.tanh(50*(x - 0.3)) - np.tanh(50*(x - 0.5)))
        
        # Solution array
        u = np.zeros((nt_actual, nx))
        u[0] = u0
        
        # Upwind scheme
        r = c * dt / dx
        if abs(r) > 1.0:
            raise ValueError(f"CFL condition violated: |r|={abs(r):.4f} > 1.0")
        
        for n in range(nt_actual - 1):
            if c > 0:
                u[n+1, 1:] = u[n, 1:] - r * (u[n, 1:] - u[n, :-1])
                u[n+1, 0] = u[n, 0] - r * (u[n, 0] - u[n, -1])
            else:
                u[n+1, :-1] = u[n, :-1] - r * (u[n, 1:] - u[n, :-1])
                u[n+1, -1] = u[n, -1] - r * (u[n, 0] - u[n, -1])
            
            if not PDESolver._check_stability(u[n+1], n+1):
                return x, t[:n+1], u[:n+1]
        
        return x, t, u