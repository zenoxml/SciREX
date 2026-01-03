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

class PDESolver:
    """Base class for PDE solvers"""
    
    @staticmethod
    def _check_stability(u: np.ndarray, step: int) -> bool:
        # Check if solution is stable (no NaNs or Infs)
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            print(f"Solution became unstable at step {step}")
            return False
        return True