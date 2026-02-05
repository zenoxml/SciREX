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
Module: optics.py

This module provides an OPTICS (Ordering Points To Identify the Clustering Structure)
implementation.

It allows optional user-defined 'min_samples' and 'min_cluster_size', or applies
a heuristic approach if they're not provided.

Classes:
    Optics: Implements OPTICS with optional user override or heuristic-based approach.

Dependencies:
    - numpy
    - sklearn.cluster.OPTICS
    - base.py (Clustering)

Key Features:
    - If user-defined 'min_samples' or 'min_cluster_size' is set, skip auto-heuristic
    - Otherwise, compute a simple heuristic

Authors:
    - Debajyoti Sahoo (debajyotis@iisc.ac.in)

Version Info:
    - 28/Dec/2024: Initial release
"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import OPTICS as SKLearnOPTICS

# Local imports
from .base import Clustering


class Optics(Clustering):
    """
    OPTICS clustering with optional user-defined 'min_samples' and 'min_cluster_size',
    or a heuristic-based approach if they are not provided.

    Attributes:
        min_samples (Optional[int]):
            If provided, used directly by OPTICS; otherwise estimated.
        min_cluster_size (Optional[int]):
            If provided, used directly by OPTICS; otherwise estimated.
        xi (float):
            Determines the minimum steepness on the reachability plot for cluster extraction.
        labels (Optional[np.ndarray]):
            Cluster labels for each data point after fitting (-1 for noise).
        n_clusters_ (Optional[int]):
            Number of clusters discovered (excluding noise).
        n_noise_ (Optional[int]):
            Number of data points labeled as noise.
    """

    def __init__(
        self,
        min_samples: Optional[int] = None,
        min_cluster_size: Optional[int] = None,
        xi: float = 0.05,
    ) -> None:
        """
        Initialize the OPTICS clustering model.

        Args:
            min_samples (Optional[int], optional):
                If provided, the algorithm uses this min_samples. Otherwise, a heuristic is used.
            min_cluster_size (Optional[int], optional):
                If provided, the algorithm uses this min_cluster_size. Otherwise, a heuristic is used.
            xi (float, optional):
                Determines the minimum steepness on the reachability plot for cluster extraction.
                Defaults to 0.05.
        """
        super().__init__("optics")
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.xi = xi

        # Attributes set after fitting
        self.labels: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.n_noise_: Optional[int] = None

    def _estimate_params(self, X: np.ndarray) -> None:
        """
        Estimate 'min_samples' and 'min_cluster_size' if they are not already provided
        by the user. We apply a simple heuristic approach:
          - min_samples = max(5, floor(log2(n)) + 1)
          - min_cluster_size = max(min_samples, min(50, floor(0.05 * n)))
        """
        n_samples = X.shape[0]

        if self.min_samples is None:
            auto_min_samples = max(5, int(np.log2(n_samples)) + 1)
            self.min_samples = auto_min_samples

        if self.min_cluster_size is None:
            # Use min_cluster_size >= min_samples, up to 5% of data or 50
            auto_min_cluster_size = max(
                self.min_samples, min(50, int(0.05 * n_samples))
            )
            self.min_cluster_size = auto_min_cluster_size

        print("Auto-estimated parameters for OPTICS:")
        print(
            f"min_samples = {self.min_samples}, min_cluster_size = {self.min_cluster_size}"
        )

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the OPTICS model to the data.
        - If min_samples/min_cluster_size are not set, estimate them heuristically.
        - Then create and fit an OPTICS model, storing labels and cluster info.

        Args:
            X (np.ndarray): Input data array of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]

        # If user did not provide min_samples or min_cluster_size, estimate them
        if self.min_samples is None or self.min_cluster_size is None:
            self._estimate_params(X)
        else:
            print(
                f"Using user-defined parameters: min_samples={self.min_samples}, "
                f"min_cluster_size={self.min_cluster_size}"
            )

        # fit the model
        self.model = SKLearnOPTICS(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            xi=self.xi,
            metric="euclidean",
            n_jobs=-1,
        )
        self.model.fit(X)

        self.labels = self.model.labels_
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise_ = np.count_nonzero(self.labels == -1)

    def get_model_params(self) -> Dict[str, Any]:
        """
        Retrieve key parameters and results from the fitted OPTICS model.

        Returns:
            Dict[str, Any]:
                - min_samples (int): The final min_samples used
                - min_cluster_size (int): The final min_cluster_size used
                - n_clusters (int): Number of clusters discovered (excluding noise)
                - n_noise (int): Number of data points labeled as noise
        """
        return {
            "min_samples": self.min_samples,
            "min_cluster_size": self.min_cluster_size,
            "n_clusters": self.n_clusters_,
            "n_noise": self.n_noise_,
        }
