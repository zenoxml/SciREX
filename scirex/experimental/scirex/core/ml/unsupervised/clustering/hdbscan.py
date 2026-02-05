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
Module: hdbscan.py

This module provides an HDBSCAN (Hierarchical Density-Based Spatial Clustering of
Applications with Noise) implementation. It optionally allows user-defined
`min_cluster_size` and `min_samples`, or applies heuristics to determine them.

Classes:
    Hdbscan: Implements HDBSCAN with an optional user override or heuristic-based approach.

Dependencies:
    - numpy
    - hdbscan (pip install hdbscan)
    - base.py (Clustering)

Key Features:
    - If user-defined 'min_cluster_size' or 'min_samples' is given, skip auto-heuristic
    - Otherwise, compute simple heuristics
    - Inherits from base `Clustering` for consistency with other clustering modules

Authors:
    - Debajyoti Sahoo (debajyotis@iisc.ac.in)

Version Info:
    - 28/Dec/2024: Initial release
"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import HDBSCAN

# Local imports
from .base import Clustering


class Hdbscan(Clustering):
    """
    HDBSCAN clustering with optional user-defined 'min_cluster_size' and 'min_samples',
    or a heuristic-based approach if they are not provided.

    Attributes:
        min_cluster_size (Optional[int]):
            User-specified or auto-calculated minimum cluster size.
        min_samples (Optional[int]):
            User-specified or auto-calculated minimum samples for a point to be core.
        cluster_selection_method (str):
            Method for extracting clusters from the condensed tree. Defaults to 'eom'.
        labels (Optional[np.ndarray]):
            Cluster labels for each data point after fitting (some may be -1 for noise).
        n_clusters_ (Optional[int]):
            Number of clusters discovered (excluding noise).
        n_noise_ (Optional[int]):
            Number of data points labeled as noise (-1).
    """

    def __init__(
        self,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        cluster_selection_method: str = "eom",
    ) -> None:
        """
        Initialize the HDBSCAN clustering model.

        Args:
            min_cluster_size (Optional[int], optional):
                If provided, HDBSCAN will use this min cluster size directly.
            min_samples (Optional[int], optional):
                If provided, HDBSCAN will use this min_samples directly.
            cluster_selection_method (str, optional):
                The method to extract clusters from condensed tree:
                'eom' (Excess of Mass) or 'leaf'. Defaults to 'eom'.
        """
        super().__init__("hdbscan")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method

        # Attributes set after fitting
        self.labels: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.n_noise_: Optional[int] = None
        self.model: Optional[HDBSCAN] = None

    def _estimate_params(self, X: np.ndarray) -> None:
        """
        Estimate min_cluster_size and min_samples using simple heuristics:
          - min_samples = max(1, floor(log(n)))
          - min_cluster_size = max(5, floor(2% of n))
        """
        n_samples = X.shape[0]
        # Heuristic for min_samples
        auto_min_samples = max(1, int(np.log(n_samples)))
        # Heuristic for min_cluster_size
        auto_min_cluster_size = max(5, int(0.02 * n_samples))

        # If not user-defined, assign
        if self.min_cluster_size is None:
            self.min_cluster_size = auto_min_cluster_size
        if self.min_samples is None:
            self.min_samples = auto_min_samples

        print("Auto-estimated parameters for HDBSCAN:")
        print(
            f"min_cluster_size = {self.min_cluster_size}, "
            f"min_samples = {self.min_samples}"
        )

    def fit(self, X: np.ndarray) -> None:
        """
        Fit HDBSCAN to the data.

        - If min_cluster_size/min_samples are None, estimate them heuristically.
        - Then create and fit an HDBSCAN model, storing labels, cluster count, and noise count.

        Args:
            X (np.ndarray): Input data array of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)

        # If user did not provide min_cluster_size or min_samples, estimate them
        if self.min_cluster_size is None or self.min_samples is None:
            self._estimate_params(X)
        else:
            print(
                f"Using user-defined parameters: "
                f"min_cluster_size={self.min_cluster_size}, "
                f"min_samples={self.min_samples}"
            )

        # Create and fit the model
        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
        )
        self.labels = self.model.fit_predict(X)

        # Count clusters (excluding noise)
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise_ = np.count_nonzero(self.labels == -1)

        print(
            f"\nHDBSCAN fitted with min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}"
        )

    def get_model_params(self) -> Dict[str, Any]:
        """
        Retrieve key parameters and results from the fitted HDBSCAN model.

        Returns:
            Dict[str, Any]:
                - min_cluster_size (int): Final min_cluster_size used
                - min_samples (int): Final min_samples used
                - n_clusters (int): Number of clusters discovered
                - n_noise (int): Number of noise points
        """
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters_,
            "n_noise": self.n_noise_,
        }
