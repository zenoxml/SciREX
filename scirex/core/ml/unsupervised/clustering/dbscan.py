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
    Module: dbscan.py

    This module provides a DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    implementation.

    It includes an optional automated heuristic for estimating `eps` and `min_samples`
    by analyzing neighborhood distances. The user can override these defaults before fitting.

    Classes:
        Dbscan: Implements DBSCAN with a simple heuristic for `eps` and `min_samples`.

    Dependencies:
        - numpy
        - sklearn.cluster.DBSCAN
        - sklearn.neighbors.NearestNeighbors
        - base.py (Clustering)

    Key Features:
        - Automatic estimation of `eps` via median k-distances
        - Automatic estimation of `min_samples` via log2(n) heuristic
        - Counting of discovered clusters and noise points

    Authors:
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

"""
# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Local imports
from .base import Clustering


class Dbscan(Clustering):
    """
    DBSCAN clustering algorithm with optional automatic estimation of `eps` and `min_samples`.

    Attributes:
        eps (Optional[float]):
            If provided, use this neighborhood distance for DBSCAN.
        min_samples (Optional[int]):
            If provided, use this minimum samples count for a point to be considered a core point.
        labels (Optional[np.ndarray]):
            Cluster labels for each data point after fitting.
        n_clusters_ (Optional[int]):
            The number of clusters found (excluding noise).
        n_noise_ (Optional[int]):
            The number of noise points labeled as -1.
    """

    def __init__(
        self, eps: Optional[float] = None, min_samples: Optional[int] = None
    ) -> None:
        """
        Initialize the Dbscan clustering class.

        Args:
            eps (Optional[float], optional):
                User-defined neighborhood distance. If None, auto-estimation is used.
            min_samples (Optional[int], optional):
                User-defined min_samples. If None, auto-estimation is used.
        """
        super().__init__("dbscan")
        self.eps = eps
        self.min_samples = min_samples

        # Final results after fitting
        self.labels: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.n_noise_: Optional[int] = None
        self.model: Optional[DBSCAN] = None

    def _estimate_params(self, X: np.ndarray) -> None:
        """
        Estimate eps and min_samples if they are not already set by the user,
        using a heuristic approach:
          - min_samples = max(5, floor(log2(n)) + 1)
          - eps = median distance to the 'min_samples-th' nearest neighbor
        """
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Estimate min_samples if not set by user
        if self.min_samples is None:
            auto_min_samples = max(5, int(np.log2(n_samples)) + 1)
            self.min_samples = auto_min_samples
        else:
            auto_min_samples = self.min_samples

        # Estimate eps if not set by user
        if self.eps is None:
            # Subsample for distance estimation
            sample_size = min(1000, n_samples)
            indices = rng.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]

            # Compute the distance to the (min_samples)-th neighbor
            nbrs = NearestNeighbors(n_neighbors=auto_min_samples)
            nbrs.fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            k_distances = distances[:, -1]

            auto_eps = float(np.median(k_distances))
            self.eps = auto_eps

        print("Auto-estimated parameters:")
        print(f"eps = {self.eps:.4f}, min_samples = {self.min_samples}")

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the DBSCAN model to the data.

        If eps or min_samples are None, a heuristic is used to estimate them.
        The resulting DBSCAN model is stored in self.model, along with labels,
        cluster count, and noise count.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)

        # If user didn't provide eps or min_samples, estimate them
        if self.eps is None or self.min_samples is None:
            self._estimate_params(X)
        else:
            print(
                f"Using user-defined eps = {self.eps}, min_samples = {self.min_samples}"
            )

        # fit DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(X)

        # Count clusters (excluding noise)
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise_ = np.count_nonzero(self.labels == -1)

        print(f"\nDBSCAN fitted with eps={self.eps}, min_samples={self.min_samples}")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Retrieve key parameters and results from the fitted DBSCAN model.

        Returns:
            Dict[str, Any]:
                - model_type (str): "dbscan"
                - eps (float): The final eps used
                - min_samples (int): The final min_samples used
                - n_clusters (int): Number of clusters found (excluding noise)
                - n_noise (int): Number of noise points (-1 label)
        """
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters_,
            "n_noise": self.n_noise_,
        }
