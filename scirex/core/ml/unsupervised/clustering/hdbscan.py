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

    This module provides an implementation of the HDBSCAN (Hierarchical Density-Based Spatial 
    Clustering of Applications with Noise) algorithm.

    HDBSCAN automatically finds clusters of varying densities by constructing a hierarchical
    tree structure and extracting stable clusters from the hierarchy using the EOM
    (Excess of Mass) method or another selection approach.

    Classes:
        Hdbscan: Implements HDBSCAN with a heuristic for `min_cluster_size` and `min_samples`,
                 plus optional user override.

    Dependencies:
        - numpy
        - sklearn.cluster.HDBSCAN
        - base.py (Clustering)

    Key Features:
        - Automatic heuristic for `min_cluster_size` and `min_samples`
        - Optional user override for parameters
        - Summarizes discovered clusters and noise points
        - Inherits from base `Clustering` for a consistent pipeline

    Authors:
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

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
    HDBSCAN clustering with automatic heuristic for `min_cluster_size` and `min_samples`,
    plus optional user override.

    Attributes:
        min_cluster_size (Optional[int]): The minimum size of a cluster (in samples).
        min_samples (Optional[int]): The minimum number of samples in a neighborhood for
                                     a point to be considered a core point.
        n_clusters (Optional[int]): Number of clusters discovered (excluding noise).
        n_noise (Optional[int]): Number of points labeled as noise (i.e., assigned label -1).
    """

    def __init__(self) -> None:
        """
        Initialize the HDBSCAN clustering model.

        By default, `min_cluster_size` and `min_samples` are computed at fit time
        based on dataset properties, then optionally overridden by user input.
        """
        super().__init__("hdbscan")
        self.min_cluster_size: Optional[int] = None
        self.min_samples: Optional[int] = None
        self.n_clusters: Optional[int] = None
        self.n_noise: Optional[int] = None

    def fit(self, X: np.ndarray, expected_clusters: Optional[int] = None) -> None:
        """
        Fit the HDBSCAN model to the data using a simple heuristic for initial parameters.

        Args:
            X (np.ndarray): Input data array of shape (n_samples, n_features).
            expected_clusters (Optional[int]): Potential future extension for approximate cluster count.
        """
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]

        # Heuristic estimation of parameters
        self.min_samples = max(1, int(np.log(n_samples)))
        self.min_cluster_size = max(5, int(0.02 * n_samples))

        print("Estimated parameters:")
        print(
            f"min_cluster_size = {self.min_cluster_size}, min_samples = {self.min_samples}"
        )

        # (Optional) prompt user for overrides if desired (not shown in snippet)
        # For example: user_input = input("Do you want to override? ...")

        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method="eom",
        )
        self.model.fit(X)

        self.labels = self.model.labels_
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = np.count_nonzero(self.labels == -1)

        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of noise points: {self.n_noise}")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the model parameters and clustering results.

        Returns:
            Dict[str, Any]:
                - model_type (str): "hdbscan"
                - min_cluster_size (int)
                - min_samples (int)
                - n_clusters (int)
        """
        return {
            "model_type": self.model_type,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
        }
