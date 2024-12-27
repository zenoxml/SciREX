# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab,
# Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# SciREX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with SciREX. If not, see <https://www.gnu.org/licenses/>.
#
# For any clarifications or special considerations,
# please contact <scirex@zenteiq.ai>

# Author: Dev Sahoo
# Linkedin: https://www.linkedin.com/in/debajyoti-sahoo13/

"""HDBSCAN (Hierarchical Density-Based Spatial Clustering) implementation.

This module provides an implementation of the HDBSCAN clustering algorithm.
"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import HDBSCAN

# Local imports
from base import Clustering


class Hdbscan(Clustering):
    """
    HDBSCAN clustering with automatic heuristic for `min_cluster_size` and `min_samples`,
    plus optional user override.

    Attributes:
        min_cluster_size (int): The minimum size of a cluster (in samples).
        min_samples (int): The minimum number of samples in a neighborhood for a point
                           to be considered a core point.
        n_clusters (int): Number of clusters discovered (excluding noise).
        n_noise (int): Number of points labeled as noise (i.e., assigned label -1).
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
            expected_clusters (Optional[int]): If you have a rough expectation of how many
                                               clusters should be found, you can pass it here
                                               (currently not used in the default logic, but
                                               available for future extension).
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

        # User override
        user_input = (
            input("Do you want to input your own parameters? (y/n): ").strip().lower()
        )
        if user_input == "y":
            try:
                min_cluster_size_input = input(
                    f"Enter min_cluster_size (current: {self.min_cluster_size}): "
                )
                min_samples_input = input(
                    f"Enter min_samples (current: {self.min_samples}): "
                )
                if min_cluster_size_input.strip():
                    self.min_cluster_size = int(min_cluster_size_input)
                if min_samples_input.strip():
                    self.min_samples = int(min_samples_input)
            except ValueError:
                print("Invalid input. Using estimated parameters.")

        # Fit HDBSCAN
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
                A dictionary containing:
                - model_type (str): Name of the clustering algorithm ("hdbscan").
                - min_cluster_size (int): Final min_cluster_size used.
                - min_samples (int): Final min_samples used.
                - n_clusters (int): Number of clusters found.
        """
        return {
            "model_type": self.model_type,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
        }
