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

"""OPTICS (Ordering Points To Identify Clustering Structure) implementation.

This module provides an implementation of the OPTICS clustering algorithm using. 
It automatically estimates minimum cluster size and minimum samples based on the dataset, 
and lets the user override these parameters.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import OPTICS as SKLearnOPTICS

# Local imports
from base import Clustering


class Optics(Clustering):
    """
    OPTICS clustering implementation with parameter estimation and optional user override.

    Attributes:
        min_samples (int): The minimum number of samples in a neighborhood for a point
                           to be considered a core point. Initialized after heuristic
                           or user input.
        min_cluster_size (int): The minimum number of samples in a cluster.
                                Initialized after heuristic or user input.
        n_clusters (int): Number of clusters found (excluding noise).
        n_noise (int): Number of points labeled as noise.
    """

    def __init__(self) -> None:
        """
        Initialize the OPTICS clustering model.

        By default, `min_cluster_size` and `min_samples` are estimated after analyzing
        dataset size. The user can override these parameters in the `.fit(...)` method.
        """
        super().__init__("optics")
        self.min_samples: Optional[int] = None
        self.min_cluster_size: Optional[int] = None
        self.n_clusters: Optional[int] = None
        self.n_noise: Optional[int] = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the OPTICS model to the data, estimating `min_cluster_size` and `min_samples`.

        Args:
            X (np.ndarray): Input data array of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)
        n_samples, n_features = X.shape

        # Heuristic for min_samples
        self.min_samples = max(5, int(np.log2(n_samples)) + 1)

        # Heuristic for min_cluster_size:
        #   - minimum of 5% of data or 50
        #   - but not less than min_samples
        self.min_cluster_size = max(self.min_samples, min(50, int(0.05 * n_samples)))

        print("Estimated parameters:")
        print(
            f"min_cluster_size = {self.min_cluster_size}, min_samples = {self.min_samples}"
        )

        user_input = (
            input("Do you want to input your own parameters? (y/n): ").strip().lower()
        )
        if user_input == "y":
            min_cluster_size_input = input(
                f"Enter 'min_cluster_size' (current: {self.min_cluster_size}): "
            )
            min_samples_input = input(
                f"Enter 'min_samples' (current: {self.min_samples}): "
            )
            self.min_cluster_size = int(min_cluster_size_input)
            self.min_samples = int(min_samples_input)

        # Fit the OPTICS model
        self.model = SKLearnOPTICS(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            xi=0.05,
            metric="euclidean",
            n_jobs=-1,
        )
        self.model.fit(X)

        self.labels = self.model.labels_
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = np.count_nonzero(self.labels == -1)

        if self.n_clusters == 0:
            print("Warning: No clusters found. All points labeled as noise.")
        else:
            print(f"OPTICS found {self.n_clusters} clusters")
            print(f"Noise points: {self.n_noise} ({(self.n_noise/n_samples)*100:.1f}%)")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the model parameters and clustering results.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - model_type (str): The name of the clustering algorithm ("optics").
                - min_cluster_size (int): Final min_cluster_size used.
                - min_samples (int): Final min_samples used.
                - n_clusters (int, optional): Number of clusters found.
        """
        return {
            "model_type": self.model_type,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
        }
