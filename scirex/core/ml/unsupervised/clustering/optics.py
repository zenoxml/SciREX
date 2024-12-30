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

    This module provides an implementation of the OPTICS (Ordering Points To Identify Clustering Structure)
    clustering algorithm using scikit-learn's OPTICS. It automatically estimates minimum cluster size
    and minimum samples based on the dataset, and lets the user override these parameters.

    Classes:
        Optics: An OPTICS clustering implementation with heuristic parameter estimation and optional user override.

    Dependencies:
        - numpy
        - sklearn.cluster.OPTICS
        - base.py (Clustering)

    Key Features:
        - Automatic estimation of `min_samples` based on dataset size (log2 heuristic)
        - Automatic estimation of `min_cluster_size` (5% of data or 50, whichever is smaller,
          but not less than `min_samples`)
        - Computation of discovered clusters, noise points, and summary messages

    Authors:
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

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
        #   - Minimum of 5% of data or 50
        #   - but not less than min_samples
        self.min_cluster_size = max(self.min_samples, min(50, int(0.05 * n_samples)))

        print("Estimated parameters:")
        print(
            f"min_cluster_size = {self.min_cluster_size}, min_samples = {self.min_samples}"
        )

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
