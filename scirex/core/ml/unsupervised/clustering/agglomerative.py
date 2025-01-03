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
Module: agglomerative.py

This module provides an Agglomerative Clustering implementation using scikit-learn.
It optionally allows a user-defined cluster count or uses silhouette scores (2..max_k) to
auto-select the optimal cluster count.

Classes:
    Agglomerative: Implements an agglomerative clustering approach with optional
                   user-specified n_clusters or silhouette-based auto selection.

Dependencies:
    - numpy
    - sklearn.cluster.AgglomerativeClustering
    - sklearn.metrics.silhouette_score
    - base.py (Clustering)

Key Features:
    - Automatic scanning of possible cluster counts (2..max_k) if n_clusters is not provided
    - Silhouette-based selection of the best cluster count
    - Provides get_model_params() for retrieving final clustering info

Authors:
    - Debajyoti Sahoo (debajyotis@iisc.ac.in)

Version Info:
    - 28/Dec/2024: Initial release
"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Local imports
from .base import Clustering


class Agglomerative(Clustering):
    """
    Agglomerative Clustering with optional user-defined 'n_clusters' or
    automatic silhouette-based selection.

    Attributes:
        n_clusters (Optional[int]):
            User-specified cluster count. If not provided, the algorithm auto-selects.
        max_k (int):
            Maximum number of clusters for auto-selection if n_clusters is None.
        labels (Optional[np.ndarray]):
            Cluster labels for each data point after fitting.
        n_clusters_ (Optional[int]):
            The actual number of clusters used in the final model.
    """

    def __init__(self, n_clusters: Optional[int] = None, max_k: int = 10) -> None:
        """
        Initialize the Agglomerative clustering class.

        Args:
            n_clusters (Optional[int], optional):
                If provided, the class will use this cluster count directly.
                Otherwise, it scans 2..max_k using silhouette. Defaults to None.
            max_k (int, optional):
                Maximum number of clusters to try (auto selection) if n_clusters is None.
                Defaults to 10.
        """
        super().__init__("agglomerative")
        self.n_clusters = n_clusters
        self.max_k = max_k

        self.labels: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.model: Optional[AgglomerativeClustering] = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the Agglomerative Clustering model to the data.

        - If n_clusters is provided, skip auto selection and use that value.
        - Else, compute silhouette scores for k in [2..max_k],
          pick the best k, and finalize the clustering.

        Args:
            X (np.ndarray): Input data array of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]

        # 1) If user specified a cluster count
        if self.n_clusters is not None:
            optimal_k = self.n_clusters
            print(
                f"Fitting AgglomerativeClustering with user-defined n_clusters={optimal_k}.\n"
            )
        else:
            # 2) Auto selection using silhouette
            k_values = range(2, self.max_k + 1)
            silhouette_scores = []
            sample_size = min(1000, n_samples)

            for k in k_values:
                model = AgglomerativeClustering(n_clusters=k, linkage="average")
                labels = model.fit_predict(X)

                if len(np.unique(labels)) <= 1:
                    silhouette_scores.append(-1)
                    continue

                if n_samples > sample_size:
                    indices = np.random.choice(n_samples, sample_size, replace=False)
                    score = silhouette_score(X[indices], labels[indices])
                else:
                    score = silhouette_score(X, labels)

                silhouette_scores.append(score)

            # pick best silhouette
            idx_best = np.argmax(silhouette_scores)
            optimal_k = k_values[idx_best]
            print(f"Optimal k (silhouette) = {optimal_k}")

        # Final fit
        self.model = AgglomerativeClustering(n_clusters=optimal_k, linkage="average")
        self.labels = self.model.fit_predict(X)
        self.n_clusters_ = len(np.unique(self.labels))

        print(f"AgglomerativeClustering fitted with {optimal_k} clusters.\n")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Retrieve key parameters and results from the fitted AgglomerativeClustering model.

        Returns:
            Dict[str, Any]:
                - n_clusters (int): The final number of clusters used
        """
        return {"n_clusters": self.n_clusters_}
