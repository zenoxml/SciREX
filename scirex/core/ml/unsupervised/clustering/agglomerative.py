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

"""Agglomerative Clustering implementation.

This module provides an Agglomerative Clustering implementation. 

It includes automatic selection of the optimal number of clusters
using silhouette scores, with an optional user override.
"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Local imports
from base import Clustering


class Agglomerative(Clustering):
    """
    Agglomerative Clustering with automatic selection of the optimal cluster count via silhouette.

    Attributes:
        max_k (int): The maximum number of clusters to consider.
        optimal_k (Optional[int]): The chosen number of clusters after evaluating silhouette
                                   scores and optional user input.
        n_clusters (Optional[int]): Final cluster count (same as optimal_k).
    """

    def __init__(self, max_k: int = 10) -> None:
        """
        Initialize the Agglomerative clustering class.

        Args:
            max_k (int, optional): Maximum number of clusters to try. Defaults to 10.
        """
        super().__init__("agglomerative")
        self.max_k = max_k
        self.optimal_k: Optional[int] = None
        self.n_clusters: Optional[int] = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit method for Agglomerative Clustering with automatic cluster count selection.

        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]
        k_values = range(2, self.max_k + 1)
        silhouettes = []

        # Subsample size for silhouette
        silhouette_sample_size = min(1000, n_samples)

        for k in k_values:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="average",  # or 'ward', 'complete', 'single' as needed
            )
            labels = model.fit_predict(X)
            unique_labels = np.unique(labels)

            # If there's only one cluster, silhouette is invalid
            if len(unique_labels) <= 1:
                silhouettes.append(-1)
                continue

            # Subsample if data is large
            if n_samples > silhouette_sample_size:
                indices = np.random.choice(
                    n_samples, silhouette_sample_size, replace=False
                )
                silhouettes.append(silhouette_score(X[indices], labels[indices]))
            else:
                silhouettes.append(silhouette_score(X, labels))

        # Choose best k from silhouette
        self.optimal_k = k_values[np.argmax(silhouettes)]
        print(f"Estimated optimal number of clusters (optimal_k): {self.optimal_k}")

        # Optional user override
        user_input = (
            input("Do you want to input your own number of clusters? (y/n): ")
            .strip()
            .lower()
        )
        if user_input == "y":
            k_input = int(
                input(
                    f"Enter the number of clusters (k), current estimate is {self.optimal_k}: "
                )
            )
            if k_input >= 2:
                self.optimal_k = k_input
            else:
                print(
                    "Number of clusters must be at least 2. Using the estimated optimal_k."
                )

        # Final fit with optimal_k
        self.model = AgglomerativeClustering(
            n_clusters=self.optimal_k, linkage="average"
        )
        self.labels = self.model.fit_predict(X)
        self.n_clusters = len(np.unique(self.labels))

        print(f"Agglomerative Clustering fitted with k = {self.optimal_k}")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters of the fitted Agglomerative Clustering model.

        Returns:
            Dict[str, Any]:
                A dictionary containing:
                - model_type (str): "agglomerative"
                - n_clusters (int): The final chosen cluster count
        """
        return {"model_type": self.model_type, "n_clusters": self.optimal_k}
