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
    It includes automatic selection of the optimal number of clusters using silhouette scores.

    Classes:
        Agglomerative: Implements an agglomerative clustering approach with auto cluster selection.

    Dependencies:
        - numpy
        - sklearn.cluster.AgglomerativeClustering
        - sklearn.metrics.silhouette_score
        - base.py (Clustering)

    Key Features:
        - Automatic scanning of possible cluster counts (2..max_k)
        - Silhouette-based selection of the best cluster count
        - Final model/labels accessible after `.fit(...)`

    Authors:
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

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
        Fit the Agglomerative Clustering model with automatic cluster count selection.

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
                linkage="average",  # or 'ward', 'complete', 'single'
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
