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
    Module: kmeans.py

    This module provides a K-means clustering implementation using scikit-learn's MiniBatchKMeans.
    The Kmeans class inherits from a generic Clustering base class and offers:
      - Option for the user to input a custom number of clusters
      - Option for Automatic selection of the optimal number of clusters via silhouette or elbow methods
      
    Classes:
        Kmeans: K-Means clustering with automatic parameter selection and optional user override.

    Dependencies:
        - numpy
        - sklearn.cluster.MiniBatchKMeans
        - sklearn.metrics.silhouette_score
        - base.py (Clustering)

    Key Features:
        - Scans [2..max_k] to find the best cluster count using silhouette or elbow
        - Final cluster count stored in `optimal_k`, with a fitted model and labels
        - Inherits from the base `Clustering` for consistent plotting and metric computation

    Authors:
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Local imports
from .base import Clustering


class Kmeans(Clustering):
    """
    K-Means clustering with optional user-defined 'n_clusters' or automatic selection.

    Attributes:
        n_clusters (Optional[int]):
            If provided, the class will skip automatic selection and use this number of clusters.
        max_k (int):
            Maximum number of clusters to consider for automatic selection if n_clusters is None.
        labels (Optional[np.ndarray]):
            Cluster labels for each data point after fitting.
        n_clusters_ (Optional[int]):
            The actual number of clusters used by the final fitted model.
        inertia_ (Optional[float]):
            The final inertia (sum of squared distances to the closest centroid).
        cluster_centers_ (Optional[np.ndarray]):
            Coordinates of cluster centers in the final fitted model.
    """

    def __init__(
        self, n_clusters: Optional[int] = None, max_k: Optional[int] = 10
    ) -> None:
        """
        Initialize the Kmeans clustering class.

        Args:
            n_clusters (Optional[int], optional):
                User-defined number of clusters when provided, the algorithm will ignore automatic selection and directly use 'n_clusters'.
                Defaults to None.
            max_k (Optional[int], optional):
                Maximum number of clusters to try for automatic selection if n_clusters is None.
                Defaults to 10.
        """
        super().__init__("kmeans")
        self.n_clusters = n_clusters
        self.max_k = max_k

        # Attributes populated after fitting
        self.labels: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.inertia_: Optional[float] = None
        self.cluster_centers_: Optional[np.ndarray] = None

    def _calculate_elbow_scores(self, X: np.ndarray, k_values: range) -> np.ndarray:
        """
        Calculate inertia scores for each candidate k, then compute a second-derivative-like ratio
        for the elbow method.

        Args:
            X (np.ndarray):
                Input data of shape (n_samples, n_features).
            k_values (range):
                Range of k values to evaluate (e.g., range(2, max_k+1)).

        Returns:
            np.ndarray:
                An array of "elbow scores" computed from the second derivative of inertia
                normalized by the first derivative. Higher values indicate a stronger elbow.
                If there are insufficient points to compute second derivatives, returns an empty array.
        """
        inertias = []
        for k in k_values:
            kmeans = MiniBatchKMeans(
                n_clusters=k, random_state=self.random_state, batch_size=1000
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        inertias = np.array(inertias)
        diffs = np.diff(inertias)
        if len(diffs) < 2:
            return np.zeros(0)

        # Second derivative
        diffs_r = np.diff(diffs)

        # Avoid division by zero or extremely small values
        valid_mask = np.abs(diffs[1:]) > 1e-10
        elbow_score = np.zeros_like(diffs_r)
        elbow_score[valid_mask] = diffs_r[valid_mask] / np.abs(diffs[1:][valid_mask])

        return elbow_score

    def _calculate_silhouette_scores(
        self, X: np.ndarray, k_values: range
    ) -> np.ndarray:
        """
        Calculate silhouette scores for each candidate k in k_values.

        Args:
            X (np.ndarray):
                Input data of shape (n_samples, n_features).
            k_values (range):
                Range of k values to evaluate (e.g., range(2, max_k+1)).

        Returns:
            np.ndarray:
                Silhouette scores for each candidate k. A higher silhouette score indicates
                better separation between clusters.
        """
        silhouette_scores = []
        sample_size = min(1000, X.shape[0])
        rng = np.random.default_rng(self.random_state)

        for k in k_values:
            kmeans = MiniBatchKMeans(
                n_clusters=k, random_state=self.random_state, batch_size=1000
            )
            labels = kmeans.fit_predict(X)

            # Silhouette is undefined for a single cluster
            if len(np.unique(labels)) <= 1:
                silhouette_scores.append(-1)
                continue

            # Subsample to speed computations if dataset is very large
            if X.shape[0] > sample_size:
                indices = rng.choice(X.shape[0], sample_size, replace=False)
                X_sample = X[indices]
                labels_sample = labels[indices]
            else:
                X_sample = X
                labels_sample = labels

            score = silhouette_score(X_sample, labels_sample)
            silhouette_scores.append(score)

        return np.array(silhouette_scores)

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the K-Means model to the data. If 'n_clusters' is set by the user, it uses that directly.
        Otherwise, it performs automatic selection of the optimal number of clusters using both
        the silhouette and elbow methods.

        Args:
            X (np.ndarray):
                Scaled feature matrix of shape (n_samples, n_features).
        """
        if self.n_clusters is not None:
            # User-specified number of clusters
            self.optimal_k = self.n_clusters
        else:
            # Automatic selection
            k_values = range(2, self.max_k + 1)
            silhouette_scores = self._calculate_silhouette_scores(X, k_values)
            elbow_scores = self._calculate_elbow_scores(X, k_values)

            # Fallback if we can't compute elbow for some reason
            if len(k_values) <= 2 or len(elbow_scores) == 0:
                self.optimal_k = 2
            else:
                optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
                optimal_k_elbow = k_values[:-2][np.argmax(elbow_scores)]

                print(f"Optimal k (silhouette) = {optimal_k_silhouette}")
                print(f"Optimal k (elbow)      = {optimal_k_elbow}")

                self.optimal_k = optimal_k_elbow

        # Fit final model with chosen k
        self.model = MiniBatchKMeans(
            n_clusters=self.optimal_k, random_state=self.random_state, batch_size=1000
        )
        self.labels = self.model.fit_predict(X)
        self.n_clusters_ = len(np.unique(self.labels))

        self.inertia_ = self.model.inertia_
        self.cluster_centers_ = self.model.cluster_centers_

        print(f"\nKMeans fitted with {self.optimal_k} clusters.")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Retrieve key parameters and results from the fitted K-Means model.

        Returns:
            Dict[str, Any]:
                - max_k (int): Maximum clusters originally specified for auto-selection
                - n_clusters (int): The actual number of clusters used in the final model
                - inertia_ (float): Final within-cluster sum of squares (inertia)
                - cluster_centers_ (Optional[List[List[float]]]): Cluster centers as a list of lists
        """
        return {
            "max_k": self.max_k,
            "n_clusters": self.n_clusters_,
            "inertia_": self.inertia_,
            "cluster_centers_": (
                self.cluster_centers_.tolist()
                if self.cluster_centers_ is not None
                else None
            ),
        }
