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
      - Automatic selection of the optimal number of clusters via silhouette or elbow methods
      - Option for the user to input a custom number of clusters if desired

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
    K-Means clustering with automatic selection of the optimal number of clusters.

    Attributes:
        max_k (int): The maximum number of clusters to consider when scanning for the optimal cluster count.
        optimal_k (Optional[int]): The chosen number of clusters after fitting.
        model (Optional[MiniBatchKMeans]): The underlying scikit-learn MiniBatchKMeans model.
        labels (Optional[np.ndarray]): Cluster labels for each data point after fitting.
        n_clusters (Optional[int]): The actual number of clusters used by the final fitted model.
    """

    def __init__(self, max_k: int = 10) -> None:
        """
        Initialize the Kmeans clustering class.

        Args:
            max_k (int, optional): Maximum number of clusters to try. Defaults to 10.
        """
        super().__init__("kmeans")
        self.max_k = max_k
        self.optimal_k: Optional[int] = None
        self.model: Optional[MiniBatchKMeans] = None
        self.labels: Optional[np.ndarray] = None
        self.n_clusters: Optional[int] = None

    def _calculate_elbow_scores(self, X: np.ndarray, k_values: range) -> np.ndarray:
        """
        Calculate inertia scores (in elbow method), then compute a "second derivative" style metric.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            k_values (range): Range of k values to evaluate, e.g. range(2, max_k+1).

        Returns:
            np.ndarray: A numeric array of elbow scores, where higher indicates a stronger elbow.
        """
        inertias = []
        for k in k_values:
            kmeans = MiniBatchKMeans(
                n_clusters=k, random_state=self.random_state, batch_size=1000
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        inertias = np.array(inertias)

        # First derivative of inertia
        diffs = np.diff(inertias)
        # Second derivative
        diffs_r = np.diff(diffs)
        # A ratio capturing second derivative relative to the first derivative
        elbow_score = diffs_r / np.abs(diffs[1:])

        return elbow_score

    def _calculate_silhouette_scores(
        self, X: np.ndarray, k_values: range
    ) -> np.ndarray:
        """
        Calculate silhouette scores for each candidate k in k_values.

        Steps:
          1. For each k, fit a MiniBatchKMeans and obtain cluster labels.
          2. Subsample large datasets (up to 1000 points) to speed silhouette calculations.
          3. Return an array of silhouette scores, one per k.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            k_values (range): Range of k values to evaluate, e.g. range(2, max_k+1).

        Returns:
            np.ndarray: Silhouette scores for each candidate k.
        """
        silhouette_scores = []
        sample_size = min(1000, X.shape[0])
        rng = np.random.default_rng(self.random_state)

        for k in k_values:
            kmeans = MiniBatchKMeans(
                n_clusters=k, random_state=self.random_state, batch_size=1000
            )
            labels = kmeans.fit_predict(X)

            # If there's only one cluster, silhouette is invalid
            if len(np.unique(labels)) <= 1:
                silhouette_scores.append(-1)
                continue

            # Subsample if the dataset is very large
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
        Fit the K-Means model to the data with automatic cluster selection.

        Steps:
          1. Define k_values as range(2..max_k).
          2. Compute silhouette scores and elbow scores across those k_values.
          3. Determine an optimal k from either method (this example picks elbow).
          4. Fit a final MiniBatchKMeans with the chosen k, storing labels and cluster count.

        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
        """
        k_values = range(2, self.max_k + 1)

        # Evaluate silhouette and elbow
        silhouette_scores = self._calculate_silhouette_scores(X, k_values)
        elbow_scores = self._calculate_elbow_scores(X, k_values)

        # Best k from silhouette = index of maximum silhouette
        optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]

        # Best k from elbow = index of maximum elbow score
        # Note: elbow_scores has length (len(k_values) - 2), so we align indexing
        optimal_k_elbow = k_values[:-2][np.argmax(elbow_scores)]

        # Choose whichever approach you'd like; here using elbow by default
        self.optimal_k = optimal_k_elbow
        print(f"Optimal k from silhouette: {optimal_k_silhouette}")
        print(f"Optimal k from elbow method: {self.optimal_k}")

        self.model = MiniBatchKMeans(
            n_clusters=self.optimal_k, random_state=self.random_state, batch_size=1000
        )
        self.labels = self.model.fit_predict(X)
        self.n_clusters = len(np.unique(self.labels))

        print(f"\nKMeans fitted with {self.optimal_k} clusters")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters of the fitted K-Means model.

        Returns:
            Dict[str, Any]:
                - model_type (str): "kmeans"
                - max_k (int): The maximum number of clusters originally specified
                - n_clusters (int): The final number of clusters used
        """
        return {
            "model_type": self.model_type,
            "max_k": self.max_k,
            "n_clusters": self.n_clusters,
        }
