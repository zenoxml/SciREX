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

"""K-means clustering implementation.

This module provides a Kmeans clustering implementation. 

The Kmeans class inherits from a generic Clustering base class and offers:
- Automatic selection of the optimal number of clusters via silhouette or elbow methods.
- Option for the user to input a custom number of clusters if desired.
"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Local imports
from base import Clustering


class Kmeans(Clustering):
    """
    K-Means clustering with automatic selection of the optimal number of clusters.

    Attributes:
        max_k (int): The maximum number of clusters to consider
                     when scanning for the optimal cluster count.
        optimal_k (Optional[int]): The chosen number of clusters after fitting.
        model (MiniBatchKMeans): The underlying scikit-learn MiniBatchKMeans model.
        labels (np.ndarray): Cluster labels for each data point after fitting.
        n_clusters (int): The actual number of clusters used by the final fitted model.
    """

    def __init__(self, max_k: int = 10) -> None:
        """
        Initialize the Kmeans clustering class.

        Args:
            max_k (int, optional): Maximum number of clusters to try.
                                   Defaults to 10.
        """
        super().__init__("kmeans")
        self.max_k = max_k
        self.optimal_k: Optional[int] = None
        self.model: Optional[MiniBatchKMeans] = None
        self.labels: Optional[np.ndarray] = None
        self.n_clusters: Optional[int] = None

    def _calculate_elbow_scores(self, X: np.ndarray, k_values: range) -> np.ndarray:
        """
        Calculate inertia scores used by the elbow method,
        then compute a 'second derivative' style metric to find the elbow.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            k_values (range): Range of k values to evaluate, e.g. range(2, max_k+1).

        Returns:
            np.ndarray: An array of derived elbow scores (the more positive
                        the score, the stronger the elbow).
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
        # A simple ratio to quantify the change
        elbow_score = diffs_r / np.abs(diffs[1:])

        return np.array(elbow_score)

    def _calculate_silhouette_scores(
        self, X: np.ndarray, k_values: range
    ) -> np.ndarray:
        """
        Calculate silhouette scores for each candidate k in k_values.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            k_values (range): Range of k values to evaluate, e.g. range(2, max_k+1).

        Returns:
            np.ndarray: An array of silhouette scores for each k.
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

            silhouette_scores.append(silhouette_score(X_sample, labels_sample))

        return np.array(silhouette_scores)

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the K-Means model to the data with automatic cluster selection.

        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
        """
        k_values = range(2, self.max_k + 1)

        silhouette_scores = self._calculate_silhouette_scores(X, k_values)
        elbow_scores = self._calculate_elbow_scores(X, k_values)

        # The best k from silhouette is where the silhouette score is maximal
        optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]

        # The best k from elbow is the index where the elbow score is maximal
        optimal_k_elbow = k_values[:-2][np.argmax(elbow_scores)]

        # Show the suggestions
        print(f"Optimal k from silhouette score: {optimal_k_silhouette}")
        print(f"Optimal k from elbow method: {optimal_k_elbow}")

        print("\nChoose k for the model?")
        print("1: Silhouette method")
        print("2: Elbow method")
        print("3: Input custom value")

        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            self.optimal_k = optimal_k_silhouette
        elif choice == "2":
            self.optimal_k = optimal_k_elbow
        elif choice == "3":
            k_input = int(input("Enter the number of clusters (k): "))
            if k_input >= 2:
                self.optimal_k = k_input
            else:
                print(
                    "Number of clusters must be at least 2. Using silhouette method's optimal k."
                )
                self.optimal_k = optimal_k_silhouette
        else:
            print("Invalid choice. Using silhouette method's optimal k.")
            self.optimal_k = optimal_k_silhouette

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
                A dictionary containing:
                - model_type (str): The name of the clustering model.
                - max_k (int): The maximum number of clusters originally specified.
                - n_clusters (int): The final number of clusters used.
        """
        return {
            "model_type": self.model_type,
            "max_k": self.max_k,
            "n_clusters": self.n_clusters,
        }
