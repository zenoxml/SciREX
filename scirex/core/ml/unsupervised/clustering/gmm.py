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
    Module: gmm.py

    This module provides a Gaussian Mixture Model (GMM) clustering implementation using
    scikit-learn's GaussianMixture. The Gmm class automatically estimates the optimal
    number of components (clusters) via silhouette scores.

    Classes:
        Gmm: Gaussian Mixture Model clustering with automatic component selection.

    Dependencies:
        - numpy
        - sklearn.mixture.GaussianMixture
        - sklearn.metrics.silhouette_score
        - base.py (Clustering)

    Key Features:
        - Scans [2..max_k] for the best silhouette score
        - Final model is stored, along with predicted cluster labels
        - Ties into the base `Clustering` for plotting/metrics

    Authors:
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Local imports
from .base import Clustering


class Gmm(Clustering):
    """
    Gaussian Mixture Model clustering with automatic component selection via silhouette scores.

    Attributes:
        max_k (int): Maximum number of components (clusters) to consider when searching
                     for the optimal mixture size.
        optimal_k (Optional[int]): The chosen (user-verified) number of components after fitting.
        model (Optional[GaussianMixture]): The underlying scikit-learn GaussianMixture model.
        labels (Optional[np.ndarray]): Cluster/component labels for each data point after fitting.
    """

    def __init__(self, max_k: int = 10) -> None:
        """
        Initialize the GMM clustering.

        Args:
            max_k (int, optional): Maximum number of components to try. Defaults to 10.
        """
        super().__init__("gmm")
        self.max_k = max_k
        self.optimal_k: Optional[int] = None
        self.model: Optional[GaussianMixture] = None
        self.labels: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the GMM model to the data with automatic component selection.

        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)
        n_samples, n_features = X.shape

        k_values = range(2, self.max_k + 1)
        silhouettes = []

        # Prepare a random generator for subsampling
        rng = np.random.default_rng(self.random_state)
        silhouette_sample_size = min(1000, n_samples)

        # Evaluate silhouette scores for each candidate k
        for k in k_values:
            gmm = GaussianMixture(n_components=k, random_state=self.random_state)
            gmm.fit(X)
            labels = gmm.predict(X)

            # Ensure at least 2 distinct clusters
            if len(np.unique(labels)) > 1:
                # Subsample if large
                if n_samples > silhouette_sample_size:
                    sample_indices = rng.choice(
                        n_samples, silhouette_sample_size, replace=False
                    )
                    X_sample_silhouette = X[sample_indices]
                    labels_sample = labels[sample_indices]
                else:
                    X_sample_silhouette = X
                    labels_sample = labels

                silhouettes.append(
                    silhouette_score(
                        X_sample_silhouette.reshape(-1, n_features), labels_sample
                    )
                )
            else:
                silhouettes.append(-1)  # Invalid silhouette if only one cluster

        # Pick k with best silhouette
        self.optimal_k = k_values[np.argmax(silhouettes)]
        print(f"Estimated optimal number of clusters (optimal_k): {self.optimal_k}")

        # Final fit
        self.model = GaussianMixture(
            n_components=self.optimal_k, random_state=self.random_state
        )
        self.labels = self.model.fit_predict(X)
        print(f"GMM fitted with optimal_k = {self.optimal_k}")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters of the fitted GMM model.

        Returns:
            Dict[str, Any]:
                A dictionary containing:
                - model_type (str)
                - optimal_k (int)
                - max_k (int)
        """
        return {
            "model_type": self.model_type,
            "optimal_k": self.optimal_k,
            "max_k": self.max_k,
        }
