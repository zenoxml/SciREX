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
scikit-learn's GaussianMixture. 

It optionally allows a user-defined number of components or automatically scans
[2..max_k] for the best silhouette score.

Classes:
    Gmm: Gaussian Mixture Model clustering with optional user-specified n_components
         or silhouette-based auto selection.

Dependencies:
    - numpy
    - sklearn.mixture.GaussianMixture
    - sklearn.metrics.silhouette_score
    - base.py (Clustering)

Key Features:
    - Automatic scanning of [2..max_k] for best silhouette score if n_components is None
    - Final model is stored, along with predicted cluster labels
    - Ties into the base `Clustering` for plotting/metrics

Authors:
    - Debajyoti Sahoo (debajyotis@iisc.ac.in)

Version Info:
    - 28/Dec/2024: Initial release
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
    Gaussian Mixture Model clustering with optional user-defined 'n_components'
    or automatic silhouette-based selection.

    Attributes:
        n_components (Optional[int]):
            The actual number of components used in the final fitted model.
            If provided, the class will skip auto-selection
            and directly use this many mixture components.
        max_k (int):
            Maximum number of components to consider for auto selection if n_components is None.
        labels (Optional[np.ndarray]):
            Cluster/component labels for each data point after fitting.
    """

    def __init__(self, n_components: Optional[int] = None, max_k: int = 10) -> None:
        """
        Initialize the Gmm clustering class.

        Args:
            n_components (Optional[int], optional):
                If provided, the model will directly use this many Gaussian components.
                Otherwise, it scans [2..max_k] for the best silhouette score. Defaults to None.
            max_k (int, optional):
                Maximum components to try for auto selection if n_components is None. Defaults to 10.
        """
        super().__init__("gmm")
        self.n_components = n_components
        self.max_k = max_k

        # Populated after fitting
        self.labels: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.model: Optional[GaussianMixture] = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the GMM model to the data.

        If user-defined n_components is set, skip auto selection.
        Otherwise, compute silhouette scores across [2..max_k]
        and pick the best.

        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
        """
        X = X.astype(np.float32, copy=False)
        n_samples, n_features = X.shape

        if self.n_components is not None:
            # Use user-specified
            self.n_components_ = self.n_components
            print(f"Fitting GMM with user-defined n_components={self.n_components_}.\n")
        else:
            # Automatic silhouette-based selection
            k_values = range(2, self.max_k + 1)
            silhouettes = []

            # Subsampling for silhouette
            rng = np.random.default_rng(self.random_state)
            sample_size = min(1000, n_samples)

            for k in k_values:
                gmm = GaussianMixture(n_components=k, random_state=self.random_state)
                gmm.fit(X)
                labels_candidate = gmm.predict(X)

                # Must have at least 2 distinct clusters for silhouette
                if len(np.unique(labels_candidate)) > 1:
                    if n_samples > sample_size:
                        indices = rng.choice(n_samples, sample_size, replace=False)
                        X_sample = X[indices]
                        labels_sample = labels_candidate[indices]
                    else:
                        X_sample = X
                        labels_sample = labels_candidate
                    score = silhouette_score(X_sample, labels_sample)
                else:
                    score = -1  # invalid silhouette

                silhouettes.append(score)

            best_k = k_values[np.argmax(silhouettes)]
            self.n_components_ = best_k
            print(f"Optimal k (silhouette) = {best_k}\n")

        self.model = GaussianMixture(
            n_components=self.n_components_, random_state=self.random_state
        )
        self.labels = self.model.fit_predict(X)

        print(f"GMM fitted with n_components={self.n_components_}.\n")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters/results of the fitted GMM model.

        Returns:
            Dict[str, Any]:
                - n_components (int): The final number of components used
                - max_k (int): The maximum considered if auto
        """
        return {"n_components": self.n_components_, "max_k": self.max_k}
