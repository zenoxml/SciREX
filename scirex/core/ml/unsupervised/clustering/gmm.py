"""Gaussian Mixture Model (GMM) clustering implementation.

This module provides a Gaussian Mixture Model (GMM) clustering implementation.
"""
# Standard library imports
from typing import Dict, Any

# Third-party imports
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Local imports
from base import Clustering

class Gmm(Clustering):
    """
    This implementation uses scikit-learn's GaussianMixture algorithm with automatic
    estimation of the optimal number of components using silhouette scores.

    Attributes:
        max_k (int): Maximum number of components to consider.
    """

    def __init__(
        self,
        max_k: int = 10
    ) -> None:
        """Initialize GMM clustering.

        Args:
            max_k: Maximum number of components to try.
        """
        super().__init__("gmm")
        self.max_k = max_k

    def fit(self, X: np.ndarray) -> None:
        """Fit the GMM model.

        Args:
            X: Scaled feature matrix.
        """
        X = X.astype(np.float32, copy=False)
        n_samples, n_features = X.shape
        
        k_values = range(2, self.max_k + 1)
        silhouettes = []

        silhouette_sample_size = min(1000, n_samples)
        rng = np.random.default_rng(self.random_state)

        for k in k_values:
            gmm = GaussianMixture(n_components=k, random_state=self.random_state)
            gmm.fit(X)
            labels = gmm.predict(X)
            if len(np.unique(labels)) > 1:
                if n_samples > silhouette_sample_size:
                    X_sample_silhouette = X[rng.choice(n_samples, silhouette_sample_size, replace=False)]
                    labels_sample = labels[rng.choice(n_samples, silhouette_sample_size, replace=False)]
                else:
                    X_sample_silhouette = X
                    labels_sample = labels
                silhouettes.append(silhouette_score(X_sample_silhouette.reshape(-1, n_features), labels_sample))
            else:
                silhouettes.append(-1)

        self.optimal_k = k_values[np.argmax(silhouettes)]

        print(f"Estimated optimal number of clusters (optimal_k): {self.optimal_k}")

        user_input = input("Do you want to input your own number of clusters? (y/n): ").strip().lower()
        if user_input == 'y':
            k_input = int(input(f"Enter the number of clusters (k), current estimate is {self.optimal_k}: "))
            if k_input >= 2:
                self.optimal_k = k_input
            else:
                print("Number of clusters must be at least 2. Using estimated optimal_k.")

        self.model = GaussianMixture(n_components=self.optimal_k, random_state=self.random_state)
        self.labels = self.model.fit_predict(X)
        print(f"GMM fitted with optimal_k = {self.optimal_k}")

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "optimal_k": self.optimal_k,
            "max_k": self.max_k
        }
