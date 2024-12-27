"""DBSCAN Clustering Implementation.

This module provides a DBSCAN clustering implementation.
"""
# Standard library imports
from typing import Dict, Any

# Third-party imports
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Local imports
from base import Clustering
   
class Dbscan(Clustering):
    """
    This class implements the DBSCAN clustering algorithm with optional automatic
    estimation of the `eps` and `min_samples` parameters. After estimation, it
    allows the user to input their own values for these parameters before fitting
    the model.
    """

    def __init__(self) -> None:
        """
        Initialize the DBSCAN clustering model.

        Args:
            eps (float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int, optional): The number of samples in a neighborhood for a point to be considered a core point.
        """
        super().__init__("dbscan")
        self.eps = None
        self.min_samples = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the DBSCAN model to the data.

        Args:
            X (np.ndarray): Input data array.
        """
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Use a heuristic to determine min_samples (k)
        self.min_samples = max(5, int(np.log2(n_samples)) + 1)

        # Use a sample to estimate eps
        sample_size = min(1000, n_samples)
        indices = rng.choice(n_samples, sample_size, replace=False)
        X_sample = X[indices]
            
        # Compute k-distance graph
        nbrs = NearestNeighbors(n_neighbors=self.min_samples)
        nbrs.fit(X_sample)
        distances, _ = nbrs.kneighbors(X_sample)
        k_distances = distances[:, -1]
        self.eps = np.median(k_distances)

        print(f"Estimated parameters:")
        print(f"eps = {self.eps:.4f}, min_samples = {self.min_samples}")

        user_input = input("Do you want to input your own 'eps' and 'min_samples' values? (y/n): ").strip().lower()
        if user_input == 'y':
            eps_input = input(f"Enter 'eps' value (current estimate is {self.eps:.4f}): ")
            min_samples_input = input(f"Enter 'min_samples' value (current estimate is {self.min_samples}): ")
            self.eps = float(eps_input)
            self.min_samples = int(min_samples_input)

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(X)

        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = np.count_nonzero(self.labels == -1)

        print(f"DBSCAN fitted with eps = {self.eps:.4f}, min_samples = {self.min_samples}")
        print(f"Number of clusters found: {self.n_clusters}")
        print(f"Number of noise points: {self.n_noise}")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters of the fitted DBSCAN model.

        Returns:
            Dict[str, Any]: Dictionary containing model parameters and results.
        """
        return {
            "model_type": self.model_type,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
        }
