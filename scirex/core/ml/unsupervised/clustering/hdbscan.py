"""HDBSCAN (Hierarchical Density-Based Spatial Clustering) implementation.

This module provides an implementation of HDBSCAN clustering algorithm.

"""
# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
from sklearn.cluster import HDBSCAN

# Local imports
from base import Clustering

class Hdbscan(Clustering):
    def __init__(self) -> None:
        """
        Initialize the HDBSCAN clustering model.
        """
        super().__init__("hdbscan")
        self.min_cluster_size = None
        self.min_samples = None

    def fit(self, X: np.ndarray, expected_clusters: Optional[int] = None) -> None:
        """
        Fit the HDBSCAN model to the data.

        Args:
            X (np.ndarray): Input data array.
            expected_clusters (Optional[int]): Expected number of clusters.
        """
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]

        # Estimate parameters
        self.min_samples = max(1, int(np.log(n_samples)))
        self.min_cluster_size = max(5, int(0.02 * n_samples))

        print(f"Estimated parameters:")
        print(f"min_cluster_size = {self.min_cluster_size}, min_samples = {self.min_samples}")

        user_input = input("Do you want to input your own parameters? (y/n): ").strip().lower()
        if user_input == 'y':
            try:
                min_cluster_size_input = input(f"Enter min_cluster_size (current: {self.min_cluster_size}): ")
                min_samples_input = input(f"Enter min_samples (current: {self.min_samples}): ")
                self.min_cluster_size = int(min_cluster_size_input) if min_cluster_size_input.strip() else self.min_cluster_size
                self.min_samples = int(min_samples_input) if min_samples_input.strip() else self.min_samples
            except ValueError:
                print("Invalid input. Using estimated parameters.")

        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method='eom'
        )
        
        self.model.fit(X)
        self.labels = self.model.labels_
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = np.count_nonzero(self.labels == -1)

        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of noise points: {self.n_noise}")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the model parameters and clustering results.

        Returns:
            Dict[str, Any]: Dictionary containing model parameters and results.
        """
        return {
            "model_type": self.model_type,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
        }