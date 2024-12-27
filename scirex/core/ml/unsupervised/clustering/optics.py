"""OPTICS (Ordering Points To Identify Clustering Structure) implementation.

This module provides an implementation of the OPTICS clustering algorithm.
"""
from typing import Dict,Any
import numpy as np
from sklearn.cluster import OPTICS as SKLearnOPTICS

from base import Clustering

class Optics(Clustering):
    """
    OPTICS clustering implementation with parameter estimation.
    """

    def __init__(self) -> None:
        """Initialize the OPTICS clustering model."""
        super().__init__("optics")

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the OPTICS model to the data.

        Args:
            X (np.ndarray): Input data array.
        """
        X = X.astype(np.float32, copy=False)
        n_samples, n_features = X.shape

        self.min_samples = max(5, int(np.log2(n_samples)) + 1)
        self.min_cluster_size = max(
            self.min_samples,
            min(50, int(0.05 * n_samples)) 
        )

        print(f"Estimated parameters:")
        print(f"min_cluster_size = {self.min_cluster_size}, min_samples = {self.min_samples}")

        user_input = input("Do you want to input your own parameters? (y/n): ").strip().lower()
        if user_input == 'y':
            min_cluster_size_input = input(f"Enter 'min_cluster_size' value (current estimate is {self.min_cluster_size}): ")
            min_samples_input = input(f"Enter 'min_samples' value (current estimate is {self.min_samples}): ")
            self.min_cluster_size = int(min_cluster_size_input)
            self.min_samples = int(min_samples_input)

        self.model = SKLearnOPTICS(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            xi=0.05,
            metric='euclidean',
            n_jobs=-1
        )
        
        self.model.fit(X)
        self.labels = self.model.labels_

        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = np.count_nonzero(self.labels == -1)

        if self.n_clusters == 0:
            print("Warning: No clusters found. All points labeled as noise.")
        else:
            print(f"OPTICS found {self.n_clusters} clusters")
            print(f"Noise points: {self.n_noise} ({(self.n_noise/n_samples)*100:.1f}%)")
            
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the model parameters and clustering results.

        Returns:
            Dict[str, Any]: Dictionary containing model parameters and results.
        """
        return {
            "model_type": self.model_type,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples
        }