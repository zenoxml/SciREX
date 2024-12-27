"""Agglomerative Clustering implementation.

This module provides an Agglomerative Clustering implementation using scikit-learn.
"""

# Standard library imports
from typing import Dict, Any

# Third-party imports
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Local imports
from base import Clustering

class Agglomerative(Clustering):
    """
    Agglomerative Clustering with automatic selection of optimal number of clusters using silhouette scores.
    """

    def __init__(self, max_k: int = 10) -> None:
        """
        Initialize the Agglomerative clustering class.

        Args:
            max_k (int): Maximum number of clusters to try. Defaults to 10.
        """
        super().__init__("agglomerative")
        self.max_k = max_k
        self.optimal_k = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit method for Agglomerative Clustering with efficient sampling for silhouette score calculation.
        
        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features)
        """
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]
        k_values = range(2, self.max_k + 1)
        silhouettes = []
        
        silhouette_sample_size = min(1000, n_samples)
            
        for k in k_values:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage='average'  
            )
            labels = model.fit_predict(X)
            unique_labels = np.unique(labels)
            
            if len(unique_labels) <= 1:
                silhouettes.append(-1)
                continue
                
            if n_samples > silhouette_sample_size:
                indices = np.random.choice(n_samples, silhouette_sample_size, replace=False)
                silhouette_avg = silhouette_score(X[indices], labels[indices])
            else:
                silhouette_avg = silhouette_score(X, labels)
                
            silhouettes.append(silhouette_avg)

        self.optimal_k = k_values[np.argmax(silhouettes)]
        print(f"Estimated optimal number of clusters (optimal_k): {self.optimal_k}")
        
        user_input = input("Do you want to input your own number of clusters? (y/n): ").strip().lower()
        if user_input == 'y':
            k_input = int(input(f"Enter the number of clusters (k), current estimate is {self.optimal_k}: "))
            if k_input >= 2:
                self.optimal_k = k_input
            else:
                print("Number of clusters must be at least 2. Using estimated optimal_k.")
        
        self.model = AgglomerativeClustering(
            n_clusters=self.optimal_k,
            linkage='average'
        )
        self.labels = self.model.fit_predict(X)
        self.n_clusters = len(np.unique(self.labels))
        print(f"Agglomerative Clustering fitted with k = {self.optimal_k}")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters of the fitted Agglomerative Clustering model.

        Returns:
            Dict[str, Any]: Dictionary containing model parameters and results.
        """
        return {
            "model_type": self.model_type,
            "n_clusters": self.optimal_k
        }