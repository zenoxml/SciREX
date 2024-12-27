"""K-means clustering implementation.

This module provides a Kmeans clustering implementation.
"""
# Standard library imports
from typing import Dict, Any

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
        max_k (int): Maximum number of clusters to consider
        optimal_k (Optional[int]): Estimated optimal number of clusters after fitting
    """
    def __init__(self, max_k: int = 10) -> None:
        """
        Initialize the Kmeans clustering class.

        Args:
            max_k (int): Maximum number of clusters to try. Defaults to 10.
        """
        super().__init__("kmeans")
        self.max_k = max_k

    def _calculate_elbow_scores(self, X: np.ndarray, k_values: range) -> np.ndarray:
        """
        Calculate inertia scores for elbow method.

        Args:
            X (np.ndarray): Input data
            k_values (range): Range of k values to try

        Returns:
            np.ndarray: Array of inertia scores
        """
        inertias = []
        for k in k_values:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.random_state,
                batch_size=1000
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Calculate the optimal k using the elbow method
        inertias = np.array(inertias)
        diffs = np.diff(inertias)
        diffs_r = np.diff(diffs)
        elbow_score = diffs_r/np.abs(diffs[1:])
        return np.array(elbow_score)

    def _calculate_silhouette_scores(self, X: np.ndarray, k_values: range) -> np.ndarray:
        """
        Calculate silhouette scores.

        Args:
            X (np.ndarray): Input data
            k_values (range): Range of k values to try

        Returns:
            np.ndarray: Array of silhouette scores
        """
        silhouette_scores = []
        sample_size = min(1000, X.shape[0])
        rng = np.random.default_rng(self.random_state)

        for k in k_values:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.random_state,
                batch_size=1000
            )
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) <= 1:
                silhouette_scores.append(-1)
                continue

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
        Fit the kmeans model to the data.
        
        Args:
            X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features)
        """
        k_values = range(2, self.max_k + 1)
        
        silhouette_scores = self._calculate_silhouette_scores(X, k_values)
        elbow_scores = self._calculate_elbow_scores(X, k_values)
        
        optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
        optimal_k_elbow = k_values[:-2][np.argmax(elbow_scores)]
        
        print(f"Optimal k from silhouette score: {optimal_k_silhouette}")
        print(f"Optimal k from elbow method: {optimal_k_elbow}")
        
        print("\nChoose k for the model?")
        print("1: Silhouette method")
        print("2: Elbow method")
        print("3: Input custom value")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '1':
            self.optimal_k = optimal_k_silhouette
        elif choice == '2':
            self.optimal_k = optimal_k_elbow
        elif choice == '3':
            k_input = int(input(f"Enter the number of clusters (k): "))
            if k_input >= 2:
                self.optimal_k = k_input
            else:
                print("Number of clusters must be at least 2. Using silhouette method's optimal k.")
                self.optimal_k = optimal_k_silhouette
        else:
            print("Invalid choice. Using silhouette method's optimal k.")
            self.optimal_k = optimal_k_silhouette

        self.model = MiniBatchKMeans(
            n_clusters=self.optimal_k,
            random_state=self.random_state,
            batch_size=1000
        )
        self.labels = self.model.fit_predict(X)
        self.n_clusters = len(np.unique(self.labels))
        
        print(f"\nKMeans fitted with {self.optimal_k} clusters")

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters of the fitted K-Means model.

        Returns:
            Dict[str, Any]: Dictionary containing model parameters and results.
        """
        return {
            "model_type": self.model_type,
            "max_k": self.max_k,
            "n_clusters": self.n_clusters
        }


