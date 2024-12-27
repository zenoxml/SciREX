"""
Base clustering implementation.

This module provides the abstract base class for all clustering implementations.It defines shared functionality for data preparation and scoring.

"""
# Standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import time

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

class Clustering(ABC):
    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
    ) -> None:
        """
        Base clustering class that produces a 2D scatter plot using PCA.
        
        """
        self.model_type = model_type
        self.random_state = random_state

        # Directory for saving plots
        self.plots_dir = Path.cwd() / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Subclasses must populate self.labels after fitting
        self.labels: Optional[np.ndarray] = None

    def prepare_data(self, path: str) -> np.ndarray:
        """
        Load and preprocess data from a CSV file, returning a scaled NumPy array.
        """
        df = pd.read_csv(Path(path))
        df = df.dropna()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if numeric_columns.empty:
            raise ValueError("No numeric columns found in the data.")
        features = df[numeric_columns].values
        return StandardScaler().fit_transform(features)

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the clustering model on a preprocessed dataset.
        """
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Return model parameters for logging or debugging."""
        pass

    def plots(self, X: np.ndarray, labels: np.ndarray) -> Tuple[Figure, Path]:
        """
        Create a 2D scatter plot of clusters using PCA.
        """
        n_features = X.shape[1]
        if n_features >= 2:
            pca = PCA(n_components=2, random_state=self.random_state)
            X_pca = pca.fit_transform(X)
        else:
            print("Dataset has only 1 feature. Zero-padding to 2D for visualization.")
            pca = PCA(n_components=1, random_state=self.random_state)
            X_1d = pca.fit_transform(X)
            X_pca = np.hstack([X_1d, np.zeros((X_1d.shape[0], 1))])

        unique_labels = np.unique(labels)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            mask = (labels == label)
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                color=("k" if label == -1 else color),
                label=f"Cluster {label}",
                alpha=0.7,
                s=60,
            )
        ax.set_xlabel("PCA Component 1", fontsize=10)
        ax.set_ylabel("PCA Component 2", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_title(
            f"{self.model_type.upper()} Clustering Results (2D PCA)",
            fontsize=12,
            pad=15,
        )
        plt.tight_layout()

        # Save the figure
        plot_path = self.plots_dir / f"cluster_plot_{self.model_type}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig) 
        return fig, plot_path

    def run(
        self,
        data: Optional[np.ndarray] = None,
        path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete clustering pipeline.

        Args:
            data: Optional preprocessed data array.
            path: Optional path to CSV data file.

        Returns:
            Dictionary containing clustering results.
        """
        if data is None and path is None:
            raise ValueError("Either 'data' or 'path' must be provided.")

        X = data if data is not None else self.prepare_data(path)

        # Fit the model
        start_time = time.perf_counter()
        self.fit(X)
        time_taken = time.perf_counter() - start_time

        if self.labels is None:
            raise ValueError("Model has not assigned labels. Did you implement fit()?")

        silhouette = silhouette_score(X, self.labels)
        calinski_harabasz = calinski_harabasz_score(X, self.labels)
        davies_bouldin = davies_bouldin_score(X, self.labels)

        return {
            "params": self.get_model_params(),
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski_harabasz,
            "davies_bouldin_score": davies_bouldin,
            "time_taken": time_taken
        }