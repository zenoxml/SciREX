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

"""
Base clustering implementation.

This module provides the abstract base class for all clustering implementations.
It defines shared functionality for:
 - Data preparation (loading from CSV and standard scaling)
 - Clustering metric computation (silhouette, calinski-harabasz, davies-bouldin)
 - 2D plotting using PCA for visualization

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
    """
    Abstract base class for clustering algorithms in the SciREX library.

    This class provides:
      - A consistent interface for loading and preparing data
      - A standard approach to computing and returning clustering metrics
      - A PCA-based 2D plotting routine for visualizing clusters

    Subclasses must:
      1. Implement the `fit(X: np.ndarray) -> None` method, which should populate `self.labels`.
      2. Implement the `get_model_params() -> Dict[str, Any]` method, which returns a dict
         of model parameters for logging/debugging.
    """

    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the base clustering class.

        Args:
            model_type (str): A string identifier for the clustering algorithm
                              (e.g. "kmeans", "dbscan", etc.).
            random_state (int, optional): Seed for reproducibility where applicable.
                                          Defaults to 42.

        Attributes:
            model_type (str): The name or identifier of the clustering model.
            random_state (int): Random seed for reproducibility.
            plots_dir (Path): Directory where cluster plots will be saved.
            labels (Optional[np.ndarray]): Array of cluster labels assigned to each sample
                                           after fitting. Subclasses must populate.
        """
        self.model_type = model_type
        self.random_state = random_state

        # Directory for saving plots
        self.plots_dir = Path.cwd() / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Subclasses must set self.labels after fitting
        self.labels: Optional[np.ndarray] = None

    def prepare_data(self, path: str) -> np.ndarray:
        """
        Load and preprocess data from a CSV file, returning a scaled NumPy array.

        This method:
          1. Reads the CSV file into a pandas DataFrame.
          2. Drops rows containing NaN values.
          3. Selects only numeric columns from the DataFrame.
          4. Scales these features using scikit-learn's StandardScaler.
          5. Returns the scaled values as a NumPy array.

        Args:
            path (str): Filepath to the CSV data file.

        Returns:
            np.ndarray: A 2D array of shape (n_samples, n_features) containing
                        standardized numeric data.

        Raises:
            ValueError: If no numeric columns are found in the data.
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
        Fit the clustering model on a preprocessed dataset, assigning labels to `self.labels`.

        Args:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) containing
                            the data to be clustered.

        Subclasses must implement this method. After fitting the model,
        `self.labels` should be set to an array of cluster labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Return model parameters for logging or debugging.

        Returns:
            Dict[str, Any]: A dictionary containing key model parameters and
                            potentially any learned attributes (e.g. number of clusters).
        """
        pass

    def plots(self, X: np.ndarray, labels: np.ndarray) -> Tuple[Figure, Path]:
        """
        Create a 2D scatter plot of clusters using PCA for dimensionality reduction.

        Args:
            X (np.ndarray): Data array of shape (n_samples, n_features).
            labels (np.ndarray): Cluster labels for each sample.

        Returns:
            Tuple[Figure, Path]:
              - The matplotlib Figure object.
              - The path where the figure was saved (plot_path).

        Notes:
            Subclasses typically do not override this method. Instead, they rely on the
            base implementation for consistent plotting.
        """
        n_features = X.shape[1]
        if n_features >= 2:
            pca = PCA(n_components=2, random_state=self.random_state)
            X_pca = pca.fit_transform(X)
        else:
            # If there's only 1 feature, we simulate a second dimension with zeros
            print("Dataset has only 1 feature. Zero-padding to 2D for visualization.")
            pca = PCA(n_components=1, random_state=self.random_state)
            X_1d = pca.fit_transform(X)
            X_pca = np.hstack([X_1d, np.zeros((X_1d.shape[0], 1))])

        unique_labels = np.unique(labels)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            mask = labels == label
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
        self, data: Optional[np.ndarray] = None, path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete clustering pipeline: data loading/preprocessing,
        fitting the model, and computing standard clustering metrics.

        Args:
            data (Optional[np.ndarray]): Preprocessed data array of shape (n_samples, n_features).
            path (Optional[str]): Path to a CSV file from which to read data.
                                  If `data` is not provided, this must be specified.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - "params" (Dict[str, Any]): Model parameters from `self.get_model_params()`
                - "silhouette_score" (float)
                - "calinski_harabasz_score" (float)
                - "davies_bouldin_score" (float)
                - "time_taken" (float): Fitting time in seconds

        Raises:
            ValueError: If neither `data` nor `path` is provided, or if `self.labels`
                        remains None after fitting (indicating a subclass didn't set it).
        """
        if data is None and path is None:
            raise ValueError("Either 'data' or 'path' must be provided.")

        # Load/prepare data if needed
        X = data if data is not None else self.prepare_data(path)

        # Fit the model
        self.fit(X)

        # Check labels
        if self.labels is None:
            raise ValueError("Model has not assigned labels. Did you implement fit()?")

        # Compute clustering metrics
        silhouette = silhouette_score(X, self.labels)
        calinski_harabasz = calinski_harabasz_score(X, self.labels)
        davies_bouldin = davies_bouldin_score(X, self.labels)

        # Return results
        return {
            "params": self.get_model_params(),
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski_harabasz,
            "davies_bouldin_score": davies_bouldin,
        }
