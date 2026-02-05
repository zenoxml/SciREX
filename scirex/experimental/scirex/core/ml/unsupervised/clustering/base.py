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
    Module: base.py

    This module provides the abstract base class for all clustering implementations in SciREX.
    It defines shared functionality for:
        - Data preparation (loading from CSV and standard scaling)
        - Clustering metric computation (silhouette, calinski-harabasz, davies-bouldin)
        - 2D plotting using PCA for visualization

    Classes:
        Clustering: Abstract base class that outlines common behavior for clustering algorithms.

    Dependencies:
        - numpy, pandas, matplotlib, sklearn
        - abc, pathlib, time, typing (for structural and type support)

    Key Features:
        - Consistent interface for loading and preparing data
        - Standard approach to computing and returning clustering metrics
        - PCA-based 2D plotting routine for visualizing clusters in two dimensions
        - Enforces subclasses to implement `fit` and `get_model_params`

    Authors:
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

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

    Attributes:
        model_type (str): The name or identifier of the clustering model (e.g., "kmeans", "dbscan").
        random_state (int): Random seed for reproducibility.
        labels (Optional[np.ndarray]): Array of cluster labels assigned to each sample after fitting.
        plots_dir (Path): Directory where cluster plots will be saved.
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

        Steps:
          1. If X has >=2 features, run PCA to reduce it to 2 components.
          2. If X has only 1 feature, it is zero-padded to form a 2D embedding for plotting.
          3. Each unique cluster label is plotted with a distinct color.
          4. The figure is saved in `self.plots_dir` as `cluster_plot_{self.model_type}.png`.

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
