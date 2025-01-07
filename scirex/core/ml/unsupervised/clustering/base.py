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

This module provides the abstract base class for all clustering implementations
in SciREX. It consolidates:

1. Data preparation for CPU (CSV loading, NaN dropping, standard scaling).
2. Clustering metric computations (CPU-based silhouette, calinski-harabasz, davies-bouldin).
3. 2D plotting using PCA for visualization.
4. GPU data preparation and GPU-based silhouette (O(n^2) approach) to avoid
   code duplication across multiple RAPIDS-based clustering algorithms.

Classes:
    Clustering (ABC):
        - Subclasses must implement:
            * fit(X: np.ndarray) -> None, which populates self.labels
            * get_model_params() -> Dict[str, Any]
        - Provides a default run(...) that computes CPU metrics
        - Offers optional GPU utilities (prepare_gpu_data, silhouette_gpu)

Dependencies:
    - numpy, pandas, matplotlib, sklearn
    - abc, pathlib, typing, time
    - cupy, cudf (only needed if RAPIDS-based GPU code is used)

Key Features & Best Practices:
    - Minimizes code duplication by centralizing GPU logic in silhouette_gpu & prepare_gpu_data.
    - Adopts float32 on GPU for better memory and speed efficiency.
    - Mentions performance tips from CuPy docs: e.g. use CUPY_ACCELERATORS='cub' for faster reductions.
    - Mentions subtle differences between CuPy & NumPy (casting rules, zero-dimensional arrays, etc.).

References:
    - https://rapids.ai/
    - https://developer.nvidia.com/cuda
    - https://cupy.dev/
    - https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html#gs.iyhgha
    - https://github.com/rapidsai/cuml
    - https://docs.rapids.ai/api/cudf/stable/
    - https://github.com/rapidsai/cudf

Authors:
    - Palakurthi Raja Karthik (kraja@iisc.ac.in)
    - Debajyoti Sahoo (debajyotis@iisc.ac.in, initial version for KMeans)
    - Additional references & contributions from the SciREX development team

Version Info:
    - 30/Dec/2024: GPU-based silhouette & data preparation added
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class Clustering(ABC):
    def __init__(self, model_type: str, random_state: int = 42) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.plots_dir = Path.cwd() / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.labels: Optional[np.ndarray] = None

    def prepare_data(self, path: str) -> np.ndarray:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"The file at path '{path}' does not exist. Please check the file path or permissions."
            )

        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            raise ValueError("No numeric columns found in the CSV file.")
        features = df[numeric_cols].values
        return StandardScaler().fit_transform(features)

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        pass

    def plots(self, X: np.ndarray, labels: np.ndarray) -> Tuple[Figure, Path]:
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X) if X.shape[1] >= 2 else np.hstack([pca.fit_transform(X), np.zeros((X.shape[0], 1))])

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(labels)
        cmap = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for lbl, color in zip(unique_labels, cmap):
            mask = (labels == lbl)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=("k" if lbl == -1 else color), label=f"Cluster {lbl}", alpha=0.7, s=60)

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend(fontsize=8)
        ax.set_title(f"{self.model_type.upper()} (2D PCA)")

        save_path = self.plots_dir / f"cluster_plot_{self.model_type}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig, save_path

    def run(self, data: Optional[np.ndarray] = None, path: Optional[str] = None) -> Dict[str, Any]:
        if data is None and path is None:
            raise ValueError("Either 'data' or 'path' must be provided to the 'run' method.")

        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError("The 'data' parameter must be a NumPy array if provided.")

        if path is not None and not isinstance(path, str):
            raise TypeError("The 'path' parameter must be a string representing the file path.")

        X = data if data is not None else self.prepare_data(path)
        self.fit(X)

        if self.labels is None:
            raise ValueError("fit() did not assign self.labels.")

        s_val = silhouette_score(X, self.labels)
        ch_val = calinski_harabasz_score(X, self.labels)
        db_val = davies_bouldin_score(X, self.labels)

        self.plots(X, self.labels)

        return {
            "params": self.get_model_params(),
            "silhouette_score": s_val,
            "calinski_harabasz_score": ch_val,
            "davies_bouldin_score": db_val,
        }

    def prepare_gpu_data(self, X) -> "cupy.ndarray":
        try:
            import cupy as cp
            import cudf
        except ImportError as e:
            raise ImportError("Requires CuPy & cuDF for GPU data preparation.") from e

        if isinstance(X, np.ndarray):
            try:
                return cp.asarray(X, dtype=cp.float32)
            except MemoryError:
                raise MemoryError("Conversion to CuPy array failed due to insufficient GPU memory. Consider using smaller batches or a more memory-efficient approach.")

        if isinstance(X, cudf.DataFrame):
            numeric_cols = X.select_dtypes(include=["float", "int"]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns found in cuDF DataFrame.")
            try:
                return cp.asarray(X[numeric_cols].values, dtype=cp.float32)
            except MemoryError:
                raise MemoryError("Conversion of cuDF DataFrame to CuPy array failed due to insufficient GPU memory.")

        if "cupy.core.core.ndarray" in str(type(X)) or "cupy._core.core.ndarray" in str(type(X)):
            if X.dtype != cp.float32:
                return X.astype(cp.float32)
            return X

        try:
            return cp.asarray(X, dtype=cp.float32)
        except MemoryError:
            raise MemoryError("Conversion to CuPy array failed due to insufficient GPU memory. Consider optimizing your dataset size.")

    def silhouette_gpu(self, X_gpu: "cupy.ndarray", labels_gpu: "cupy.ndarray") -> float:
        try:
            import cupy as cp
        except ImportError:
            raise ImportError("silhouette_gpu requires CuPy. Please install RAPIDS libraries.")

        if not hasattr(X_gpu, 'shape') or not hasattr(labels_gpu, 'shape'):
            raise TypeError("Invalid input: 'X_gpu' and 'labels_gpu' must be CuPy arrays with valid shapes.")

        if X_gpu.dtype != cp.float32:
            X_gpu = X_gpu.astype(cp.float32)
        if labels_gpu.dtype not in (cp.int32, cp.int64):
            labels_gpu = labels_gpu.astype(cp.int32)

        unique_labels = cp.unique(labels_gpu)
        if unique_labels.size < 2:
            return -1.0

        X_gpu = cp.ascontiguousarray(X_gpu)

        # Optimized blockwise computation for large datasets
        n_samples = X_gpu.shape[0]
        batch_size = min(1024, n_samples)  # Define a suitable block size
        s_vals = cp.zeros(n_samples, dtype=cp.float32)

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            pairwise_dists = cp.linalg.norm(
                X_gpu[i:end, None, :] - X_gpu[None, :, :], axis=2
            )

            for cluster_id in unique_labels:
                mask = (labels_gpu == cluster_id)
                cluster_indices = cp.where(mask)[0]
                if cluster_indices.size == 0:
                    continue

                dist_intra = pairwise_dists[:, cluster_indices]
                a_vals = cp.mean(dist_intra, axis=1)

                b_vals = cp.full(a_vals.shape, cp.inf, dtype=cp.float32)
                for other_cluster_id in unique_labels:
                    if other_cluster_id == cluster_id:
                        continue
                    other_idx = cp.where(labels_gpu == other_cluster_id)[0]
                    if other_idx.size == 0:
                        continue
                    dist_inter = pairwise_dists[:, other_idx]
                    b_vals = cp.minimum(b_vals, cp.mean(dist_inter, axis=1))

                s_cluster = (b_vals - a_vals) / cp.maximum(a_vals, b_vals)
                s_vals[i:end] = s_cluster

        return float(cp.mean(s_vals))
