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
    Module: kmeans.py

    This module provides multiple K-Means clustering implementations in SciREX:

    - Kmeans (scikit-learn CPU)
    - KmeansIntel (Intel-patched scikit-learn)
    - KmeansGPU (RAPIDS cuML on GPU, including a custom GPU-based silhouette)

    Classes:
    Kmeans:      Plain CPU-based K-Means using scikit-learn's MiniBatchKMeans.
    KmeansIntel: Intel-patched CPU-based K-Means (requires sklearnex).
    KmeansGPU:   GPU-based K-Means (cuDF + cuML), including a custom GPU silhouette.

    All range-scanning logic ([2..max_k], silhouette, elbow, user prompt) is managed
    by the base Clustering class in `base.py`. Each subclass here implements:
        - fit_at_k(X, k): Training at a given cluster count k
        - compute_inertia_at_k(X, k): Inertia for elbow scanning
        - (Optional) silhouette_gpu(X, k): GPU-based silhouette for KmeansGPU

    Dependencies:
        - scirex.core.ml.unsupervised.clustering.base (Clustering base)
        - scikit-learn or scikit-learn-intelex for CPU/Intel
        - RAPIDS libraries (cudf, cupy, cuml) for GPU

    References:
    [1] https://scirex.org
    [2] https://scikit-learn.org/
    [3] https://rapids.ai/

    Authors:
        - PALAKURTHI RAJA KARTHIK (kraja@iisc.ac.in)
        - Debajyoti Sahoo (debajyotis@iisc.ac.in)
    
    Version Info:
        - 30/Dec/2024: Hardware Optimised version
"""

# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans as SklKMeans

# Local imports
from .base import Clustering


class Kmeans(Clustering):
    """Plain CPU-based K-Means using scikit-learn's MiniBatchKMeans.

    Attributes:
        max_k (int): Maximum cluster count to consider in scanning.
        random_state (int): Seed for reproducibility.
    """

    def __init__(self, max_k: int = 10, random_state: int = 42) -> None:
        """Initialize the CPU-based Kmeans.

        Args:
            max_k: Upper bound for cluster scanning in [2..max_k].
            random_state: Random seed for reproducibility.
        """
        super().__init__("kmeans", random_state)
        self.max_k = max_k

    def fit_at_k(self, X: np.ndarray, k: int) -> np.ndarray:
        """Train a MiniBatchKMeans model at k clusters, returning labels.

        Args:
            X: NumPy array, shape (n_samples, n_features).
            k: Desired number of clusters.

        Returns:
            np.ndarray: Array of integer labels, shape (n_samples,).
        """
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.random_state,
            batch_size=1000
        )
        labels = model.fit_predict(X)
        return labels

    def compute_inertia_at_k(self, X: np.ndarray, k: int) -> float:
        """Compute inertia for elbow-based scanning at cluster count k.

        Args:
            X: NumPy array, shape (n_samples, n_features).
            k: Desired number of clusters.

        Returns:
            float: The inertia from the trained model.
        """
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.random_state,
            batch_size=1000
        )
        model.fit(X)
        return model.inertia_

    def fit(self, X: np.ndarray) -> None:
        """Runs the scanning logic from base, sets final labels.

        Args:
            X: NumPy array of shape (n_samples, n_features).
        """
        chosen_labels, chosen_k = self._range_scan_fit(X, self.max_k)
        self.labels = chosen_labels
        self.n_clusters = len(np.unique(chosen_labels))


class KmeansIntel(Clustering):
    """Intel-patched CPU-based K-Means (requires scikit-learn-intelex).

    Attributes:
        max_k (int): Maximum cluster count for scanning.
        random_state (int): Seed for reproducibility.
    """

    def __init__(self, max_k: int = 10, random_state: int = 42) -> None:
        """Initialize Intel-based CPU KMeans, patches sklearn if installed.

        Args:
            max_k: Upper bound for scanning in [2..max_k].
            random_state: Seed for reproducibility.

        Raises:
            ImportError: If scikit-learn-intelex is not installed.
        """
        try:
            from sklearnex import patch_sklearn
            patch_sklearn()
        except ImportError as e:
            raise ImportError(
                "KmeansIntel requires 'scikit-learn-intelex'. Please install it."
            ) from e

        super().__init__("kmeans_intel", random_state)
        self.max_k = max_k

    def fit_at_k(self, X: np.ndarray, k: int) -> np.ndarray:
        """Train Intel-patched scikit-learn KMeans at k clusters.

        Args:
            X: NumPy array, shape (n_samples, n_features).
            k: Desired number of clusters.

        Returns:
            np.ndarray: Labels array, shape (n_samples,).
        """
        model = SklKMeans(n_clusters=k, random_state=self.random_state)
        labels = model.fit_predict(X)
        return labels

    def compute_inertia_at_k(self, X: np.ndarray, k: int) -> float:
        """Compute inertia for elbow scanning at cluster count k.

        Args:
            X: NumPy array, shape (n_samples, n_features).
            k: Desired number of clusters.

        Returns:
            float: Inertia from the fitted model.
        """
        model = SklKMeans(n_clusters=k, random_state=self.random_state)
        model.fit(X)
        return model.inertia_

    def fit(self, X: np.ndarray) -> None:
        """Uses scanning logic from base, sets final labels.

        Args:
            X: NumPy array, shape (n_samples, n_features).
        """
        chosen_labels, chosen_k = self._range_scan_fit(X, self.max_k)
        self.labels = chosen_labels
        self.n_clusters = len(np.unique(chosen_labels))


class KmeansGPU(Clustering):
    """GPU-based K-Means with cuDF + cuML, featuring a GPU silhouette option.

    Attributes:
        max_k (int): Max clusters for scanning.
        random_state (int): Seed for reproducibility.
    """

    def __init__(self, max_k: int = 10, random_state: int = 42) -> None:
        """Initialize GPU-based KMeans if RAPIDS libraries are available.

        Args:
            max_k: Upper bound for cluster scanning [2..max_k].
            random_state: Seed for reproducibility.

        Raises:
            ImportError: If cudf, cupy, or cuml are missing.
        """
        try:
            import cudf
            import cupy
            import cuml
        except ImportError as e:
            raise ImportError(
                "KmeansGPU requires cudf, cupy, and cuml (RAPIDS). Not found."
            ) from e

        super().__init__("kmeans_gpu", random_state)
        self.max_k = max_k

    def prepare_gpu_data(self, X: np.ndarray) -> "cupy.ndarray":
        """Converts input data to a CuPy array on GPU if needed.

        Args:
            X: Can be NumPy ndarray, cuDF DataFrame, or CuPy ndarray.

        Returns:
            cupy.ndarray: Data on GPU.

        Raises:
            ValueError: If cuDF DataFrame has no numeric columns.
        """
        import numpy as np
        import cupy as cp
        import cudf

        if isinstance(X, np.ndarray):
            return cp.asarray(X)
        if isinstance(X, cudf.DataFrame):
            numeric_cols = X.select_dtypes(include=["float", "int"]).columns
            if not numeric_cols.size:
                raise ValueError("No numeric columns found in cuDF DataFrame.")
            return X[numeric_cols].values  # returns a CuPy array
        if isinstance(X, cp.ndarray):
            return X
        return cp.asarray(X)

    def fit_at_k(self, X: np.ndarray, k: int) -> np.ndarray:
        """Train cuML KMeans at k clusters on GPU, returning CPU labels.

        Args:
            X: Data array (NumPy, cuDF, or CuPy).
            k: Number of clusters.

        Returns:
            np.ndarray: Labels moved back from GPU to CPU.
        """
        import cupy as cp
        from cuml.cluster import KMeans as CuKMeans

        X_cupy = self.prepare_gpu_data(X)
        model = CuKMeans(n_clusters=k, random_state=self.random_state)
        labels_cupy = model.fit_predict(X_cupy)
        return labels_cupy.get()

    def compute_inertia_at_k(self, X: np.ndarray, k: int) -> float:
        """Compute inertia using cuML for elbow-based scanning.

        Args:
            X: Data array in CPU or GPU format.
            k: Number of clusters.

        Returns:
            float: The inertia from the GPU-based model.
        """
        from cuml.cluster import KMeans as CuKMeans

        X_cupy = self.prepare_gpu_data(X)
        model = CuKMeans(n_clusters=k, random_state=self.random_state)
        model.fit(X_cupy)
        return float(model.inertia_)

    def silhouette_gpu(self, X: np.ndarray, k: int) -> float:
        """Computes silhouette fully on GPU (O(n^2) approach).

        The base can call this if you override base scanning to do GPU-based silhouette.
        Otherwise, base might do CPU silhouette. This method is available for advanced usage.

        Args:
            X: Data array (CPU or GPU).
            k: Number of clusters.

        Returns:
            float: Mean silhouette score, or -1.0 if only one cluster.
        """
        import cupy as cp
        from cuml.cluster import KMeans as CuKMeans

        X_cupy = self.prepare_gpu_data(X)
        model = CuKMeans(n_clusters=k, random_state=self.random_state)
        labels_cupy = model.fit_predict(X_cupy)

        unique_labels = cp.unique(labels_cupy)
        if len(unique_labels) < 2:
            return -1.0

        # Distances between all pairs of points (O(n^2))
        dist_matrix = cp.sqrt(
            cp.sum((X_cupy[:, None, :] - X_cupy[None, :, :]) ** 2, axis=2)
        )

        n_samples = X_cupy.shape[0]
        n_clusters = len(unique_labels)
        cluster_means = cp.zeros((n_samples, n_clusters), dtype=cp.float32)

        # cluster_means[i, c] = avg distance from i to cluster c
        for idx_c, cid in enumerate(unique_labels):
            mask_c = (labels_cupy == cid)
            idxs_c = cp.where(mask_c)[0]
            dist_to_c = dist_matrix[:, idxs_c]
            cluster_means[:, idx_c] = cp.mean(dist_to_c, axis=1)

        a_vals = cp.zeros(n_samples, dtype=cp.float32)
        b_vals = cp.zeros(n_samples, dtype=cp.float32)

        for idx_c, cid in enumerate(unique_labels):
            mask_c = (labels_cupy == cid)
            a_vals[mask_c] = cluster_means[mask_c, idx_c]
            sub_means = cluster_means[mask_c]
            left_part = sub_means[:, :idx_c]
            right_part = sub_means[:, idx_c+1:]
            combined = cp.concatenate([left_part, right_part], axis=1)
            b_vals[mask_c] = cp.min(combined, axis=1)

        s_vals = (b_vals - a_vals) / cp.maximum(a_vals, b_vals)
        silhouette_mean = cp.mean(s_vals)
        return float(silhouette_mean)

    def fit(self, X: np.ndarray) -> None:
        """Runs scanning from base, sets final cluster labels.

        Args:
            X: Data array (CPU or GPU).
        """
        chosen_labels, chosen_k = self._range_scan_fit(X, self.max_k)
        self.labels = chosen_labels
        self.n_clusters = len(np.unique(chosen_labels))