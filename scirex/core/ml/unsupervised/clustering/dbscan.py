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
Module: dbscan.py

This module provides multiple DBSCAN clustering implementations in SciREX:

- Dbscan (Plain CPU-based DBSCAN using scikit-learn)
- DbscanIntel (Intel-patched scikit-learn DBSCAN, requires sklearnex)
- DbscanGPU (RAPIDS cuML DBSCAN on GPU, optionally using a GPU-based
             silhouette method from base.py)

Each class inherits from the shared Clustering base for consistent:
    - Data loading/preprocessing
    - PCA-based 2D plotting
    - CPU-based silhouette defaults
    - Optional GPU-based silhouette in silhouette_gpu(...)

Dependencies:
    - scirex.core.ml.unsupervised.clustering.base (Clustering base)
    - scikit-learn or scikit-learn-intelex for CPU/Intel
    - rapids (cudf, cupy, cuml) for GPU-based DBSCAN if desired

Authors:
    - Palakurthi Raja Karthik (kraja@iisc.ac.in)
    - Debajyoti Sahoo (debajyotis@iisc.ac.in)

Version Info:
    - 30/Dec/2024: Hardware Optimized version with both CPU & GPU approaches
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors
from .base import Clustering

class Dbscan(Clustering):
    def __init__(self, eps: Optional[float] = None, min_samples: Optional[int] = None, random_state: int = 42) -> None:
        super().__init__("dbscan", random_state)
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters: Optional[int] = None
        self.n_noise: Optional[int] = None
        self.model: Optional[SkDBSCAN] = None

    def fit(self, X: np.ndarray) -> None:
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]

        if self.min_samples is None:
            import math
            self.min_samples = max(5, int(math.log2(n_samples)) + 1)

        if self.eps is None:
            rng = np.random.default_rng(self.random_state)
            sample_size = min(n_samples, 1000)

            # Efficient sampling method using permutation instead of choice
            indices = rng.permutation(n_samples)[:sample_size]
            X_sample = X[indices]

            if len(X_sample) < self.min_samples:
                raise ValueError(
                    f"Sample size is less than the required min_samples={self.min_samples}. "
                    "Increase the sample size or reduce min_samples."
                )

            nbrs = SkNearestNeighbors(n_neighbors=self.min_samples, algorithm="auto")
            nbrs.fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            k_distances = distances[:, -1]
            self.eps = float(np.median(k_distances))

        self.model = SkDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1
        )
        labels = self.model.fit_predict(X)
        self.labels = labels

        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = int(np.sum(labels == -1))

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
        }

class DbscanIntel(Clustering):
    def __init__(self, eps: Optional[float] = None, min_samples: Optional[int] = None, random_state: int = 42) -> None:
        try:
            import sklearnex
            if not hasattr(sklearnex, "patch_sklearn"):
                raise ImportError("sklearnex module does not have 'patch_sklearn'. Ensure sklearnex is correctly installed.")
            from sklearnex import patch_sklearn
            patch_sklearn()
        except ImportError as e:
            raise ImportError("DbscanIntel requires 'scikit-learn-intelex'.") from e

        super().__init__("dbscan_intel", random_state)
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters: Optional[int] = None
        self.n_noise: Optional[int] = None
        self.model: Optional[SkDBSCAN] = None

    def fit(self, X: np.ndarray) -> None:
        X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]
        import math

        if self.min_samples is None:
            self.min_samples = max(5, int(math.log2(n_samples)) + 1)

        if self.eps is None:
            rng = np.random.default_rng(self.random_state)
            sample_size = min(n_samples, 1000)

            # Efficient sampling method using permutation instead of choice
            indices = rng.permutation(n_samples)[:sample_size]
            X_sample = X[indices]

            if len(X_sample) < self.min_samples:
                raise ValueError(
                    f"Sample size is less than the required min_samples={self.min_samples}. "
                    "Increase the sample size or reduce min_samples."
                )

            nbrs = SkNearestNeighbors(n_neighbors=self.min_samples, algorithm="auto")
            nbrs.fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            k_distances = distances[:, -1]
            self.eps = float(np.median(k_distances))

        from sklearn.cluster import DBSCAN as IntelDBSCAN
        self.model = IntelDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1
        )
        labels = self.model.fit_predict(X)
        self.labels = labels

        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = int(np.sum(labels == -1))

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
        }

class DbscanGPU(Clustering):
    def __init__(self, eps: float = 0.5, min_samples: int = 5, random_state: int = 42) -> None:
        super().__init__("dbscan_gpu", random_state)
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters: Optional[int] = None
        self.n_noise: Optional[int] = None
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        try:
            from cuml.cluster import DBSCAN as CuDBSCAN
            import cuml
            cuda_version = cuml.__cuda_version__
            if cuda_version < "11.0":
                raise ImportError(
                    f"Incompatible CUDA version ({cuda_version}). Upgrade to CUDA 11.0 or newer for RAPIDS cuML compatibility."
                )
        except ImportError as exc:
            raise ImportError("DbscanGPU requires RAPIDS cuML libraries.") from exc

        X_cupy = self.prepare_gpu_data(X)
        self.model = CuDBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels_cupy = self.model.fit_predict(X_cupy)

        labels = labels_cupy.get()
        self.labels = labels

        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = np.count_nonzero(labels == -1)

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
        }
