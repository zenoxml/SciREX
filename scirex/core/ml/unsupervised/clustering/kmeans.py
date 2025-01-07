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

- Kmeans (Plain CPU):       scikit-learn's MiniBatchKMeans
- KmeansIntel (Intel CPU):  Intel-patched scikit-learn (sklearnex)
- KmeansGPU (RAPIDS GPU):   cuML KMeans, optionally using silhouette_gpu

Each class inherits from the shared Clustering base, giving:
    - Data loading/preprocessing
    - GPU data preparation
    - CPU-based silhouette defaults, or GPU silhouette if desired
    - PCA-based plotting routine

References:
    - https://scikit-learn.org/
    - https://github.com/intel/scikit-learn-intelex
    - https://docs.rapids.ai/api/cuml/stable/api.html#cuml.KMeans
    - https://rapids.ai/

Authors:
    - Palakurthi Raja Karthik (kraja@iisc.ac.in)
    - Debajyoti Sahoo (debajyotis@iisc.ac.in, initial version for KMeans)
    - Additional references & contributions from the SciREX development team

Version Info:
    - 30/Dec/2024: CPU & GPU versions, supporting SciREX base structure
"""

from typing import Optional, Dict, Any
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans as SkKMeans
from packaging import version
from .base import Clustering

class Kmeans(Clustering):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> None:
        super().__init__("kmeans", random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        X = X.astype(np.float32, copy=False)
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            batch_size=1024
        )
        labels = self.model.fit_predict(X)
        self.labels = labels

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
        }

class KmeansIntel(Clustering):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        random_state: int = 42,
        init: str = "k-means++",
    ) -> None:
        try:
            import sklearnex
            if not hasattr(sklearnex, "patch_sklearn"):
                raise ImportError(
                    "Found 'sklearnex' but 'patch_sklearn' is missing. Check your sklearnex install."
                )
            from sklearnex import patch_sklearn
            patch_sklearn()
        except ImportError as e:
            raise ImportError(
                "KmeansIntel requires 'scikit-learn-intelex' to be installed."
            ) from e

        super().__init__("kmeans_intel", random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        X = X.astype(np.float32, copy=False)
        self.model = SkKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init=self.init,
            n_init=10
        )
        labels = self.model.fit_predict(X)
        self.labels = labels

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "init": self.init,
        }

class KmeansGPU(Clustering):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        random_state: int = 42,
        init: str = "k-means||",
    ) -> None:
        super().__init__("kmeans_gpu", random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        try:
            from cuml.cluster import KMeans as CuKMeans
            import cuml

            if hasattr(cuml, "__cuda_version__"):
                cuda_version = cuml.__cuda_version__
                if version.parse(cuda_version) < version.parse("11.0"):
                    raise ImportError(
                        f"Detected CUDA version {cuda_version}, "
                        "which is less than 11.0. Upgrade needed for RAPIDS."
                    )
        except ImportError as exc:
            raise ImportError(
                "KmeansGPU requires rapids cuML (cuml, cupy, cudf)."
            ) from exc

        X_cupy = self.prepare_gpu_data(X)
        self.model = CuKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init=self.init
        )
        labels_cupy = self.model.fit_predict(X_cupy)
        self.labels = labels_cupy.get()

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "init": self.init,
        }
