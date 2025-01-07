# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform).
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
# Author : Naren Vohra
# Added test to check HDBSCAN clustering algorithm on benchmark dataset.
# The dataset is taken from "Thrun, Ultsch, 2020, Clustering benchmark
# datasets exploiting the fundamental clustering problems, Data in Brief".

import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from scirex.core.ml.unsupervised.clustering.hdbscan import Hdbscan
from sklearn.metrics import silhouette_score


def test_hdbscan():
    # Load and scale the data
    data = np.loadtxt("tests/support_files/chainlink.txt")

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Perform HDBSCAN clustering
    hdbscan = Hdbscan()
    hdbscan.fit(data)

    # Get number of clusters
    n_clusters = hdbscan.n_clusters

    # Assert that number of clusters is 2
    assert n_clusters == 2

    # Calculate Silhouette score
    labels = hdbscan.labels
    silhouette_score_val = silhouette_score(data, labels, random_state=42)

    assert abs(silhouette_score_val - 0.15363442914939499) < 1.0e-2  # For 2 clusters


if __name__ == "__main__":
    test_hdbscan()
