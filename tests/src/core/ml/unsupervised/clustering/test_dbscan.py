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
# Added test to check DBSCAN clustering algorithm on benchmark dataset.
# The dataset is taken from "Thrun, Ultsch, 2020, Clustering benchmark
# datasets exploiting the fundamental clustering problems, Data in Brief".

import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from scirex.core.ml.unsupervised.clustering.dbscan import Dbscan
from sklearn.metrics import silhouette_score


def test_dbscan():
    # Load and scale the data
    data = np.loadtxt("tests/support_files/chainlink.txt")

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Perform DBSCAN clustering
    dbscan = Dbscan()
    dbscan.fit(data)

    # Calculate silhouette score
    labels = dbscan.labels
    silhouette_score_val = silhouette_score(data, labels, random_state=42)

    assert (
        abs(silhouette_score_val - 0.3301836619003867) < 1.0e-12
    )  # For 24 clusters found using eps = 0.1617 and min samples = 10


if __name__ == "__main__":
    test_dbscan()
