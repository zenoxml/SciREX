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
# Added test to check OPTICS clustering algorithm on benchmark dataset. 
# The dataset is taken from "Thrun, Ultsch, 2020, Clustering benchmark
# datasets exploiting the fundamental clustering problems, Data in Brief". 

import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from scirex.core.ml.unsupervised.clustering.optics import Optics
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

def test_kmeans():
    # Load and scale the data
    # data = np.loadtxt("tests/support_files/chainlink_data.txt")
    data = np.loadtxt("../../../../../support_files/engytime.txt") 
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Perform KMeans clustering
    optics = Optics()
    optics.fit(data)
    #hdbscan = Hdbscan()
    #hdbscan.fit(data)

    # calculate silhouette score
    labels = optics.labels



    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize = (8, 6))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 0]

        class_member_mask = labels == k

        xy = data[class_member_mask]

        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor = tuple(col), markeredgecolor = "k", markersize = 12, label = k)

    plt.title("GMM clustering", fontsize = 18)
    plt.xlabel("x", fontsize = 18)
    plt.ylabel("y", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(loc = "upper right", fontsize = 14)
    plt.show()
    

    silhouette_score_val = silhouette_score(data, labels)
    print(f"Silhouette score val is {silhouette_score_val}")

if __name__ == "__main__":
    test_kmeans()
