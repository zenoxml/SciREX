# Standard library imports
import os
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np

from scirex.core.ml.unsupervised.clustering.hdbscan import Hdbscan

def test_hdbscan():
    min_samples = 5
    eps_val = 0.5

    # load and scale the data
    data = np.loadtxt("../../../../../support_files/chainlink_data.txt") 
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # perform HDBSCAN clustering
    hdbscan = Hdbscan()
    hdbscan.fit(data)

    # get number of clusters
    n_clusters = hdbscan.n_clusters
    
    # assert that number of clusters is 2
    assert (n_clusters == 2)


    
    # visualize the clustering data
    labels = hdbscan.labels

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize = (8, 6))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 0]

        class_member_mask = labels == k

        xy = data[class_member_mask]

        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor = tuple(col), markeredgecolor = "k", markersize = 12, label = k)

    plt.title("HDBSCAN clustering", fontsize = 18)
    plt.xlabel("x", fontsize = 18)
    plt.ylabel("y", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(loc = "upper right", fontsize = 14)
    plt.show()


if __name__ == "__main__":
    test_hdbscan()

