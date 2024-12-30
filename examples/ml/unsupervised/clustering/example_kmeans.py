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
Example Script: example_kmeans.py

This script demonstrates how to use the KMeans clustering class from SciREX 
library to perform clustering on synthetic datasets. The example shows parameter
selection, metrics calculation and visualization capabilities.

The example includes:
   - Generating synthetic datasets (blobs and moons) using sklearn
   - Data preprocessing using StandardScaler
   - Running KMeans clustering with automatic parameter selection
   - Evaluating results using multiple metrics
   - Visualizing the clustering results

Dependencies:
   - numpy
   - matplotlib
   - scikit-learn 
   - scirex.core.ml.unsupervised.clustering.kmeans

Authors:
   - Debajyoti Sahoo (debajyotis@iisc.ac.in)

Version Info:
   - 30/Dec/2024: Initial version
"""

# Standard library imports
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Third-party imports
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

# Import SciREX model
from scirex.core.ml.unsupervised.clustering.kmeans import Kmeans


def run(model, X, dataset_name: str):
    """
    Run clustering model and display results.

    Args:
        model: Initialized clustering model instance
        X (np.ndarray): Input data matrix
        dataset_name (str): Name of dataset for display purposes

    Prints:
        - Clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
        - Number of clusters (if applicable)

    Displays:
        - Clustering visualization plot
    """
    # Run model and get results
    results = model.run(data=X)

    # Print metrics
    print(f"\n--- {model.model_type.upper()} on {dataset_name} dataset ---")
    print(f"Silhouette Score:      {results['silhouette_score']:.3f}")
    print(f"Calinski-Harabasz:     {results['calinski_harabasz_score']:.3f}")
    print(f"Davies-Bouldin:        {results['davies_bouldin_score']:.3f}")

    if hasattr(model, "n_clusters"):
        print(f"Number of clusters:    {model.n_clusters}")

    # Generate and display plot
    fig, plot_path = model.plots(X, model.labels)

    if not os.path.exists(plot_path):
        print(f"[WARNING] Plot file not found: {plot_path}")
    else:
        img = mpimg.imread(plot_path)
        plt.figure(figsize=(7, 5))
        plt.imshow(img)
        plt.title(
            f"{model.model_type.upper()} Clustering Plot for {dataset_name}",
            fontsize=14,
        )
        plt.axis("off")
        plt.show()


def main():
    """
    Main function to demonstrate KMeans clustering on synthetic datasets.
    Generates two datasets (blobs and moons) and applies KMeans clustering.
    """
    # Generate blob dataset
    X_blobs, _ = make_blobs(n_samples=1000, centers=4, random_state=42)
    X_blobs = StandardScaler().fit_transform(X_blobs)

    # Generate moons dataset
    X_moons, _ = make_moons(n_samples=1000, noise=0.05, random_state=42)
    X_moons = StandardScaler().fit_transform(X_moons)

    # Initialize KMeans models
    kmeans_blobs = Kmeans(max_k=10)
    kmeans_moons = Kmeans(max_k=10)

    # Run clustering on both datasets
    run(kmeans_blobs, X_blobs, "blobs")
    run(kmeans_moons, X_moons, "moons")


if __name__ == "__main__":
    main()
