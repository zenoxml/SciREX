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
    Module: example_naive_bayes.py

    This module demonstrates the usage of the NaiveBayes class
    for Gaussian class-conditional densities using the Iris dataset.

    Dependencies:
        - scikit-learn
        - pandas
        - numpy
        - scirex.core.ml.unsupervised.clustering.kmeans.NaiveBayes

    Example Usage:
        Run this script to load the Iris dataset, fit the NaiveBayes
        classifier, and evaluate its performance metrics.

    Authors:
        - Protyush P. Chowdhury (protyushc@iisc.ac.in)

    Version Info:
        - 30/Dec/2024: Initial version

"""

# Standard library imports
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# SciREX NaiveBayes import
from scirex.core.ml.supervised.classification.naive_bayes import NaiveBayes

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and run the NaiveBayes classifier (Gaussian assumption)
nb_classifier = NaiveBayes(model_type='gaussian')

# Fit the model to the training data
nb_classifier.fit(X_train, y_train)

# Evaluate the model on the test data
performance_metrics = nb_classifier.run(X_test, y_test)

# Print the performance metrics
print("Performance Metrics:")
for metric, value in performance_metrics.items():
    print(f"{metric}: {value}")

# Generate and display the confusion matrix
print("\nConfusion Matrix:")
nb_classifier.plot(X_test, y_test)
