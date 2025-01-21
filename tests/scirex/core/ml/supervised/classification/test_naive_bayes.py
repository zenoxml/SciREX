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
    Module: test_naive_bayes.py

    This module provides unit tests for the NaiveBayes classifier implementations
    with different class conditional densities (Gaussian, Multinomial, Bernoulli).

    Classes:
        TestNaiveBayes: Contains unit tests for Gaussian, Multinomial, and Bernoulli
                        Naive Bayes classification algorithms.

    Dependencies:
        - unittest
        - numpy
        - sklearn.datasets (for Iris dataset)
        - scirex.core.ml.unsupervised.clustering.kmeans.NaiveBayes

    Authors:
        - Protyush P. Chowdhury (protyushc@iisc.ac.in) 

    Version Info:
        - 28/Dec/2024: Initial version
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scirex.core.ml.supervised.classification.naive_bayes import NaiveBayes


class TestNaiveBayes(unittest.TestCase):
    """
    Unit tests for NaiveBayes classifier with Gaussian, Multinomial, and Bernoulli class conditional densities.
    """

    def setUp(self):
        """
        Prepare the Iris dataset for testing.
        """
        self.data = load_iris()
        self.X = self.data.data
        self.y = self.data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_gaussian_naive_bayes(self):
        """
        Test the NaiveBayes classifier with Gaussian class conditional densities.
        """
        model = NaiveBayes(model_type="gaussian")
        model.run(self.X, self.y, test_size=0.2)
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertGreaterEqual(
            metrics["accuracy"],
            0.4,
            "Accuracy should be >= 40% for Gaussian Naive Bayes.",
        )

    def test_multinomial_naive_bayes(self):
        """
        Test the NaiveBayes classifier with Multinomial class conditional densities.
        """
        model = NaiveBayes(model_type="multinomial")
        model.run(self.X, self.y, test_size=0.2)
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertGreaterEqual(
            metrics["accuracy"],
            0.5,
            "Accuracy should be >= 50% for Multinomial Naive Bayes.",
        )

    def test_bernoulli_naive_bayes(self):
        """
        Test the NaiveBayes classifier with Bernoulli class conditional densities.
        """
        # Simulate binary data for Bernoulli Naive Bayes
        X_binary = np.where(self.X > np.median(self.X, axis=0), 1, 0)
        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
            X_binary, self.y, test_size=0.2, random_state=42
        )

        model = NaiveBayes(model_type="bernoulli")
        model.run(self.X, self.y, test_size=0.2)
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertGreaterEqual(
            metrics["accuracy"],
            0.3,
            "Accuracy should be >= 30% for Bernoulli Naive Bayes.",
        )


if __name__ == "__main__":
    unittest.main()
