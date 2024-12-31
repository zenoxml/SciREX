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
    Module: test_svm.py

    This module contains unit tests for the Support Vector Machine (SVM) 
    implementation within the SciREX framework. It tests various kernel
    configurations and functionalities of the SVMClassifier class.

    Classes:
        TestSVM: Defines unit tests for different kernels and functionalities
        of the SVM classifier.

    Dependencies:
        - unittest
        - numpy
        - sklearn.datasets
        - scirex.core.ml.supervised.classification.svm

    Authors:
            - Harshwardhan S. Fartale (harshwardha1c@iisc.ac.in)


    Version Info:
        - 31/Dec/2024: Initial version
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scirex.core.ml.supervised.classification.svm import SVMClassifier


class TestSVM(unittest.TestCase):
    """
    Unit test class for SVM implementation.

    This class includes tests for:
      - Different kernel configurations
      - Model training and parameter optimization
      - Prediction accuracy
      - Evaluation metrics
      - Probability estimation
    """

    @classmethod
    def setUpClass(cls):
        """
        Load the Iris dataset and prepare train-test splits for testing.
        """
        iris = load_iris()
        cls.X = iris.data
        cls.y = iris.target

        # Standardize the dataset
        scaler = StandardScaler()
        cls.X = scaler.fit_transform(cls.X)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_kernel_configurations(self):
        """
        Test different kernel configurations.
        """
        kernels = ["linear", "rbf", "poly", "sigmoid"]
        for kernel in kernels:
            with self.subTest(kernel=kernel):
                model = SVMClassifier(kernel=kernel)
                model.fit(self.X_train, self.y_train)
                metrics = model.evaluate(self.X_test, self.y_test)
                
                self.assertIn("accuracy", metrics)
                self.assertGreaterEqual(
                    metrics["accuracy"], 0.4,
                    f"Accuracy for {kernel} kernel should be >= 40%"
                )

    def test_parameter_optimization(self):
        """
        Test parameter optimization through grid search.
        """
        model = SVMClassifier(kernel="rbf", cv=3)
        model.fit(self.X_train, self.y_train)
        
        params = model.get_model_params()
        self.assertIsNotNone(params["best_params"])
        self.assertIn("C", params["best_params"])
        
        if model.kernel != "linear":
            self.assertIn("gamma", params["best_params"])

    def test_evaluation_metrics(self):
        """
        Test evaluation metrics computation.
        """
        model = SVMClassifier()
        model.fit(self.X_train, self.y_train)
        metrics = model.evaluate(self.X_test, self.y_test)

        required_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"{metric} not found in evaluation metrics")
            self.assertGreaterEqual(metrics[metric], 0.0, f"{metric} should be >= 0")
            self.assertLessEqual(metrics[metric], 1.0, f"{metric} should be <= 1")

    def test_probability_estimation(self):
        """
        Test probability estimation functionality.
        """
        model = SVMClassifier()
        model.fit(self.X_train, self.y_train)
        probabilities = model.predict_proba(self.X_test)

        self.assertEqual(
            probabilities.shape, (len(self.y_test), len(np.unique(self.y))),
            "Probability matrix shape mismatch"
        )
        
        # Check if probabilities sum to 1 for each prediction
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_array_almost_equal(
            prob_sums, np.ones_like(prob_sums),
            decimal=5,
            err_msg="Probabilities should sum to 1 for each prediction"
        )

    def test_model_params(self):
        """
        Test model parameter retrieval and validation.
        """
        model = SVMClassifier(kernel="rbf", cv=5)
        model.fit(self.X_train, self.y_train)
        params = model.get_model_params()

        required_params = ["model_type", "kernel", "best_params", "cv"]
        for param in required_params:
            self.assertIn(param, params, f"Missing parameter: {param}")

        self.assertEqual(params["model_type"], "svm")
        self.assertEqual(params["kernel"], "rbf")
        self.assertEqual(params["cv"], 5)

    def test_input_validation(self):
        """
        Test input validation for invalid kernel specification.
        """
        with self.assertRaises(ValueError):
            SVMClassifier(kernel="invalid_kernel")


if __name__ == "__main__":
    unittest.main()
