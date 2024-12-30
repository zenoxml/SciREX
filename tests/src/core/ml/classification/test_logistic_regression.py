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
    Module: test_logistic_regression.py

    This module contains unit tests for the Logistic Regression implementation 
    within the SciREX framework. It tests various configurations of the 
    LogisticRegression class.

    Classes:
        TestLogisticRegression: Defines unit tests for different functionalities 
        and configurations of the LogisticRegression class.

    Dependencies:
        - unittest
        - numpy
        - sklearn.datasets
        - scirex.core.ml.supervised.classification.logistic_regression

    Authors:
        - Protyush P. Chowdhury (protyushc@iisc.ac.in)

    Version Info:
        - 30/Dec/2024: Initial version
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scirex.core.ml.supervised.classification.logistic_regression import LogisticRegressionClassifier

class TestLogisticRegression(unittest.TestCase):
    """
    Unit test class for Logistic Regression implementation.

    This class includes tests for:
      - Model training
      - Prediction accuracy
      - Evaluation metrics
      - Support for multi-class classification
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

    def test_fit_and_predict(self):
        """
        Test the fit and predict functionality of Logistic Regression.
        """
        model = LogisticRegressionClassifier()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.y_test), "Prediction length mismatch.")

    def test_evaluation_metrics(self):
        """
        Test evaluation metrics for the Logistic Regression model.
        """
        model = LogisticRegressionClassifier()
        model.fit(self.X_train, self.y_train)
        metrics = model.evaluate(self.X_test, self.y_test)

        self.assertIn("accuracy", metrics, "Accuracy metric not found.")
        self.assertIn("precision", metrics, "Precision metric not found.")
        self.assertIn("recall", metrics, "Recall metric not found.")
        self.assertIn("f1_score", metrics, "F1 Score metric not found.")

    def test_multi_class_support(self):
        """
        Verify that the Logistic Regression model supports multi-class classification.
        """
        model = LogisticRegressionClassifier()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        unique_classes = np.unique(predictions)
        expected_classes = np.unique(self.y)

        self.assertTrue(
            set(unique_classes).issubset(set(expected_classes)),
            "Predicted classes do not match expected classes."
        )


if __name__ == "__main__":
    unittest.main()
