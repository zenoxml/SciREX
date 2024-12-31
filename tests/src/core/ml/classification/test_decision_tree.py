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
    Module: test_decision_tree.py

    This module contains unit tests for the Decision Tree implementation
    within the SciREX framework. It tests various functionalities and 
    configurations of the DecisionTreeClassifier class.

    Classes:
        TestDecisionTree: Defines unit tests for different functionalities
        and configurations of the DecisionTree class.

    Dependencies:
        - unittest
        - numpy
        - sklearn.datasets
        - scirex.core.ml.supervised.classification.decision_tree

    Authors:
        - Protyush P. Chowdhury (protyushc@iisc.ac.in)

    Version Info:
        - 31/Dec/2024: Initial version
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scirex.core.ml.supervised.classification.decision_tree import DecisionTreeClassifier


class TestDecisionTree(unittest.TestCase):
    """
    Unit test class for Decision Tree implementation.

    This class includes tests for:
      - Model training and parameter optimization
      - Prediction accuracy
      - Evaluation metrics
      - Feature importance computation
      - Model parameter validation
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
        Test the model's fitting and prediction capabilities.
        """
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(
            len(predictions), len(self.y_test), 
            "Prediction length mismatch."
        )

    def test_evaluation_metrics(self):
        """
        Test the evaluation metrics computation.
        """
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        metrics = model.evaluate(self.X_test, self.y_test)

        required_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"{metric} not found in evaluation metrics")
            self.assertGreaterEqual(metrics[metric], 0.0, f"{metric} should be >= 0")
            self.assertLessEqual(metrics[metric], 1.0, f"{metric} should be <= 1")

    def test_feature_importance(self):
        """
        Test feature importance computation functionality.
        """
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        importance = model.get_feature_importance()

        self.assertEqual(
            len(importance), self.X.shape[1],
            "Feature importance length should match number of features"
        )
        
        # Check if importance scores sum to approximately 1
        total_importance = sum(importance.values())
        self.assertAlmostEqual(
            total_importance, 1.0, places=5,
            msg="Feature importance scores should sum to 1"
        )

    def test_model_params(self):
        """
        Test the retrieval and validity of model parameters.
        """
        model = DecisionTreeClassifier(cv=5)
        model.fit(self.X_train, self.y_train)
        params = model.get_model_params()

        required_params = ["model_type", "best_params", "cv"]
        for param in required_params:
            self.assertIn(param, params, f"Missing parameter: {param}")

        self.assertEqual(params["model_type"], "decision_tree")
        self.assertEqual(params["cv"], 5)
        self.assertIsNotNone(params["best_params"])

    def test_prediction_probability(self):
        """
        Test probability estimation for predictions.
        """
        model = DecisionTreeClassifier()
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


if __name__ == "__main__":
    unittest.main()
