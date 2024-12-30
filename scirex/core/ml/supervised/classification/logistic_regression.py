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
    Module: logistic_regression.py

    This module implements Logistic Regression for binary and multi-class classification tasks.

    It provides functionality to:
      - Train Logistic Regression classifiers on a dataset
      - Evaluate model performance using classification metrics
      - Visualize results with a confusion matrix
      - Optimize hyperparameters using grid search

    Classes:
        LogisticRegressionClassifier: Implements Logistic Regression using scikit-learn.

    Dependencies:
        - numpy
        - sklearn.linear_model.LogisticRegression
        - sklearn.metrics (classification metrics)
        - sklearn.model_selection.GridSearchCV
        - matplotlib, seaborn
        - base.py (Classification)

    Key Features:
        - Support for binary and multi-class classification
        - Regularization options (L1, L2, ElasticNet)
        - Grid search for hyperparameter tuning
        - Automatic data preparation and evaluation

    Authors:
        - Protyush P. Chowdhury (protyushc@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

"""

# Required Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any

from .base import Classification


class LogisticRegressionClassifier(Classification):
    """
    Implements Logistic Regression for classification tasks.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the Logistic Regression classifier.

        Args:
            random_state (int): Seed for reproducibility.
        """
        super().__init__(model_type="logistic_regression", random_state=random_state)
        self.model = LogisticRegression(random_state=random_state, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Logistic Regression model.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test data features.
            y_test (np.ndarray): Test data labels.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics (accuracy, precision, recall, F1-score).
        """
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
        }

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the test data.

        Args:
            X_test (np.ndarray): Test data features.

        Returns:
            np.ndarray: Array of predicted labels.
        """
        # Use the model to predict the labels for the given test data
        return self.model.predict(X_test)

    def plot(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plot the confusion matrix for the test data.

        Args:
            X_test (np.ndarray): Test data features.
            y_test (np.ndarray): Test data labels.
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test),
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Logistic Regression")
        plt.show()

    def grid_search(
        self, X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict[str, Any]
    ) -> None:
        """
        Perform hyperparameter tuning using grid search.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
            param_grid (Dict[str, Any]): Dictionary of hyperparameters to search.
        """
        grid = GridSearchCV(
            estimator=self.model, param_grid=param_grid, scoring="accuracy", cv=5
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best Cross-Validated Accuracy: {grid.best_score_}")

    def run(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        split_ratio: float = 0.2,
        param_grid: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full classification pipeline with optional grid search.

        Args:
            data (np.ndarray): Input features.
            labels (np.ndarray): Input labels.
            split_ratio (float): Proportion of data to use for testing. Defaults to 0.2.
            param_grid (Dict[str, Any], optional): Dictionary of hyperparameters to search for grid search. If None, grid search will not be performed.

        Returns:
            Dict[str, Any]: Performance metrics.
        """
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=split_ratio
        )

        # Perform grid search if param_grid is provided
        if param_grid:
            self.grid_search(X_train, y_train, param_grid)

        # Train the model with the (possibly tuned) parameters
        self.fit(X_train, y_train)

        # Evaluate the model
        metrics = self.evaluate(X_test, y_test)

        # Plot the confusion matrix
        self.plot(X_test, y_test)

        return metrics

    # Implementing the abstract method get_model_params
    def get_model_params(self) -> Dict[str, Any]:
        """
        Return the parameters of the model.

        Returns:
            Dict[str, Any]: Dictionary containing model parameters.
        """
        return {
            "C": self.model.C,
            "max_iter": self.model.max_iter,
            "solver": self.model.solver,
            "penalty": self.model.penalty,
        }
