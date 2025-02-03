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
Module: polynomial_regression.py

This module implements the PolynomialRegressionModel class for regression tasks,
extending the base Regression class from the SciREX library.

The implementation uses scikit-learn's PolynomialFeatures and LinearRegression model for:
    - Polynomial feature generation
    - Training the polynomial regression algorithm
    - Generating predictions for input features
    - Evaluating performance using standard regression metrics

Key Features:
    - Polynomial feature transformation for non-linear regression
    - Seamless integration with the SciREX regression pipeline
    - Access to model parameters (coefficients, intercept, degree)

Classes:
    PolynomialRegressionModel: Implements Polynomial Regression.

Dependencies:
    - numpy
    - scikit-learn
    - scirex.core.ml.supervised.regression.base

Authors:
    - Paranidharan (paranidharan@iisc.ac.in)

Version Info:
    - 01/Feb/2025: Initial version
"""

from scirex.core.ml.supervised.regression.base import Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Dict, Any
import numpy as np


class PolynomialRegressionModel(Regression):
    """
    Polynomial Regression model implementation using scikit-learn's PolynomialFeatures
    and LinearRegression.

    This model is designed to capture non-linear relationships in data by transforming
    the features into polynomial features and applying linear regression on those transformed features.
    """

    def __init__(self, degree: int = 2, random_state: int = 42) -> None:
        """
        Initialize the PolynomialRegressionModel class.

        Args:
            degree (int, optional): The degree of the polynomial features. Default is 2.
            random_state (int, optional): Seed for reproducibility where applicable. Default is 42.
        """
        super().__init__(model_type="polynomial_regression", random_state=random_state)
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the polynomial regression model to the data.

        This method transforms the input features into polynomial features, then fits a
        linear regression model to the transformed features.

        Args:
            X (np.ndarray): The input training features (n_samples, n_features).
            y (np.ndarray): The target training values (n_samples).

        Returns:
            None
        """
        X_poly = self.poly_features.fit_transform(X)
        self.model.fit(X_poly, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data.

        This method transforms the input features into polynomial features before making predictions.

        Args:
            X (np.ndarray): The input data (n_samples, n_features).

        Returns:
            np.ndarray: The predicted target values.
        """
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the polynomial regression model.

        Returns:
            Dict[str, Any]: The parameters of the polynomial regression model, including:
                             - Coefficients
                             - Intercept
                             - Degree of the polynomial.
        """
        return {
            "coefficients": self.model.coef_,
            "intercept": self.model.intercept_,
            "degree": self.degree,
        }
