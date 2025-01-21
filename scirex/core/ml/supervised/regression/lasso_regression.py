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
    Module: lasso_regression.py

    This module implements the LassoRegressionModel class for Lasso regression tasks,
    extending the base Regression class from the SciREX library.

    The implementation uses scikit-learn's Lasso regression model for:
        - Fitting the Lasso regression algorithm
        - Generating predictions for input features
        - Evaluating performance using standard regression metrics

    Classes:
        LassoRegressionModel: Implements a Lasso Regression model.

    Dependencies:
        - numpy
        - scikit-learn
        - scirex.core.ml.supervised.regression.base

    Authors:
        - Paranidharan (paranidharan@iisc.ac.in)

    Version Info:
        - 16/Jan/2025: Initial version
"""

from scirex.core.ml.supervised.regression.base import Regression
from sklearn.linear_model import Lasso
from typing import Dict, Any
import numpy as np


class LassoRegressionModel(Regression):
    """
    Class: LassoRegressionModel

    This class implements a Lasso Regression model for regression tasks.
    The model is based on scikit-learn's Lasso regression implementation.

    The class provides methods for:
        - Training the Lasso regression model
        - Generating predictions for input features
        - Evaluating performance using standard regression metrics

    Attributes:
        - model: Lasso regression model
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        """
        Initialize the LassoRegressionModel class.

        Args:
            alpha (float, optional): Regularization strength. Defaults to 1.0.
            random_state (int, optional): Seed for reproducibility. Defaults to 42.
        """
        super().__init__(model_type="lasso_regression", random_state=random_state)
        self.model = Lasso(alpha=alpha, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Lasso regression model to the data.

        Args:
            X (np.ndarray): Input features for training (n_samples, n_features).
            y (np.ndarray): Target values for training (n_samples).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained Lasso regression model.

        Args:
            X (np.ndarray): Input features for prediction (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values (n_samples,).
        """
        return self.model.predict(X)

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the model parameters coefficients and intercept.

        Returns:
            Dict[str, Any]: Model parameters, including coefficients and intercept.
        """
        return {
            "coefficients": self.model.coef_,
            "intercept": self.model.intercept_,
        }
