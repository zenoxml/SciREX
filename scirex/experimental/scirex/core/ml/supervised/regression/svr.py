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
Module: svr.py

This module implements the SVRModel class for regression tasks,
extending the base Regression class from the SciREX library.

The implementation uses scikit-learn's SVR (Support Vector Regression) model for:
    - Training the SVR algorithm
    - Generating predictions for input features
    - Evaluating performance using standard regression metrics

Key Features:
    - Support for fitting an SVR model
    - Access to model parameters (kernel, C, epsilon)
    - Seamless integration with the SciREX regression pipeline

Classes:
    SVRModel: Implements a Support Vector Regression model.

Dependencies:
    - numpy
    - scikit-learn
    - scirex.core.ml.supervised.regression.base

Authors:
    - Paranidharan (paranidharan@iisc.ac.in)

Version Info:
    - 01/Feb/2025: Initial version
"""
# Module: SVR

from scirex.core.ml.supervised.regression.base import Regression
from sklearn.svm import SVR
from typing import Dict, Any
import numpy as np


class SVRModel(Regression):
    """
    Support Vector Regression (SVR) model implementation using scikit-learn.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the SVRModel class.

        Args:
            kernel (str, optional): The kernel type used in the model (default is 'rbf').
            C (float, optional): Regularization parameter (default is 1.0).
            epsilon (float, optional): Epsilon parameter for the margin (default is 0.1).
            random_state (int, optional): Seed for reproducibility (default is 42).
        """
        super().__init__(model_type="svr", random_state=random_state)
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVR model to the data.

        Args:
            X (np.ndarray): The input training features (n_samples, n_features).
            y (np.ndarray): The target training values (n_samples).

        Returns:
            None
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data.

        Args:
            X (np.ndarray): The input data (n_samples, n_features).

        Returns:
            np.ndarray: The predicted target values.
        """
        return self.model.predict(X)

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the SVR model.

        Returns:
            Dict[str, Any]: The parameters of the SVR model.
        """
        return {
            "kernel": self.model.kernel,
            "C": self.model.C,
            "epsilon": self.model.epsilon,
        }
