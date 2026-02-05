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

"""Support Vector Machine (SVM) classification implementation for SciREX.

This module provides a comprehensive SVM implementation using scikit-learn,
supporting multiple kernel types with automatic parameter tuning.

Mathematical Background:
    SVM solves the optimization problem:
    min_{w,b} 1/2||w||² + C∑max(0, 1 - yᵢ(w·xᵢ + b))
    
    Kernel functions supported:
    1. Linear: K(x,y) = x·y
    2. RBF: K(x,y) = exp(-γ||x-y||²)
    3. Polynomial: K(x,y) = (γx·y + r)^d
    4. Sigmoid: K(x,y) = tanh(γx·y + r)
    
    The dual formulation solves:
    max_α ∑αᵢ - 1/2∑∑αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
    subject to: 0 ≤ αᵢ ≤ C, ∑αᵢyᵢ = 0

Key Features:
    - Multiple kernel functions
    - Automatic parameter optimization
    - Probability estimation support
    - Efficient optimization for large datasets

References:
    [1] Vapnik, V. (1998). Statistical Learning Theory
    [2] Scholkopf, B., & Smola, A. J. (2002). Learning with Kernels
    [3] Platt, J. (1999). Probabilistic Outputs for SVMs
"""

from typing import Dict, Any, Optional, Literal
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from .base import Classification


class SVMClassifier(Classification):
    """SVM classifier with automatic parameter tuning.

    This implementation supports different kernel types and includes
    automatic parameter optimization using grid search with cross-validation.
    Each kernel is optimized for its specific characteristics and use cases.

    Attributes:
        kernel: Type of kernel function
        cv: Number of cross-validation folds
        best_params: Best parameters found by grid search

    Example:
        >>> classifier = SVMClassifier(kernel="rbf", cv=5)
        >>> X_train = np.array([[1, 2], [2, 3], [3, 4]])
        >>> y_train = np.array([0, 0, 1])
        >>> classifier.fit(X_train, y_train)
        >>> print(classifier.best_params)
    """

    def __init__(
        self,
        kernel: Literal["linear", "rbf", "poly", "sigmoid"] = "rbf",
        cv: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize SVM classifier.

        Args:
            kernel: Kernel function type. Options:
                   "linear": Linear kernel for linearly separable data
                   "rbf": Radial basis function for non-linear patterns
                   "poly": Polynomial kernel for non-linear patterns
                   "sigmoid": Sigmoid kernel for neural network-like behavior
            cv: Number of cross-validation folds. Defaults to 5.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__("svm", **kwargs)
        self._validate_kernel(kernel)
        self.kernel = kernel
        self.cv = cv
        self.best_params: Optional[Dict[str, Any]] = None

    def _get_param_grid(self):
        """Get parameter grid for grid search based on kernel type.

        Returns:
            Dictionary of parameters to search for the chosen kernel.

        Notes:
            Parameter grids are optimized for each kernel:
            - Linear: Only C (regularization)
            - RBF: C and gamma (kernel coefficient)
            - Polynomial: C, gamma, degree, and coef0
            - Sigmoid: C, gamma, and coef0
        """
        param_grid = {
            "C": [0.1, 1, 10, 100],
        }

        if self.kernel == "linear":
            return param_grid

        elif self.kernel == "rbf":
            param_grid.update(
                {
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                }
            )

        elif self.kernel == "poly":
            param_grid.update(
                {
                    "degree": [2, 3, 4],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                    "coef0": [0, 1],
                }
            )

        elif self.kernel == "sigmoid":
            param_grid.update(
                {
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                    "coef0": [0, 1],
                }
            )

        return param_grid

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit SVM model with parameter tuning.

        Performs grid search to find optimal parameters for the chosen
        kernel type.

        Args:
            X: Training feature matrix of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Notes:
            - Uses probability estimation for better prediction granularity
            - Employs parallel processing for faster grid search
            - May take longer for larger datasets due to quadratic complexity
        """
        base_model = SVC(
            kernel=self.kernel, random_state=self.random_state, probability=True
        )

        param_grid = self._get_param_grid()

        grid_search = GridSearchCV(
            base_model, param_grid, cv=self.cv, scoring="accuracy", n_jobs=-1
        )

        print(f"Training SVM with {self.kernel} kernel...")
        print("This may take a while for larger datasets.")

        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print(f"Best parameters found: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters of the fitted model.

        Returns:
            Dictionary containing:
                - model_type: Type of classifier
                - kernel: Kernel function used
                - best_params: Best parameters found by grid search
                - cv: Number of cross-validation folds used
        """
        return {
            "model_type": self.model_type,
            "kernel": self.kernel,
            "best_params": self.best_params,
            "cv": self.cv,
        }

    # Add these methods to svm.py

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Args:
            X: Test samples of shape (n_samples, n_features)

        Returns:
            Array of predicted class labels

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Args:
            X: Test samples of shape (n_samples, n_features)

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            X_test: Test features of shape (n_samples, n_features)
            y_test: True labels of shape (n_samples,)

        Returns:
            Dictionary containing evaluation metrics:
                - accuracy: Overall classification accuracy
                - precision: Precision score (micro-averaged)
                - recall: Recall score (micro-averaged)
                - f1_score: F1 score (micro-averaged)
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")

        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

    def _validate_kernel(self, kernel: str) -> None:
        """Validate the kernel type.

        Args:
            kernel: Kernel type to validate

        Raises:
            ValueError: If kernel is not one of the supported types
        """
        valid_kernels = ["linear", "rbf", "poly", "sigmoid"]
        if kernel not in valid_kernels:
            raise ValueError(
                f"Invalid kernel type '{kernel}'. "
                f"Must be one of: {', '.join(valid_kernels)}"
            )
