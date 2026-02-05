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

"""Decision Tree classification implementation for SciREX.

This module provides a Decision Tree implementation using scikit-learn
with automatic parameter tuning using grid search. The implementation
focuses on both accuracy and interpretability.

Mathematical Background:
    Decision Trees recursively partition the feature space using:
    
    1. Splitting Criteria:
       - Gini Impurity: 1 - ∑ᵢpᵢ²
       - Entropy: -∑ᵢpᵢlog(pᵢ)
       where pᵢ is the proportion of class i in the node
    
    2. Information Gain:
       IG(parent, children) = I(parent) - ∑(nⱼ/n)I(childⱼ)
       where I is impurity measure (Gini or Entropy)
    
    3. Tree Pruning:
       Cost-Complexity: Rα(T) = R(T) + α|T|
       where R(T) is tree error, |T| is tree size, α is complexity parameter

Key Features:
    - Automatic parameter optimization
    - Multiple splitting criteria
    - Built-in tree visualization
    - Pruning capabilities
    - Feature importance estimation

References:
    [1] Breiman, L., et al. (1984). Classification and Regression Trees
    [2] Quinlan, J. R. (1986). Induction of Decision Trees
    [3] Hastie, T., et al. (2009). Elements of Statistical Learning, Ch. 9
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import GridSearchCV

from .base import Classification


class DecisionTreeClassifier(Classification):
    """Decision Tree with automatic parameter tuning.

    This implementation includes automatic selection of optimal parameters
    using grid search with cross-validation. It balances model complexity
    with performance through pruning and parameter optimization.

    Attributes:
        cv: Number of cross-validation folds
        best_params: Best parameters found by grid search
        model: Fitted DecisionTreeClassifier instance

    Example:
        >>> classifier = DecisionTreeClassifier(cv=5)
        >>> X_train = np.array([[1, 2], [2, 3], [3, 4]])
        >>> y_train = np.array([0, 0, 1])
        >>> classifier.fit(X_train, y_train)
        >>> print(classifier.best_params)
    """

    def __init__(self, cv: int = 5, **kwargs: Any) -> None:
        """Initialize Decision Tree classifier.

        Args:
            cv: Number of cross-validation folds. Defaults to 5.
            **kwargs: Additional keyword arguments passed to parent class.

        Notes:
            The classifier uses GridSearchCV for parameter optimization,
            searching over different tree depths, splitting criteria,
            and minimum sample thresholds.
        """
        super().__init__("decision_tree", **kwargs)
        self.cv = cv
        self.best_params: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Decision Tree model with parameter tuning.

        Performs grid search over tree parameters to find optimal
        model configuration using cross-validation.

        Args:
            X: Training feature matrix of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Notes:
            The grid search optimizes over:
            - Splitting criterion (gini vs entropy)
            - Maximum tree depth
            - Minimum samples for splitting
            - Minimum samples per leaf
            - Maximum features considered per split
        """
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7, 9, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }

        base_model = DTC(random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=self.cv, scoring="accuracy", n_jobs=-1
        )

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
                - best_params: Best parameters found by grid search
                - cv: Number of cross-validation folds used
        """
        return {
            "model_type": self.model_type,
            "best_params": self.best_params,
            "cv": self.cv,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature indices to importance scores

        Raises:
            ValueError: If model hasn't been fitted yet

        Notes:
            Feature importance is computed based on the decrease in
            impurity (Gini or entropy) brought by each feature across
            all tree splits.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_dict = {}
        for idx, importance in enumerate(self.model.feature_importances_):
            importance_dict[f"feature_{idx}"] = importance

        return importance_dict

    # Add these methods to decision_tree.py

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
