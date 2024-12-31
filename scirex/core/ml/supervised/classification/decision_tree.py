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
    
    def __init__(
        self,
        cv: int = 5,
        **kwargs: Any
    ) -> None:
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
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 9, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        base_model = DTC(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1
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
            "cv": self.cv
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