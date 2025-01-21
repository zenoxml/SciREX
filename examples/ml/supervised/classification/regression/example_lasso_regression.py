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
    Example Script: example_lasso_regress.py

    This script demonstrates how to use the LassoRegressionModel class
    from the SciREX library to perform regression on synthetic data.

    The example includes:
        - Generating synthetic regression data using sklearn
        - Fitting the Lasso Regression model
        - Evaluating the model's performance using regression metrics
        - Visualizing the results

    Dependencies:
        - numpy
        - scikit-learn
        - matplotlib
        - scirex.core.ml.supervised.regression.lasso_regression

    Authors:
        - Paranidharan (paranidharan@iisc.ac.in)

    Version Info:
        - 16/Jan/2025: Initial version
"""

import numpy as np
from sklearn.datasets import make_regression
from scirex.core.ml.supervised.regression.lasso_regression import LassoRegressionModel

# Generate synthetic regression data
X, y = make_regression(
    n_samples=100, 
    n_features=1, 
    noise=10, 
    random_state=42
)

# Initialize the Lasso Regression model
lasso_model = LassoRegressionModel(alpha=1.0, random_state=42)

# Fit the model
lasso_model.fit(X, y)

# Make predictions
y_pred_lasso = lasso_model.predict(X)

# Get the model parameters
params_lasso = lasso_model.get_model_params()

# Print model parameters
print("Lasso Regression Model Parameters:")
print(f"Coefficients: {params_lasso['coefficients']}")
print(f"Intercept: {params_lasso['intercept']}")

# Evaluate the model's performance
metrics_lasso = lasso_model.evaluation_metrics(y, y_pred_lasso)
print("\nLasso Regression Evaluation Metrics:")
print(f"MSE: {metrics_lasso['mse']:.2f}")
print(f"MAE: {metrics_lasso['mae']:.2f}")
print(f"R2 Score: {metrics_lasso['r2']:.2f}")

# Visualize the regression results
lasso_model.plot_regression_results(y, y_pred_lasso)
