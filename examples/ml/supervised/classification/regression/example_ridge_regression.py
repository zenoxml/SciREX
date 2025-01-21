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
    Example Script: example_ridge_regress.py

    This script demonstrates how to use the RidgeRegressionModel class
    from the SciREX library to perform regression on synthetic data.

    The example includes:
        - Generating synthetic regression data using sklearn
        - Fitting the Ridge Regression model
        - Evaluating the model's performance using regression metrics
        - Visualizing the results

    Dependencies:
        - numpy
        - scikit-learn
        - matplotlib
        - scirex.core.ml.supervised.regression.ridge_regression

    Authors:
        - Paranidharan (paranidharan@iisc.ac.in)

    Version Info:
        - 16/Jan/2025: Initial version
"""

import numpy as np
from sklearn.datasets import make_regression
from scirex.core.ml.supervised.regression.ridge_regression import RidgeRegressionModel

# Generate synthetic regression data
X, y = make_regression(
    n_samples=100, 
    n_features=1, 
    noise=10, 
    random_state=42
)

# Initialize the Ridge Regression model
ridge_model = RidgeRegressionModel(random_state=42)

# Fit the model
ridge_model.fit(X, y)

# Make predictions
y_pred_ridge = ridge_model.predict(X)

# Get the model parameters
params_ridge = ridge_model.get_model_params()

# Print model parameters
print("Ridge Regression Model Parameters:")
print(f"Coefficients: {params_ridge['coefficients']}")
print(f"Intercept: {params_ridge['intercept']}")

# Evaluate the model's performance
metrics_ridge = ridge_model.evaluation_metrics(y, y_pred_ridge)
print("\nRidge Regression Evaluation Metrics:")
print(f"MSE: {metrics_ridge['mse']:.2f}")
print(f"MAE: {metrics_ridge['mae']:.2f}")
print(f"R2 Score: {metrics_ridge['r2']:.2f}")

# Visualize the regression results
ridge_model.plot_regression_results(y, y_pred_ridge)
