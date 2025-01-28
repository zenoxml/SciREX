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
    Example Script: linear_regression_example.py

    This script demonstrates how to use the LinearRegressionModel class
    from the SciREX library to perform regression on a synthetic dataset.

    The example includes:
        - Generating synthetic regression data
        - Training a Linear Regression model
        - Evaluating model performance using regression metrics
        - Visualizing the regression results

    Dependencies:
        - numpy
        - matplotlib
        - scikit-learn
        - scirex.core.ml.supervised.regression.linear_regression

    Authors:
        - Paranidharan (paranidharan@iisc.ac.in)

    Version Info:
        - 16/Jan/2025: Initial version
"""

import numpy as np
from sklearn.datasets import make_regression
from scirex.core.ml.supervised.regression.linear_regression import LinearRegressionModel

# Step 1: Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Step 2: Initialize the Linear Regression model
model = LinearRegressionModel(random_state=42)

# Step 3: Fit the model on the training data
model.fit(X, y)

# Step 4: Make predictions on the training data
y_pred = model.predict(X)

# Step 5: Get model parameters (coefficients and intercept)
params = model.get_model_params()
print("Model Parameters:")
print(f"Coefficients: {params['coefficients']}")
print(f"Intercept: {params['intercept']}")

# Step 6: Evaluate the model using regression metrics
metrics = model.evaluation_metrics(y, y_pred)
print("\nEvaluation Metrics:")
print(f"MSE: {metrics['mse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
print(f"R2 Score: {metrics['r2']:.2f}")

# Step 7: Visualize the regression results
model.plot_regression_results(y, y_pred)
