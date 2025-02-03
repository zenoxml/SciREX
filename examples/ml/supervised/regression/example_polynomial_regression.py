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
    Example Usage: Polynomial Regression on a Sample Dataset

    This example demonstrates how to use the PolynomialRegressionModel class from the SciREX library
    to perform regression on a synthetic dataset with non-linear relationships.

    Steps:
        1. Prepare the data (using a CSV file or a synthetic dataset).
        2. Initialize the Polynomial Regression model.
        3. Fit the model to the training data.
        4. Make predictions and evaluate the model.
        5. Visualize the regression results.

        Authors:
        - Paranidharan (paranidharan@iisc.ac.in)

    Version Info:
        - 01/Feb/2025: Initial version    
"""

from scirex.core.ml.supervised.regression.polynomial_regression import PolynomialRegressionModel

# Sample dataset (for illustration purposes, replace with actual data)
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Create a synthetic dataset with non-linear relationship
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Initialize the Polynomial Regression model
polynomial_model = PolynomialRegressionModel(degree=3)

# Train the model
polynomial_model.run(data=(X, y))

# Get model parameters
params = polynomial_model.get_model_params()
print(f"Model Parameters: {params}")

# Example output: MSE, MAE, R2, and a plot saved in the current directory

# Plotting the results (visualization of polynomial regression curve)
X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # Generate test data for plotting
y_pred = polynomial_model.predict(X_test)

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', label='Polynomial fit')
plt.title('Polynomial Regression - Degree 3')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
