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
    Example Script: logistic_regression_example.py

    This script demonstrates how to use the LogisticRegression class
    from the SciREX library to perform classification on the Iris dataset.

    The example includes:
        - Loading the Iris dataset using sklearn.datasets
        - Preprocessing the data
        - Splitting the data into train and test sets
        - Running the Logistic Regression classifier
        - Evaluating and visualizing the results

    Dependencies:
        - numpy
        - pandas
        - matplotlib
        - scikit-learn
        - scirex.core.ml.supervised.classification.logistic_regression

    Authors:
        - Protyush P. Chowdhury (protyushc@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scirex.core.ml.supervised.classification.logistic_regression import (
    LogisticRegressionClassifier,
)

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into train and test sets (default split is 80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Logistic Regression classifier
logistic_model = LogisticRegressionClassifier()

# Run the classifier and get evaluation metrics
results = logistic_model.run(data=X_train, labels=y_train, split_ratio=0.2)

# Print the evaluation results
print("Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value}")

# Plot the confusion matrix
logistic_model.plot(y_test, logistic_model.predict(X_test))
