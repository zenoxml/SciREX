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
    Module: base.py

    This module provides the abstract base class for all classification algorithms in SciREX.
    It defines shared functionality for:
        - Data preparation (loading from CSV and standard scaling)
        - Classification performance metric computation (accuracy, precision, recall, f1-score)

    Classes:
        Classification: Abstract base class that outlines common behavior for classification algorithms.

    Dependencies:
        - numpy, pandas, sklearn
        - abc, pathlib, time, typing (for structural and type support)

    Key Features:
        - Consistent interface for loading and preparing data
        - Standard approach to computing and returning classification metrics
        - Enforces subclasses to implement `fit`, `predict`, and `get_model_params`

    Authors:
        - Protyush P. Chowdhury (protyushc@iisc.ac.in)

    Version Info:
        - 28/Dec/2024: Initial version

"""
# Standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import time

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


class Classification(ABC):
    """
    Abstract base class for classification algorithms in the SciREX library.

    This class provides:
      - A consistent interface for loading and preparing data
      - A standard approach to computing and returning classification metrics (accuracy, precision, recall, F1-score)
      - A method for plotting confusion matrix for classification results

    Subclasses must:
      1. Implement the `fit(X: np.ndarray, y: np.ndarray) -> None` method, which should populate `self.model`.
      2. Implement the `get_model_params() -> Dict[str, Any]` method, which returns a dict of model parameters for logging/debugging.

    Attributes:
        model_type (str): The name or identifier of the classification model (e.g., "logistic_regression", "decision_tree").
        random_state (int): Random seed for reproducibility.
        model (Optional): The trained classification model.
        plots_dir (Path): Directory where confusion matrix plots will be saved.
    """

    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the base classification class.

        Args:
            model_type (str): A string identifier for the classification algorithm
                              (e.g. "logistic_regression", "decision_tree", etc.).
            random_state (int, optional): Seed for reproducibility where applicable.
                                          Defaults to 42.
        """
        self.model_type = model_type
        self.random_state = random_state

        # Directory for saving plots
        self.plots_dir = Path.cwd() / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Subclasses must set self.model after fitting
        self.model: Optional[Any] = None

    def prepare_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from a CSV file, returning features and labels.

        This method:
          1. Reads the CSV file into a pandas DataFrame.
          2. Drops rows containing NaN values.
          3. Selects only numeric columns from the DataFrame.
          4. Scales the features using scikit-learn's StandardScaler.
          5. Assumes the last column is the target label.

        Args:
            path (str): Filepath to the CSV data file.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Features dataset (X) of shape (n_samples, n_features).
                - Labels (y) of shape (n_samples,).
        """
        df = pd.read_csv(Path(path))
        df = df.dropna()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if numeric_columns.empty:
            raise ValueError("No numeric columns found in the data.")
        X = df[numeric_columns].values
        y = df[df.columns[-1]].values  # Assuming last column is the label
        return StandardScaler().fit_transform(X), y

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classification model on the training dataset.

        Args:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) containing the features.
            y (np.ndarray): A 1D array of shape (n_samples,) containing the labels.

        Subclasses must implement this method. After fitting the model,
        `self.model` should be populated with the trained model.
        """
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Return model parameters for logging or debugging.

        Returns:
            Dict[str, Any]: A dictionary containing key model parameters and
                            potentially any learned attributes (e.g., coefficients, intercept).
        """
        pass

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
        """
        Plot the confusion matrix using the true and predicted labels.

        Args:
            y_true (np.ndarray): True labels for the test data.
            y_pred (np.ndarray): Predicted labels for the test data.

        Returns:
            Figure: A matplotlib Figure object containing the confusion matrix plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        ax.set_xticklabels([""] + [str(i) for i in np.unique(y_true)])
        ax.set_yticklabels([""] + [str(i) for i in np.unique(y_true)])

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)

        plt.tight_layout()

        plot_path = self.plots_dir / f"confusion_matrix_{self.model_type}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig

    def run(
        self,
        data: Optional[np.ndarray] = None,
        path: Optional[str] = None,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Run the complete classification pipeline: data loading/preprocessing,
        fitting the model, and computing classification metrics on the test set.

        Args:
            data (Optional[np.ndarray]): Preprocessed data array of shape (n_samples, n_features).
            path (Optional[str]): Path to a CSV file from which to read data.
                                  If `data` is not provided, this must be specified.
            test_size (float): The proportion of the dataset to include in the test split (default 0.2).

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - "params" (Dict[str, Any]): Model parameters from `self.get_model_params()`
                - "accuracy" (float): Accuracy score of the classification model.
                - "precision" (float): Precision score.
                - "recall" (float): Recall score.
                - "f1_score" (float): F1-score.
        """
        if data is None and path is None:
            raise ValueError("Either 'data' or 'path' must be provided.")

        # Load/prepare data if needed
        X, y = data if data is not None else self.prepare_data(path)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Fit the model on the training data
        self.fit(X_train, y_train)

        # Check model and make predictions
        if self.model is None:
            raise ValueError("Model is not trained. Did you implement fit()?")

        y_pred = self.model.predict(X_test)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        # Return results
        return {
            "params": self.get_model_params(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
