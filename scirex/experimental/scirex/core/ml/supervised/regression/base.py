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

    This module provides the abstract base class for all regression algorithms in SciREX.
    It defines shared functionality for:
        - Data preparation (loading from CSV and standard scaling)
        - Regression performance metric computation (MSE, MAE, R2 score)

    Classes:
        Regression: Abstract base class that outlines common behavior for regression algorithms.

    Dependencies:
        - numpy, pandas, sklearn
        - abc, pathlib, time, typing (for structural and type support)

    Key Features:.
        - Consistent interface for loading and preparing data
        - Standard approach to computing and returning regression metrics
        - Enforces subclasses to implement `fit`, `predict`, and `get_model_params`

    Authors:
        - Paranidharan (paranidharan@iisc.ac.in)

    Version Info:
        - 16/Jan/2025: Initial version
"""

# base for regression algorithms in the SciREX library.
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
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


class Regression(ABC):
    """
    Abstract base class for regression algorithms in the SciREX library.

    This class provides:
      - A consistent interface for loading and preparing data
      - A standard approach to computing and returning regression metrics (MSE, MAE, R2)
      - A method for plotting regression results

    Subclasses must:
      1. Implement the `fit(X: np.ndarray, y: np.ndarray) -> None` method, which should populate `self.model`.
      2. Implement the `get_model_params() -> Dict[str, Any]` method, which returns a dict of model parameters for logging/debugging.

    Attributes:
        model_type (str): The name or identifier of the regression model (e.g., "linear_regression", "random_forest").
        random_state (int): Random seed for reproducibility.
        model (Optional): The trained regression model.
        plots_dir (Path): Directory where regression plots will be saved.
    """

    def __init__(
        self,
        model_type: str,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the base regression class.

        Args:
            model_type (str): A string identifier for the regression algorithm
                              (e.g. "linear_regression", "random_forest", etc.).
            random_state (int, optional): Seed for reproducibility where applicable.
                                          Defaults to 42.
        """
        self.model_type = model_type
        self.random_state = random_state

        # initialize the model to None
        self.model = None

        # create a directory to save the plots
        self.plots_dir = Path.cwd() / "Plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from a CSV file.

        1.This method reads the dataset from the specified path, drops any rows with missing values,
        2.Scales the features using StandardScaler.


        Args:
           path (str): The path to the dataset.

        Returns:
           Tuple[np.ndarray, np.ndarray]: A tuple of prepared features (X) and target values (y).
        """
        df = pd.read_csv(Path(path))
        df.dropna(inplace=True)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if numeric_columns.empty:
            raise ValueError("No numeric columns found in the dataset.")

        # Split into features (X) and target (y)
        X = df[numeric_columns[:-1]].values  # All numeric columns except the last one
        y = df[numeric_columns[-1]].values  # Last numeric column as the target

        # Scale the features
        X = StandardScaler().fit_transform(X)
        return X, y

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model to the training data.

        Args:
            X (np.ndarray): The input features for training the model.
            y (np.ndarray): The target values for training the model.

        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained regression model.

        Args:
            X (np.ndarray): The input features for generating predictions.

        Returns:
            np.ndarray: The predicted target values.
        """
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the model parameters as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary of model parameters.
        """
        pass

    def evaluation_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute and return regression evaluation metrics.

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted target values.

        Returns:
            Dict[str, float]: A dictionary of regression evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }

    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
        """
        Plot the regression results.

        Args:
          y_true (np.ndarray): The true target values.
          y_pred (np.ndarray): The predicted target values.

        Returns:
        Figure: The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, color="blue", label="Predictions")
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            ls="--",
            color="red",
            label="Perfect Predictions",
        )
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{self.model_type} Regression Results")
        ax.legend()
        plt.tight_layout()

        # Save the plot
        plot_path = self.plots_dir / f"regression_results_{self.model_type}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig

    def run(
        self,
        data: Optional[np.array] = None,
        path: Optional[str] = None,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Run the complete regression pipeline: data loading/preprocessing, fitting the model, and computing regression metrics on the test set.

        Args:
        data (optional,[np.array]): preprocessed adta array of shape (n_samples, n_features).
        path(Optional[str]): The path to the dataset.
        test_size(float): The proportion of the dataset for test set is to set as default 0.2.

        Returns:
        Dict[str,Any]: A dictionary with the following keys:
            - "params" (Dict[str, Any]): Model parameters from 'self.get_model_params()
            - "MSE" (float): Mean Squared Error of the regression model.
            -  "MAE" (float): Mean Absolute Error of the regression model.
            - "R2" (float): R =-Squared Score of the regression model.
        """
        if data is None and path is None:
            raise ValueError("Either 'data' or 'path' must be provided.")

        # Load and prepare the data
        # If data is not provided, load the data from the specified path
        # If data is provided, use it directly
        X, y = data if data is not None else self.prepare_data(path)

        # spliting the dataset into training set and testing set
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Fit the model on the training data
        self.fit(x_train, y_train)

        # check model and make predictions
        if self.model is None:
            raise ValueError(
                f"{self.model_type} model is not trained. Ensure the 'fit' method is called before predictions."
            )
        y_pred = self.predict(x_test)

        # compute regression metrics
        metrics = self.evaluation_metrics(y_test, y_pred)

        # plot the regression results
        fig = self.plot_regression_results(y_test, y_pred)

        # return the model parameters and evaluation metrics
        return {
            "params": self.get_model_params(),
            "MSE": metrics["mse"],
            "MAE": metrics["mae"],
            "R2": metrics["r2"],
            "plot": fig,
        }
