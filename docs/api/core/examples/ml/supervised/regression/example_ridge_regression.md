

::: examples.ml.supervised.regression.example_ridge_regression

# Ridge Regression Example Documentation

## Introduction

This example demonstrates the usage of the `RidgeRegressionModel` class from the **SciREX** library to perform regression on synthetic data. Ridge Regression is a regularization technique that helps mitigate overfitting by adding a penalty proportional to the magnitude of the coefficients.

---

## Workflow Overview

The example covers the following steps:

1. Generating synthetic regression data using `sklearn`.
2. Fitting the Ridge Regression model.
3. Evaluating the model's performance using regression metrics.
4. Visualizing the results.

---

## Requirements

Ensure the following dependencies are installed before running the script:

- `numpy`
- `scikit-learn`
- `matplotlib`
- `SciREX` library

Install missing dependencies using pip:
```bash
pip install numpy scikit-learn matplotlib
```

---

## Example Script

Here is the complete example script:

```python
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
```

---

## Key Steps Explained

### 1. Generating Synthetic Data

We use `sklearn.datasets.make_regression` to create a synthetic dataset with one feature and added noise for demonstration purposes:

```python
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)
```

### 2. Initializing the Model

We initialize the `RidgeRegressionModel` with a fixed random state for reproducibility:

```python
ridge_model = RidgeRegressionModel(random_state=42)
```

### 3. Fitting the Model

Train the model using the `fit` method:

```python
ridge_model.fit(X, y)
```

### 4. Predictions

Make predictions using the trained model:

```python
y_pred_ridge = ridge_model.predict(X)
```

### 5. Retrieving Model Parameters

Retrieve the coefficients and intercept:

```python
params_ridge = ridge_model.get_model_params()
```

### 6. Evaluating Performance

Evaluate the model using common regression metrics such as MSE, MAE, and R2 Score:

```python
metrics_ridge = ridge_model.evaluation_metrics(y, y_pred_ridge)
```

### 7. Visualization

Visualize the regression results using the built-in plotting function:

```python
ridge_model.plot_regression_results(y, y_pred_ridge)
```

---

## Sample Output

Here’s an example of the output you can expect:

```
Ridge Regression Model Parameters:
Coefficients: 
Intercept:

Ridge Regression Evaluation Metrics:
MSE: 
MAE: 
R2 Score:
```

The visualization will show the actual vs. predicted values on a scatter plot with the regression line.

---

## Conclusion

This example showcases how to use the `RidgeRegressionModel` class from SciREX for regression tasks. Ridge Regression is particularly useful when dealing with multicollinearity or overfitting in datasets.

For more advanced use cases and detailed documentation, visit the [SciREX Documentation](https://scirex.org).

---

## Author

This example was authored by **Paranidharan** ([paranidharan@iisc.ac.in](mailto:paranidharan@iisc.ac.in)).

---

## License

This script is part of the **SciREX** library and is licensed under the Apache License 2.0. For more information, visit [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

