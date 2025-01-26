
::: examples.ml.supervised.regression.example_lasso_regression

# Lasso Regression Example Documentation

## Introduction

This example demonstrates the usage of the `LassoRegressionModel` class from the **SciREX** library to perform regression on synthetic data. Lasso Regression is a regularization technique that helps mitigate overfitting by adding a penalty proportional to the absolute value of the coefficients, leading to sparse models.

---

## Workflow Overview

The example covers the following steps:

1. Generating synthetic regression data using `sklearn`.
2. Fitting the Lasso Regression model.
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

We initialize the `LassoRegressionModel` with an alpha value for regularization and a fixed random state for reproducibility:

```python
lasso_model = LassoRegressionModel(alpha=1.0, random_state=42)
```

### 3. Fitting the Model

Train the model using the `fit` method:

```python
lasso_model.fit(X, y)
```

### 4. Predictions

Make predictions using the trained model:

```python
y_pred_lasso = lasso_model.predict(X)
```

### 5. Retrieving Model Parameters

Retrieve the coefficients and intercept:

```python
params_lasso = lasso_model.get_model_params()
```

### 6. Evaluating Performance

Evaluate the model using common regression metrics such as MSE, MAE, and R2 Score:

```python
metrics_lasso = lasso_model.evaluation_metrics(y, y_pred_lasso)
```

### 7. Visualization

Visualize the regression results using the built-in plotting function:

```python
lasso_model.plot_regression_results(y, y_pred_lasso)
```

---

## Sample Output

Here’s an example of the output you can expect:

```
Lasso Regression Model Parameters:
Coefficients: 
Intercept: 

Lasso Regression Evaluation Metrics:
MSE: 
MAE:
R2 Score:
```

The visualization will show the actual vs. predicted values on a scatter plot with the regression line.

---

## Conclusion

This example showcases how to use the `LassoRegressionModel` class from SciREX for regression tasks. Lasso Regression is particularly useful when feature selection and sparsity are important in models.

For more advanced use cases and detailed documentation, visit the [SciREX Documentation](https://scirex.org).

---

## Author

This example was authored by **Paranidharan** ([paranidharan@iisc.ac.in](mailto:paranidharan@iisc.ac.in)).

---

## License

This script is part of the **SciREX** library and is licensed under the Apache License 2.0. For more information, visit [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

