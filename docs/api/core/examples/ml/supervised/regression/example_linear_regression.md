

::: examples.ml.supervised.regression.example_linear_regression

# Linear Regression Example Documentation

## Introduction

This document demonstrates the usage of the `LinearRegressionModel` class from the **SciREX** library to perform regression on synthetic data. Linear Regression provides a baseline regression model without regularization, making it an interpretable and widely used technique for regression tasks.

---

## Workflow Overview

The example covers the following steps:

1. Generating synthetic regression data using `sklearn`.
2. Fitting the Linear Regression model.
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

## Linear Regression Example Script

Here is the complete example script for Linear Regression:

```python
import numpy as np
from sklearn.datasets import make_regression
from scirex.core.ml.supervised.regression.linear_regression import LinearRegressionModel

# Generate synthetic regression data
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Initialize the Linear Regression model
model = LinearRegressionModel(random_state=42)

# Fit the model on the training data
model.fit(X, y)

# Make predictions on the training data
y_pred = model.predict(X)

# Get model parameters (coefficients and intercept)
params = model.get_model_params()
print("Model Parameters:")
print(f"Coefficients: {params['coefficients']}")
print(f"Intercept: {params['intercept']}")

# Evaluate the model using regression metrics
metrics = model.evaluation_metrics(y, y_pred)
print("\nEvaluation Metrics:")
print(f"MSE: {metrics['mse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
print(f"R2 Score: {metrics['r2']:.2f}")

# Visualize the regression results
model.plot_regression_results(y, y_pred)
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

Initialize the `LinearRegressionModel` with an optional random seed for reproducibility:

```python
model = LinearRegressionModel(random_state=42)
```

### 3. Fitting the Model

Train the model using the `fit` method:

```python
model.fit(X, y)
```

### 4. Predictions

Make predictions using the trained model:

```python
y_pred = model.predict(X)
```

### 5. Retrieving Model Parameters

Retrieve the coefficients and intercept:

```python
params = model.get_model_params()
```

### 6. Evaluating Performance

Evaluate the model using common regression metrics such as MSE, MAE, and R2 Score:

```python
metrics = model.evaluation_metrics(y, y_pred)
```

### 7. Visualization

Visualize the regression results using the built-in plotting function:

```python
model.plot_regression_results(y, y_pred)
```

---

## Sample Output

```
Model Parameters:
Coefficients: 
Intercept: 

Evaluation Metrics:
MSE: 
MAE: 
R2 Score: 
```

The visualization will show the actual vs. predicted values on a scatter plot with the regression line.

---

## Conclusion

This example showcases how to use the `LinearRegressionModel` class from SciREX for regression tasks. Linear Regression provides a simple and interpretable baseline model, making it suitable for many applications.

For more advanced use cases and detailed documentation, visit the [SciREX Documentation](https://scirex.org).

---

## Author

This example was authored by **Paranidharan** ([paranidharan@iisc.ac.in](mailto:paranidharan@iisc.ac.in)).

---

## License

This document is part of the **SciREX** library and is licensed under the Apache License 2.0. For more information, visit [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

