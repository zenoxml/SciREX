# Running Classification Algorithms with the SciREX Package

## Import the Necessary Modules


To use the classification module, you will need to import the specific classifier class you want to use. For example, to use logistic regression:

```python

from scirex.core.ml.supervised.classification.logistic_regression import LogisticRegressionClassifier

```
## Load and Prepare Your Dataset

If you use your custom dataset, you can preprocess the data using pandas

```python 
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your custom dataset
# Example: Assume your dataset has columns 'feature1', 'feature2', ..., 'label'
data = pd.read_csv("path/to/your_dataset.csv")

# Separate features and labels
X = data.drop(columns=['label']).values
y = data['label'].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```




If you use libraries like numpy, pandas, or scikit-learn to load and preprocess your dataset, you can go the following way. For example, to load the Iris dataset:

```python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Initialize the classifier

Create an instance of the classification class from the classification module. For logistic regression:

```python
logistic_model = LogisticRegressionClassifier()
```

## Train and evaluate the classifier

Run the classification algorithm on your dataset. Use the run method to train the model and evaluate it:

```python
results = logistic_model.run(data=X_train, labels=y_train, split_ratio=0.2)
```

The run method will return evaluation metrics such as accuracy, precision, recall, and F1-score. You can print them as follows:


```python
print("Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value}")
```

## Visualize the result

If supported by the classifier, use the plot method to visualize results like the confusion matrix:

```python
logistic_model.plot(X_test, y_test)
```