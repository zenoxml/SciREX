import unittest
import numpy as np
from scirex.core.ml.supervised.regression.lasso_regression import LassoRegressionModel

class TestLassoRegression(unittest.TestCase):
    def setUp(self):
        # Sample dataset for testing
        self.X = np.array([[1], [2], [3]])
        self.y = np.array([2, 4, 6])
        self.model = LassoRegressionModel(alpha=1.0)

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.model, "Model is not initialized after fitting.")

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y), "Prediction length mismatch.")

    def test_get_model_params(self):
        self.model.fit(self.X, self.y)
        params = self.model.get_model_params()
        self.assertIn("coefficients", params, "Model parameters missing 'coefficients'.")
        self.assertIn("intercept", params, "Model parameters missing 'intercept'.")
