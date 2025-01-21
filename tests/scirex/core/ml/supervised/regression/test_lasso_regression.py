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
