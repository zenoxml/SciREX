# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab,
# Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# SciREX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with SciREX. If not, see <https://www.gnu.org/licenses/>.
#
# For any clarifications or special considerations,
# please contact <scirex@zenteiq.ai>

"""
   Example Script: quantization_base_mnist.py
   
   This script demonstrates quantization-aware training on the MNIST dataset. It gives the performance of the baseline model.

   This example includes:
       - Training baseline model on MNIST dataset
       - Evaluating model accuracy before QAT
       

   Authors:
       - Nithyashree R (nithyashreer@iisc.ac.in)

   Version Info:
       - 06/01/2024: Initial version
"""

import numpy as np
import tensorflow as tf
from scirex.core.model_compression.quantization import QuantizationAwareTraining

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize QAT with model architecture
qat = QuantizationAwareTraining(input_shape=(28, 28), num_classes=10)

# Train base model
print("Training base model...")
qat.train(train_images, train_labels, epochs=10)

# Evaluate base model directly
baseline_accuracy = qat.model.evaluate(test_images, test_labels, verbose=0)[
    1
]  # Gets accuracy
print(f"\nBaseline Model Accuracy: {baseline_accuracy:.4f}")
