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
Example Script: pruned_mnist.py
This script demonstrates the application of model pruning on the MNIST dataset using TensorFlow
Model Optimization Toolkit. The pruning process reduces model size while maintaining accuracy.

Authors:
 - Nithyashree R (nithyashreer@iisc.ac.in)

"""

import numpy as np
import tensorflow as tf
from pruning import ModelPruning

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize the input images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize ModelPruning
model_handler = ModelPruning(
    input_shape=(28, 28),
    num_classes=10,
    epochs=10,
    batch_size=35,
    validation_split=0.1,
)

# Train baseline model first
print("\nTraining baseline model...")
model_handler.train_baseline_model(train_images, train_labels)
baseline_accuracy = model_handler.evaluate_baseline(test_images, test_labels)

# Apply pruning to the model
print("\nApplying pruning...")
pruned_model = model_handler.apply_pruning(train_images)

# Train the pruned model
print("\nTraining pruned model...")
model_handler.train_pruned_model(train_images, train_labels)

# Evaluate the pruned model
print("\nEvaluating pruned model...")
pruned_accuracy = model_handler.evaluate_pruned_model(test_images, test_labels)

# Final comparison
print("\nModel Comparison:")
print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")
print(f"Pruned Model Accuracy: {pruned_accuracy:.4f}")
print(f"Compression Ratio: {model_handler.compression_ratio:.2f}x")
print(f"\nDetailed metrics have been saved to CSV file.")
