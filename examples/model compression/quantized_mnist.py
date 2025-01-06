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
   Example Script: quantized_mnist.py
   
   This script demonstrates model quantization techniques on the MNIST dataset. Compares accuracy of different quantization approaches.


   This example includes:
       - Applying quantization-aware training (QAT)
       - Implementing post-training quantization 
       - Comparing model sizes and accuracies

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

# Get initial model size
initial_model = tf.lite.TFLiteConverter.from_keras_model(qat.model).convert()
initial_size = len(initial_model) / float(2**20)  # Convert to MB

# Apply quantization-aware training
print("\nApplying quantization-aware training...")
qat.apply_quantization_aware_training()
qat.train_q_aware_model(train_images, train_labels, epochs=5)

# Get quantization-aware accuracy
qat_accuracy = qat.evaluate(test_images, test_labels)[1]
print(f"\nQuantization-Aware Model Accuracy: {qat_accuracy}")  # Removed .4f format

# Get QAT model size
qat_tflite = qat.convert_to_tflite()
qat_size = len(qat_tflite) / float(2**20)

# Get post-training quantization size
post_quant_tflite = qat.post_quantization()
post_quant_size = len(post_quant_tflite) / float(2**20)

# Print size comparisons
print("\nModel Size Comparison:")
print(f"Original Model Size: {initial_size:.2f} MB")
print(f"QAT Model Size: {qat_size:.2f} MB")
print(f"Post-Training Quantized Size: {post_quant_size:.2f} MB")
print(f"\nSize Reduction from QAT: {((initial_size - qat_size)/initial_size)*100:.1f}%")
print(
    f"Size Reduction from Post-Training: {((initial_size - post_quant_size)/initial_size)*100:.1f}%"
)
