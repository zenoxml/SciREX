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
This script applies Quantization-Aware Training on the MNIST dataset to reduce model size and memory usage while maintaining accuracy.

Authors:
 - Nithyashree R (nithyashreer@iisc.ac.in)

"""


import numpy as np
import tensorflow as tf
from quantized import QuantizationAwareTraining


def main():
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize the input images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    batch_size = 35

    # Initialize QuantizationAwareTraining
    quantization_handler = QuantizationAwareTraining(
        input_shape=(28, 28),
        num_classes=10,
        filters=12,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        baseline_epochs=10,
        qat_epochs=4,
        batch_size=batch_size,
        validation_split=0.1,
    )

    # Run complete workflow
    quantization_handler.run_complete_workflow(
        train_images, train_labels, test_images, test_labels
    )


if __name__ == "__main__":
    # Enable memory growth for GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    try:
        main()
    except Exception as e:
        print(f"\nError occurred during execution: {e}")
