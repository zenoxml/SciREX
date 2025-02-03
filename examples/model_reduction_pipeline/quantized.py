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
    Example Script: Quantization.py
    
Authors:
 - Nithyashree R (nithyashreer@iisc.ac.in)

"""

import os
import csv
import time
import psutil
import platform
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow_model_optimization as tfmot


class QuantizationAwareTraining:
    """
    A comprehensive class for performing quantization-aware training with performance metrics.

    Attributes:
        model (tf.keras.Model): The base model architecture.
        q_aware_model (tf.keras.Model): The quantization-aware trained model.
    """

    def __init__(
        self,
        input_shape=(28, 28),
        num_classes=10,
        filters=12,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        baseline_epochs=10,
        qat_epochs=4,
        batch_size=500,
        validation_split=0.1,
    ):
        """
        Initializes the model architecture and training parameters.

        Args:
            input_shape: Shape of input data (default: (28, 28))
            num_classes: Number of output classes (default: 10)
            filters: Number of filters for Conv2D layer (default: 12)
            kernel_size: Kernel size for Conv2D layer (default: (3, 3))
            pool_size: Pool size for MaxPooling2D layer (default: (2, 2))
            epochs: Number of training epochs (default: 10)
            batch_size: Batch size for training (default: 500)
            validation_split: Fraction of data for validation (default: 0.1)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.baseline_epochs = baseline_epochs
        self.qat_epochs = qat_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Initialize models
        self.model = self._build_model()
        self.q_aware_model = None

        # Performance metrics
        self.baseline_accuracy = None
        self.q_aware_accuracy = None
        self.baseline_time = None
        self.q_aware_time = None
        self.model_sizes = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self):
        """Collects system information for metrics reporting."""
        info = {
            "processor": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "total_memory": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
            "gpu_available": "No",
            "gpu_name": "None",
        }

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["gpu_available"] = "Yes"
            try:
                import subprocess

                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
                ).decode()
                info["gpu_name"] = result.strip()
            except:
                info["gpu_name"] = f"GPU Device {len(gpus)}"

        return info

    def _build_model(self):
        """Builds the base model architecture."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.input_shape),
                tf.keras.layers.Reshape(target_shape=self.input_shape + (1,)),
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=self.pool_size),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.num_classes),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def train_baseline(self, train_data, train_labels):
        """Trains the baseline model with timing."""
        start_time = time.time()

        self.model.fit(
            train_data,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.baseline_epochs,
            validation_split=self.validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            ],
        )

        self.baseline_time = time.time() - start_time

    def apply_quantization_aware_training(self):
        """Applies quantization-aware training to the base model."""
        quantize_model = tfmot.quantization.keras.quantize_model
        self.q_aware_model = quantize_model(self.model)

        self.q_aware_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train_q_aware_model(self, train_data, train_labels):
        """Trains the quantization-aware model with timing."""
        if self.q_aware_model is None:
            raise ValueError(
                "Quantization-aware model is not initialized. Call apply_quantization_aware_training first."
            )

        start_time = time.time()

        self.q_aware_model.fit(
            train_data,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.qat_epochs,
            validation_split=self.validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            ],
        )

        self.q_aware_time = time.time() - start_time

    def evaluate(self, test_data, test_labels):
        """Evaluates both models and stores accuracies."""
        self.baseline_accuracy = self.model.evaluate(test_data, test_labels, verbose=0)[
            1
        ]
        self.q_aware_accuracy = self.q_aware_model.evaluate(
            test_data, test_labels, verbose=0
        )[1]
        return self.baseline_accuracy, self.q_aware_accuracy

    def convert_to_tflite(self):
        """Converts both models to TFLite format and measures sizes without saving files."""
        # Convert baseline model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        baseline_tflite = converter.convert()

        # Convert quantized model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite = converter.convert()

        # Calculate and store sizes without saving files
        self.model_sizes["baseline_model.tflite"] = len(baseline_tflite) / float(
            2**20
        )  # Size in MB
        self.model_sizes["quantized_model.tflite"] = len(quantized_tflite) / float(
            2**20
        )  # Size in MB

        return baseline_tflite, quantized_tflite

    def _save_metrics(self):
        """Saves performance metrics to CSV file and prints to terminal."""
        # Get downloads directory path
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"quantization_metrics_{timestamp}.csv"
        file_path = os.path.join(downloads_dir, filename)

        # Calculate compression ratio
        baseline_size = self.model_sizes.get("baseline_model.tflite", 0)
        quantized_size = self.model_sizes.get("quantized_model.tflite", 0)
        compression_ratio = baseline_size / quantized_size if quantized_size > 0 else 0

        # Print results to terminal with the exact format
        print("\n=== Model Quantization Results ===")
        print("SYSTEM INFORMATION")
        print(f"Processor: {self.system_info['processor']}")
        print(f"CPU Cores: {self.system_info['cpu_cores']}")
        print(f"CPU Threads: {self.system_info['cpu_threads']}")
        print(f"Total Memory: {self.system_info['total_memory']}")
        print(f"GPU Available: {self.system_info['gpu_available']}")
        print(f"GPU Name: {self.system_info['gpu_name']}")
        print("\nMODEL PERFORMANCE")
        print(
            f"{'Model Type':<15} {'Accuracy':<12} {'Training Time':<15} {'Compression Ratio'}"
        )
        print(
            f"{'Baseline Model':<15} {self.baseline_accuracy:.4f}      {self.baseline_time:.2f}s            1.00x"
        )
        print(
            f"{'Quantized Model':<15} {self.q_aware_accuracy:.4f}      {self.q_aware_time:.2f}s            {compression_ratio:.2f}x"
        )
        print("=" * 30)

        # Prepare data for CSV in the exact format requested
        data = [
            ["Model Quantization Results"],
            ["SYSTEM INFORMATION"],
            [f"Processor: {self.system_info['processor']}"],
            [f"CPU Cores: {self.system_info['cpu_cores']}"],
            [f"CPU Threads: {self.system_info['cpu_threads']}"],
            [f"Total Memory: {self.system_info['total_memory']}"],
            [f"GPU Available: {self.system_info['gpu_available']}"],
            [f"GPU Name: {self.system_info['gpu_name']}"],
            [],
            ["MODEL PERFORMANCE"],
            ["Model Type", "Accuracy", "Training Time", "Compression Ratio"],
            [
                "Baseline Model",
                f"{self.baseline_accuracy:.4f}",
                f"{self.baseline_time:.2f}s",
                "1.00x",
            ],
            [
                "Quantized Model",
                f"{self.q_aware_accuracy:.4f}",
                f"{self.q_aware_time:.2f}s",
                f"{compression_ratio:.2f}x",
            ],
        ]

        try:
            # Create Downloads directory if it doesn't exist
            os.makedirs(downloads_dir, exist_ok=True)

            # Save CSV file
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
            print(f"\nResults saved to: {file_path}")

        except Exception as e:
            print(f"\nError saving file: {e}")

    def run_complete_workflow(self, train_data, train_labels, test_data, test_labels):
        """Executes the complete quantization workflow with metrics collection."""
        print("Starting quantization workflow...")

        # Train baseline model
        print("\nTraining baseline model...")
        self.train_baseline(train_data, train_labels)

        # Apply and train quantized model
        print("\nApplying quantization...")
        self.apply_quantization_aware_training()
        print("Training quantized model...")
        self.train_q_aware_model(train_data, train_labels)

        # Evaluate both models
        print("\nEvaluating models...")
        self.evaluate(test_data, test_labels)

        # Convert to TFLite and measure sizes
        print("\nConverting models to TFLite...")
        self.convert_to_tflite()

        # Save metrics
        self._save_metrics()
