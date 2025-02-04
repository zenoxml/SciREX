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
    Example Script: pruning_cnn_dl.py
    
Authors:
 - Nithyashree R (nithyashreer@iisc.ac.in)

"""

import tempfile
import os
import csv
import psutil
import platform
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow_model_optimization.sparsity.keras import (
    strip_pruning as tfmot_strip_pruning,
)


class ModelPruning:
    def __init__(
        self,
        input_shape=(28, 28),
        num_classes=10,
        epochs=10,
        batch_size=35,
        validation_split=0.1,
        target_sparsity=0.75,
    ):
        self.input_shape = input_shape + (1,)
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.target_sparsity = target_sparsity
        self.model = None
        self.pruned_model = None
        self.stripped_model = None
        self.baseline_model_accuracy = None
        self.pruned_model_accuracy = None
        self.baseline_time = None
        self.pruned_time = None
        self.compression_ratio = None
        self.system_info = self._get_system_info()

        # Build the model
        self.model = self._build_model()

    def _get_system_info(self):
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
        model = keras.Sequential(
            [
                # First conv block
                keras.layers.Conv2D(4, kernel_size=4, input_shape=self.input_shape),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Activation("relu"),
                # Second conv block
                keras.layers.Conv2D(8, kernel_size=4),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Activation("relu"),
                # Flatten and dense layer
                keras.layers.Flatten(),
                keras.layers.Dense(self.num_classes),
                keras.layers.Activation("log_softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def _calculate_compression_ratio(self):
        """Calculate the compression ratio between baseline and stripped model."""

        def count_nonzero_weights(model):
            total_params = 0
            nonzero_params = 0
            for layer in model.layers:
                if not layer.weights:
                    continue
                for weight in layer.weights:
                    weight_np = weight.numpy()
                    total_params += weight_np.size
                    nonzero_params += np.count_nonzero(weight_np)
            return total_params, nonzero_params

        baseline_total, _ = count_nonzero_weights(self.model)
        _, stripped_nonzero = count_nonzero_weights(self.stripped_model)
        self.compression_ratio = (
            baseline_total / stripped_nonzero if stripped_nonzero > 0 else 1.0
        )
        return self.compression_ratio

    def _save_metrics(self):
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"pruning_metrics_{timestamp}.csv"
        file_path = os.path.join(downloads_path, filename)

        system_info = [
            "\nSYSTEM INFORMATION",
            f"Processor: {self.system_info['processor']}",
            f"CPU Cores: {self.system_info['cpu_cores']}",
            f"CPU Threads: {self.system_info['cpu_threads']}",
            f"Total Memory: {self.system_info['total_memory']}",
            f"GPU Available: {self.system_info['gpu_available']}",
            f"GPU Name: {self.system_info['gpu_name']}",
        ]

        performance_info = [
            "\nMODEL PERFORMANCE",
            f"{'Model Type':<15} {'Accuracy':<12} {'Training Time':<18} {'Compression Ratio':<15}",
            f"{'Baseline':<15} {self.baseline_model_accuracy:,.4f}      {self.baseline_time:.2f}s            1.00x",
            f"{'Pruned':<15} {self.pruned_model_accuracy:,.4f}      {self.pruned_time:.2f}s            {self.compression_ratio:.2f}x",
        ]

        print("\n=== Model Pruning Results ===")
        for line in system_info + performance_info:
            print(line)
        print("\n" + "=" * 30)

        csv_data = [
            ["SYSTEM INFORMATION"],
            [f"Processor: {self.system_info['processor']}"],
            [f"CPU Cores: {self.system_info['cpu_cores']}"],
            [f"CPU Threads: {self.system_info['cpu_threads']}"],
            [f"Total Memory: {self.system_info['total_memory']}"],
            [f"GPU Available: {self.system_info['gpu_available']}"],
            [f"GPU Name: {self.system_info['gpu_name']}"],
            [],
            ["MODEL PERFORMANCE"],
            ["Model Type", "Accuracy", "Training Time (seconds)", "Compression Ratio"],
            [
                "Baseline",
                f"{self.baseline_model_accuracy:.4f}",
                f"{self.baseline_time:.2f}",
                "1.00x",
            ],
            [
                "Pruned",
                f"{self.pruned_model_accuracy:.4f}",
                f"{self.pruned_time:.2f}",
                f"{self.compression_ratio:.2f}x",
            ],
        ]

        try:
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            print(f"\nResults saved to: {file_path}")

        except Exception as e:
            print(f"\nError saving file: {e}")
            fallback_path = os.path.join(os.getcwd(), filename)
            try:
                with open(fallback_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(csv_data)
                print(f"Results saved to current directory: {fallback_path}")
            except Exception as e:
                print(f"Error saving to fallback location: {e}")

    def train_baseline_model(self, train_images, train_labels):
        start_time = time.time()
        self.model.fit(
            train_images,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
        )
        self.baseline_time = time.time() - start_time

    def evaluate_baseline(self, test_images, test_labels):
        _, self.baseline_model_accuracy = self.model.evaluate(
            test_images, test_labels, verbose=0
        )
        return self.baseline_model_accuracy

    def apply_pruning(self, train_images):
        if self.model is None:
            raise ValueError("Baseline model hasn't been trained yet!")

        num_images = len(train_images) * (1 - self.validation_split)
        end_step = np.ceil(num_images / self.batch_size).astype(np.int32) * self.epochs

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.50,
                final_sparsity=0.75,
                begin_step=0,
                end_step=end_step,
            )
        }

        self.pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model, **pruning_params
        )

        self.pruned_model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        return self.pruned_model

    def train_pruned_model(self, train_images, train_labels):
        if self.pruned_model is None:
            raise ValueError("Pruning hasn't been applied yet!")

        logdir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        start_time = time.time()
        self.pruned_model.fit(
            train_images,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
        )
        self.pruned_time = time.time() - start_time

    def strip_pruned_model(self):
        """Apply strip pruning to remove pruning wrappers and zero weights."""
        try:
            self.stripped_model = tfmot_strip_pruning(self.pruned_model)
            self.stripped_model.compile(
                optimizer="adam",
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
            return self.stripped_model
        except Exception as e:
            print(f"Warning: Error during strip pruning: {e}")
            print("Falling back to pruned model without stripping")
            self.stripped_model = self.pruned_model
            return self.stripped_model

    def evaluate_pruned_model(self, test_images, test_labels):
        if self.pruned_model is None:
            raise ValueError("Pruned model hasn't been trained yet!")

        self.stripped_model = self.strip_pruned_model()

        _, self.pruned_model_accuracy = self.stripped_model.evaluate(
            test_images, test_labels, verbose=0
        )

        self._calculate_compression_ratio()

        if (
            self.baseline_model_accuracy is not None
            and self.pruned_model_accuracy is not None
        ):
            self._save_metrics()

        return self.pruned_model_accuracy
