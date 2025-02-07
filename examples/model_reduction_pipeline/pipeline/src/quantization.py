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
# http://www.apache.org/licenses/LICENSE-2.0
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
Model Quantization Implementation Module

This module provides functionality for model quantization optimization, including system metrics
collection and performance analysis. It supports quantization-aware training (QAT) of 
TensorFlow/Keras models with detailed performance tracking and metric reporting.

The module includes:
- Quantization-aware training implementation
- System information collection
- Performance metrics tracking
- TFLite model conversion
- CSV report generation

Attributes:
    TechniqueType (Enum): Available optimization techniques
    
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
from enum import Enum
from typing import Dict, Any, Tuple


class TechniqueType(Enum):
    """
    Enumeration of supported optimization techniques.

    Attributes:
        QUANTIZATION (str): Model quantization technique
    """
    QUANTIZATION = "quantization"


class OptimizationTechnique:
    """
    Base class for optimization techniques.

    This class serves as an abstract base for different optimization techniques
    that can be applied to neural network models.

    Attributes:
        params (Dict[str, Any]): Parameters for the optimization technique
        system_info (Dict[str, Any]): System hardware information
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the optimization technique.

        :param params: Dictionary containing optimization parameters
        :type params: Dict[str, Any]
        """
        self.params = params
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Collect system hardware information.

        :return: Dictionary containing system information
        :rtype: Dict[str, Any]
        """
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
        
    def apply(self, model: tf.keras.Model, train_data: Tuple[tf.Tensor, tf.Tensor],
             test_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.keras.Model:
        """
        Apply the optimization technique to the model.

        :param model: The model to optimize
        :type model: tf.keras.Model
        :param train_data: Tuple of training data and labels
        :type train_data: Tuple[tf.Tensor, tf.Tensor]
        :param test_data: Tuple of test data and labels
        :type test_data: Tuple[tf.Tensor, tf.Tensor]
        :return: Optimized model
        :rtype: tf.keras.Model
        :raises NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement apply()")


class QuantizationTechnique(OptimizationTechnique):
    """
    Implementation of model quantization optimization technique.

    This class provides functionality for quantizing a neural network model,
    including quantization-aware training and TFLite conversion.

    Attributes:
        model (tf.keras.Model): Original model before quantization
        q_aware_model (tf.keras.Model): Quantization-aware trained model
        baseline_accuracy (float): Accuracy of the original model
        q_aware_accuracy (float): Accuracy of the quantized model
        baseline_time (float): Training time for the baseline model
        q_aware_time (float): Training time for the quantized model
        model_sizes (Dict[str, float]): Sizes of TFLite models in MB
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the quantization technique.

        :param params: Dictionary containing quantization parameters
        :type params: Dict[str, Any]
        """
        super().__init__(params)
        self.model = None
        self.q_aware_model = None
        self.baseline_accuracy = None
        self.q_aware_accuracy = None
        self.baseline_time = None
        self.q_aware_time = None
        self.model_sizes = {}

    def _save_metrics(self) -> None:
        """
        Save performance metrics to a CSV file.

        This method saves system information and model performance metrics
        to a CSV file in the user's Downloads directory.
        
        :raises Exception: If there's an error saving the metrics file
        """
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"quantization_metrics_{timestamp}.csv"
        file_path = os.path.join(downloads_dir, filename)

        # Calculate compression ratio
        baseline_size = self.model_sizes.get("baseline_model.tflite", 0)
        quantized_size = self.model_sizes.get("quantized_model.tflite", 0)
        compression_ratio = baseline_size / quantized_size if quantized_size > 0 else 0

        # Print results to terminal
        print("\n=== Model Quantization Results ===")
        print("SYSTEM INFORMATION")
        print(f"Processor: {self.system_info['processor']}")
        print(f"CPU Cores: {self.system_info['cpu_cores']}")
        print(f"CPU Threads: {self.system_info['cpu_threads']}")
        print(f"Total Memory: {self.system_info['total_memory']}")
        print(f"GPU Available: {self.system_info['gpu_available']}")
        print(f"GPU Name: {self.system_info['gpu_name']}")
        print("\nMODEL PERFORMANCE")
        print(f"{'Model Type':<15} {'Accuracy':<12} {'Training Time':<15} {'Compression Ratio'}")
        print(f"{'Baseline':<15} {self.baseline_accuracy:,.4f}      {self.baseline_time:.2f}s            1.00x")
        print(f"{'Quantized':<15} {self.q_aware_accuracy:,.4f}      {self.q_aware_time:.2f}s            {compression_ratio:.2f}x")
        print("=" * 30)

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
            ["Baseline", f"{self.baseline_accuracy:.4f}", f"{self.baseline_time:.2f}s", "1.00x"],
            ["Quantized", f"{self.q_aware_accuracy:.4f}", f"{self.q_aware_time:.2f}s", f"{compression_ratio:.2f}x"],
        ]

        try:
            os.makedirs(downloads_dir, exist_ok=True)
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
            print(f"\nResults saved to: {file_path}")
        except Exception as e:
            print(f"\nError saving file: {e}")

    def convert_to_tflite(self) -> Tuple[bytes, bytes]:
        """
        Convert models to TFLite format and measure sizes.

        :return: Tuple of (baseline_tflite_model, quantized_tflite_model)
        :rtype: Tuple[bytes, bytes]
        """
        # Convert baseline model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        baseline_tflite = converter.convert()

        # Convert quantized model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite = converter.convert()

        # Calculate sizes
        self.model_sizes["baseline_model.tflite"] = len(baseline_tflite) / float(2**20)  # Size in MB
        self.model_sizes["quantized_model.tflite"] = len(quantized_tflite) / float(2**20)  # Size in MB

        return baseline_tflite, quantized_tflite

    def apply(self, model: tf.keras.Model, train_data: Tuple[tf.Tensor, tf.Tensor],
             test_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.keras.Model:
        """
        Apply quantization optimization to the model.

        This method implements the complete quantization pipeline:
        1. Train and evaluate baseline model
        2. Apply quantization-aware training
        3. Train and evaluate quantized model
        4. Convert models to TFLite format
        5. Calculate metrics and save results

        :param model: Model to be quantized
        :type model: tf.keras.Model
        :param train_data: Tuple of training data and labels
        :type train_data: Tuple[tf.Tensor, tf.Tensor]
        :param test_data: Tuple of test data and labels
        :type test_data: Tuple[tf.Tensor, tf.Tensor]
        :return: Quantization-aware trained model
        :rtype: tf.keras.Model
        """
        self.model = model
        train_images, train_labels = train_data
        test_images, test_labels = test_data
        
        # Train baseline model
        print("\nTraining baseline model...")
        start_time = time.time()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=3, 
                restore_best_weights=True
            )
        ]
        
        self.model.fit(
            train_images,
            train_labels,
            batch_size=self.params['batch_size'],
            epochs=10,  # Fixed 10 epochs for baseline
            validation_split=self.params['validation_split'],
            callbacks=callbacks
        )
        self.baseline_time = time.time() - start_time

        # Evaluate baseline model
        _, self.baseline_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)

        # Apply quantization-aware training
        print("\nApplying quantization...")
        self.q_aware_model = tfmot.quantization.keras.quantize_model(self.model)
        self.q_aware_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Train quantized model
        print("\nTraining quantized model...")
        start_time = time.time()
        self.q_aware_model.fit(
            train_images,
            train_labels,
            batch_size=self.params['batch_size'],
            epochs=4,  # Fixed 4 epochs for QAT
            validation_split=self.params['validation_split'],
            callbacks=callbacks
        )
        self.q_aware_time = time.time() - start_time

        # Evaluate quantized model
        _, self.q_aware_accuracy = self.q_aware_model.evaluate(test_images, test_labels, verbose=0)

        # Convert to TFLite and measure sizes
        print("\nConverting models to TFLite...")
        self.convert_to_tflite()
        
        # Save metrics
        self._save_metrics()

        return self.q_aware_model


def get_technique(technique_type: TechniqueType, params: Dict[str, Any]) -> OptimizationTechnique:
    """
    Factory function to get the specified optimization technique.

    :param technique_type: Type of optimization technique to create
    :type technique_type: TechniqueType
    :param params: Parameters for the optimization technique
    :type params: Dict[str, Any]
    :return: Instance of the specified optimization technique
    :rtype: OptimizationTechnique
    :raises ValueError: If the specified technique type is not supported
    """
    technique_map = {
        TechniqueType.QUANTIZATION: QuantizationTechnique,
    }
    
    technique_class = technique_map.get(technique_type)
    if technique_class is None:
        raise ValueError(f"Unsupported technique type: {technique_type}")
    
    return technique_class(params)