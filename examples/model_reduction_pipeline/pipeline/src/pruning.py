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
    Example Script: pruning.py
    
    Model Pruning Implementation Module

This module provides functionality for model pruning optimization, including system metrics
collection and performance analysis. It supports pruning of TensorFlow/Keras models with
detailed performance tracking and metric reporting.

The module includes:
- Pruning technique implementation
- System information collection
- Performance metrics tracking
- CSV report generation

Attributes:
    TechniqueType (Enum): Available optimization techniques
    
Authors:
    - Nithyashree R (nithyashreer@iisc.ac.in).
"""



import tempfile
import os
import csv
import psutil
import platform
from datetime import datetime
import time
from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow_model_optimization.sparsity.keras import strip_pruning as tfmot_strip_pruning


class TechniqueType(Enum):
    """
    Enumeration of supported optimization techniques.

    Attributes:
        PRUNING (str): Weight pruning technique
    """
    PRUNING = "pruning"
    

class OptimizationTechnique:
    """
    Base class for optimization techniques.

    This class serves as an abstract base for different optimization techniques
    that can be applied to neural network models.

    :param params: Dictionary containing optimization parameters
    :type params: Dict[str, Any]
    """
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
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


class PruningTechnique(OptimizationTechnique):
    """
    Implementation of model pruning optimization technique.

    This class provides functionality for pruning a neural network model,
    including performance measurement and metric collection.

    Attributes:
        model (tf.keras.Model): Original model before pruning
        pruned_model (tf.keras.Model): Model during pruning process
        stripped_model (tf.keras.Model): Final pruned model with pruning wrappers removed
        baseline_model_accuracy (float): Accuracy of the original model
        pruned_model_accuracy (float): Accuracy of the pruned model
        baseline_time (float): Training time for the baseline model
        pruned_time (float): Training time for the pruned model
        compression_ratio (float): Achieved compression ratio after pruning
        system_info (dict): System hardware information
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the pruning technique.

        :param params: Dictionary containing pruning parameters
        :type params: Dict[str, Any]
        """
        super().__init__(params)
        self.model = None
        self.pruned_model = None
        self.stripped_model = None
        self.baseline_model_accuracy = None
        self.pruned_model_accuracy = None
        self.baseline_time = None
        self.pruned_time = None
        self.compression_ratio = None
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, str]:
        """
        Collect system hardware information.

        :return: Dictionary containing system information
        :rtype: Dict[str, str]
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

    def _calculate_compression_ratio(self) -> float:
        """
        Calculate compression ratio between baseline and pruned models.

        :return: Compression ratio achieved by pruning
        :rtype: float
        """
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
        self.compression_ratio = baseline_total / stripped_nonzero if stripped_nonzero > 0 else 1.0
        return self.compression_ratio

    def _save_metrics(self) -> None:
        """
        Save performance metrics to a CSV file.

        This method saves system information and model performance metrics
        to a CSV file in the user's Downloads directory or current working
        directory as fallback.

        :raises Exception: If there's an error saving the metrics file
        """
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
            ["Baseline", f"{self.baseline_model_accuracy:.4f}", f"{self.baseline_time:.2f}", "1.00x"],
            ["Pruned", f"{self.pruned_model_accuracy:.4f}", f"{self.pruned_time:.2f}", f"{self.compression_ratio:.2f}x"],
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

    def apply(self, model: tf.keras.Model, train_data: Tuple[tf.Tensor, tf.Tensor],
             test_data: Tuple[tf.Tensor, tf.Tensor]) -> tf.keras.Model:
        """
        Apply pruning optimization to the model.

        This method implements the complete pruning pipeline:
        1. Train and evaluate baseline model
        2. Apply pruning to the model
        3. Train and evaluate pruned model
        4. Strip pruning wrappers
        5. Calculate metrics and save results

        :param model: Model to be pruned
        :type model: tf.keras.Model
        :param train_data: Tuple of training data and labels
        :type train_data: Tuple[tf.Tensor, tf.Tensor]
        :param test_data: Tuple of test data and labels
        :type test_data: Tuple[tf.Tensor, tf.Tensor]
        :return: Optimized (pruned) model
        :rtype: tf.keras.Model
        """
        self.model = model
        train_images, train_labels = train_data
        test_images, test_labels = test_data
        
        # Train baseline model
        print("\nTraining baseline model...")
        start_time = time.time()
        self.model.fit(
            train_images,
            train_labels,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_split=self.params['validation_split']
        )
        self.baseline_time = time.time() - start_time

        # Evaluate baseline model
        _, self.baseline_model_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)

        # Apply pruning
        print("\nApplying pruning...")
        num_images = len(train_images) * (1 - self.params['validation_split'])
        end_step = np.ceil(num_images / self.params['batch_size']).astype(np.int32) * self.params['epochs']

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.50,
                final_sparsity=self.params['target_sparsity'],
                begin_step=0,
                end_step=end_step,
            )
        }

        self.pruned_model = tfmot.sparsity.keras.prune_low_magnitude(self.model, **pruning_params)
        self.pruned_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        # Train pruned model
        print("\nTraining pruned model...")
        logdir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        start_time = time.time()
        self.pruned_model.fit(
            train_images,
            train_labels,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_split=self.params['validation_split'],
            callbacks=callbacks
        )
        self.pruned_time = time.time() - start_time

        # Strip and evaluate pruned model
        print("\nEvaluating pruned model...")
        self.stripped_model = tfmot_strip_pruning(self.pruned_model)
        self.stripped_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        _, self.pruned_model_accuracy = self.stripped_model.evaluate(test_images, test_labels, verbose=0)
        self._calculate_compression_ratio()
        self._save_metrics()

        return self.stripped_model


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
        TechniqueType.PRUNING: PruningTechnique,
        # TechniqueType.QUANTIZATION: QuantizationTechnique,  # To be implemented
    }
    
    technique_class = technique_map.get(technique_type)
    if technique_class is None:
        raise ValueError(f"Unsupported technique type: {technique_type}")
    
    return technique_class(params)