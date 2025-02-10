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
Example Script: main.py

A reusable framework for performing model optimization through pruning and quantization.

This module provides a flexible framework for applying various optimization techniques
to different neural network architectures. The framework supports multiple model types,
datasets, and optimization strategies.

Attributes:
    OptimizationTechnique (Enum): Available optimization techniques (pruning, quantization)
    ModelArchitecture (Enum): Supported model architecture families (base, cnn)
 
Authors:
 - Nithyashree R (nithyashreer@iisc.ac.in).

"""


import argparse
from enum import Enum
from .models import ModelType as BaseModelType, get_model as get_base_model
from .model_cnn import ModelType as CNNModelType, get_model as get_cnn_model
from .datasets import DatasetType, DatasetLoader
from .pruning import TechniqueType as PruningType, get_technique as get_pruning_technique
from .quantization import TechniqueType as QuantizationType, get_technique as get_quantization_technique


class OptimizationTechnique(Enum):
    """
    Available optimization techniques for model compression.

    Attributes:
        PRUNING (str): Weight pruning technique to reduce model size
        QUANTIZATION (str): Weight quantization technique to reduce model precision
    """
    PRUNING = "pruning"
    QUANTIZATION = "quantization"


class ModelArchitecture(Enum):
    """
    Supported model architecture families.

    Attributes:
        BASE (str): Basic neural network architectures
        CNN (str): Convolutional neural network architectures
    """
    BASE = "base"
    CNN = "cnn"


def get_model_type(arch_type: ModelArchitecture, model_name: str) -> BaseModelType | CNNModelType:
    """
    Maps a model name to the correct ModelType based on architecture.

    :param arch_type: Architecture family to use
    :type arch_type: ModelArchitecture
    :param model_name: Name of the specific model within the architecture
    :type model_name: str
    :return: Appropriate ModelType enum instance
    :rtype: Union[BaseModelType, CNNModelType]
    :raises ValueError: If architecture type is not recognized
    """
    if arch_type == ModelArchitecture.BASE:
        return BaseModelType(model_name)
    elif arch_type == ModelArchitecture.CNN:
        return CNNModelType(model_name)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def get_model_instance(arch_type: ModelArchitecture, model_type: BaseModelType | CNNModelType,
                      input_shape: tuple, num_classes: int):
    """
    Creates a model instance of the specified architecture.

    :param arch_type: Architecture family to use
    :type arch_type: ModelArchitecture
    :param model_type: Specific model type within the architecture
    :type model_type: Union[BaseModelType, CNNModelType]
    :param input_shape: Shape of the input tensor
    :type input_shape: tuple
    :param num_classes: Number of output classes
    :type num_classes: int
    :return: Instantiated model of the specified type
    :rtype: tf.keras.Model
    :raises ValueError: If architecture type is not recognized
    """
    if arch_type == ModelArchitecture.BASE:
        return get_base_model(model_type, input_shape, num_classes)
    elif arch_type == ModelArchitecture.CNN:
        return get_cnn_model(model_type, input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


def main():
    """
    Main entry point for the model optimization framework.

    This function handles the complete optimization pipeline:
    1. Command line argument parsing
    2. Dataset loading
    3. Model creation
    4. Optimization technique application

    Command-line Arguments:
        --arch (str): Model architecture family (base or cnn)
        --model (str): Specific model within the architecture
        --dataset (str): Dataset to use for training/testing
        --technique (str): Optimization technique to apply
        --epochs (int): Number of training epochs
        --batch-size (int): Training batch size
        --validation-split (float): Validation data ratio
        --target-sparsity (float): Target sparsity for pruning

    :raises ValueError: If invalid arguments are provided
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Model Optimization Framework")
    
    # Model selection arguments
    parser.add_argument("--arch", type=str, choices=[a.value for a in ModelArchitecture],
                       default=ModelArchitecture.BASE.value, help="Model architecture family")
    parser.add_argument("--model", type=str, help="Specific model within the architecture")
    parser.add_argument("--dataset", type=str, choices=[d.value for d in DatasetType],
                       default=DatasetType.MNIST.value, help="Dataset to use")
    parser.add_argument("--technique", type=str, choices=[t.value for t in OptimizationTechnique],
                       default=OptimizationTechnique.PRUNING.value, help="Optimization technique to apply")
    
    # Common parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=35, help="Training batch size")
    parser.add_argument("--validation-split", type=float, default=0.1,
                       help="Validation split ratio")
    
    # Pruning-specific parameters
    parser.add_argument("--target-sparsity", type=float, default=0.75,
                       help="Target sparsity for pruning (only used with pruning)")
    
    args = parser.parse_args()
    
    # Convert string arguments to enum types
    arch_type = ModelArchitecture(args.arch)
    dataset_type = DatasetType(args.dataset)
    technique_type = OptimizationTechnique(args.technique)
    model_type = get_model_type(arch_type, args.model)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    dataset_info = DatasetLoader.get_dataset_info(dataset_type)
    train_data, test_data = DatasetLoader.load_dataset(dataset_type)
    
    # Create model
    print(f"\nCreating {args.model} model from {args.arch} architecture...")
    model = get_model_instance(arch_type, model_type, dataset_info.input_shape, dataset_info.num_classes)
    
    # Prepare common optimization parameters
    optimization_params = {
        "input_shape": dataset_info.input_shape,
        "num_classes": dataset_info.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_split": args.validation_split
    }
    
    # Apply optimization technique
    print(f"\nApplying {args.technique} optimization...")
    if technique_type == OptimizationTechnique.PRUNING:
        optimization_params["target_sparsity"] = args.target_sparsity
        technique = get_pruning_technique(PruningType.PRUNING, optimization_params)
    else:  # QUANTIZATION
        # Note: Quantization typically requires larger batch sizes for better results
        if args.batch_size == 35:  
            optimization_params["batch_size"] = 500  
        technique = get_quantization_technique(QuantizationType.QUANTIZATION, optimization_params)
    
    optimized_model = technique.apply(model, train_data, test_data)
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()