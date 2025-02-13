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
This module provides a comprehensive framework for compressing deep learning models
through various techniques such as pruning and quantization. It is designed to be
extensible and support different model architectures.

The framework currently supports:
    - Model compression techniques: pruning and quantization
    - Dataset: MNIST
    - Dynamic model loading and initialization
    - Configurable compression parameters

Key Components:
    - ModelCompressor: Main class handling the compression pipeline
    - Dataset management through DatasetLoader
    - Compression techniques through pruning and quantization modules
    - Command-line interface for easy usage

"""
import os
import warnings
import logging
import argparse
import tensorflow as tf
import importlib.util
from typing import Dict, Any, Tuple

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Import modules
from datasets import DatasetLoader, DatasetType
import pruning
import quantization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelCompressor:
    """
    A comprehensive framework for neural network model compression.
    
    This class provides functionality to load models dynamically and apply
    various compression techniques such as pruning and quantization. It handles
    the complete pipeline from model loading to compression application.
    
    Attributes:
        technique (str): Compression technique to be applied ('pruning' or 'quantization')
        model_path (str): Path to the model definition file
        train_data (tf.data.Dataset): Training dataset
        test_data (tf.data.Dataset): Testing dataset
        model (tf.keras.Model): Loaded neural network model
        compressed_model (tf.keras.Model): Compressed version of the model
        dataset_info (DatasetInfo): Information about the dataset being used
    
    """

    def __init__(
        self,
        technique: str,
        model_path: str = "models.py",
    ) -> None:
        """
        Initialize the ModelCompressor with specified parameters.
        
        Args:
            technique (str): Compression technique to apply ('pruning' or 'quantization')
            model_path (str, optional): Path to model definition file. Defaults to "models.py"
        """
        self.technique = technique
        self.model_path = model_path
        
        # Dynamically import the model module
        try:
            spec = importlib.util.spec_from_file_location("model_module", self.model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            self.get_model = model_module.get_model
            self.ModelType = model_module.ModelType
            logger.info(f"Successfully loaded model module from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model module: {str(e)}")
            raise
        
        # Load MNIST dataset
        (self.train_data, self.test_data) = DatasetLoader.load_dataset(DatasetType.MNIST)
        logger.info("Successfully loaded MNIST dataset")
        
        # Get dataset info
        self.dataset_info = DatasetLoader.get_dataset_info(DatasetType.MNIST)
        
        # Initialize model
        self.model = self._load_model()
        logger.info("Successfully initialized model")

    def _load_model(self) -> tf.keras.Model:
        """
        Load and initialize the neural network model.
        
        This method uses the dynamically imported model factory function to create
        a model instance with appropriate input shape and number of classes based
        on the dataset information.
        
        Returns:
            tf.keras.Model: Initialized neural network model

        """
        try:
            # Add channel dimension to input shape
            input_shape = (*self.dataset_info.input_shape, 1)
            
            # Create model using dynamically imported get_model
            model = self.get_model(
                model_type=self.ModelType.SIMPLE_CNN,
                input_shape=input_shape,
                num_classes=self.dataset_info.num_classes
            )
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def compress(self) -> None:
        """
        Apply the selected compression technique to the model.
        
        This method configures and applies either pruning or quantization based on
        the specified technique. It handles the complete compression process including
        parameter configuration and technique application.
        
        Compression parameters:
            - batch_size: 32
            - epochs: 10
            - validation_split: 0.2
            - target_sparsity: 0.75 (for pruning)
        """
        logger.info(f"Starting {self.technique} compression...")
        
        # Define optimization parameters
        params = {
            'batch_size': 32,
            'epochs': 10,
            'validation_split': 0.2,
            'target_sparsity': 0.75  # for pruning
        }
        
        try:
            if self.technique.lower() == "pruning":
                technique = pruning.get_technique(
                    pruning.TechniqueType.PRUNING,
                    params
                )
            elif self.technique.lower() == "quantization":
                technique = quantization.get_technique(
                    quantization.TechniqueType.QUANTIZATION,
                    params
                )
            else:
                raise ValueError(f"Unknown technique: {self.technique}")
            
            # Apply compression technique
            self.compressed_model = technique.apply(
                model=self.model,
                train_data=self.train_data,
                test_data=self.test_data
            )
            logger.info(f"Successfully applied {self.technique}")
            
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the compression framework.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - technique: Compression technique to apply ('pruning' or 'quantization')
            - model: Path to model definition file
    """
    parser = argparse.ArgumentParser(description='Model Compression Framework')
    parser.add_argument('--technique', type=str, required=True, 
                      choices=['pruning', 'quantization'],
                      help='Compression technique to apply')
    parser.add_argument('--model', type=str, 
                      default="models.py",
                      help='Path to the model file')
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the model compression framework.
    
    This function orchestrates the complete compression pipeline:
        1. Parses command line arguments
        2. Initializes the ModelCompressor
        3. Applies the selected compression technique
        4. Handles any errors that occur during execution
    """
    try:
        args = parse_args()
        logger.info(f"Starting compression with technique: {args.technique}")
        
        # Initialize compressor with the specified model path
        compressor = ModelCompressor(
            technique=args.technique,
            model_path=args.model
        )
        
        # Apply compression
        compressor.compress()
        logger.info("Compression completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    
    