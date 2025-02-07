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
Datasets Module: datasets.py
    
MNIST Dataset Loading Module

This module handles the loading and preprocessing of MNIST dataset. 

- Loading raw MNIST data files via TensorFlow keras.datasets
- Normalizing pixel values to [0,1] range
- Storing dataset metadata (28x28 input shape, 10 classes)
- Interface for dataset loading with consistent preprocessing

Attributes:
    DatasetType (Enum): Available dataset types
    
Authors:
    - Nithyashree R (nithyashreer@iisc.ac.in)
"""

from enum import Enum
import tensorflow as tf
from typing import Tuple, Dict


class DatasetType(Enum):
    """
    Enumeration of supported dataset types.

    Attributes:
        MNIST (str): Standard MNIST digits dataset
        FASHION_MNIST (str): Fashion MNIST clothing dataset
    """
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    # Add more datasets here


class DatasetInfo:
    """
    Class to store dataset metadata.

    This class maintains information about dataset characteristics
    such as input shape and number of classes.

    Attributes:
        input_shape (Tuple[int, ...]): Shape of input samples
        num_classes (int): Number of classification classes
    """
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int):
        """
        Initialize dataset metadata.

        :param input_shape: Shape of input samples
        :type input_shape: Tuple[int, ...]
        :param num_classes: Number of classification classes
        :type num_classes: int
        """
        self.input_shape = input_shape
        self.num_classes = num_classes


class DatasetLoader:
    """
    Handles dataset loading and preprocessing.

    This class provides static methods for loading and preprocessing
    different datasets with a consistent interface. It includes:
    - Dataset information retrieval
    - Data loading
    - Standardized preprocessing
    """

    @staticmethod
    def get_dataset_info(dataset_type: DatasetType) -> DatasetInfo:
        """
        Get metadata for the specified dataset.

        :param dataset_type: Type of dataset to get information for
        :type dataset_type: DatasetType
        :return: Object containing dataset metadata
        :rtype: DatasetInfo
        :raises KeyError: If dataset type is not supported
        """
        info_map = {
            DatasetType.MNIST: DatasetInfo((28, 28), 10),
            DatasetType.FASHION_MNIST: DatasetInfo((28, 28), 10)
        }
        return info_map[dataset_type]

    @staticmethod
    def load_mnist() -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        Load and preprocess MNIST dataset.

        :return: Tuple of (train_data, test_data), where each is a tuple of (images, labels)
        :rtype: Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
        """
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        return DatasetLoader._preprocess_data(train_images, train_labels, test_images, test_labels)

    @staticmethod
    def load_fashion_mnist() -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        Load and preprocess Fashion MNIST dataset.

        :return: Tuple of (train_data, test_data), where each is a tuple of (images, labels)
        :rtype: Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
        """
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        return DatasetLoader._preprocess_data(train_images, train_labels, test_images, test_labels)

    @staticmethod
    def _preprocess_data(train_images: tf.Tensor, train_labels: tf.Tensor,
                       test_images: tf.Tensor, test_labels: tf.Tensor
                       ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        Preprocess the dataset by normalizing pixel values.

        This method performs standard preprocessing:
        - Normalizes pixel values to [0, 1] range by dividing by 255

        :param train_images: Training images
        :type train_images: tf.Tensor
        :param train_labels: Training labels
        :type train_labels: tf.Tensor
        :param test_images: Test images
        :type test_images: tf.Tensor
        :param test_labels: Test labels
        :type test_labels: tf.Tensor
        :return: Tuple of processed (train_data, test_data), where each is a tuple of (images, labels)
        :rtype: Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
        """
        # Normalize pixel values
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        return (train_images, train_labels), (test_images, test_labels)

    @classmethod
    def load_dataset(cls, dataset_type: DatasetType) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        Load the specified dataset using the appropriate loader.

        This method uses a factory pattern to select and execute the
        appropriate dataset loading function based on the dataset type.

        :param dataset_type: Type of dataset to load
        :type dataset_type: DatasetType
        :return: Tuple of (train_data, test_data), where each is a tuple of (images, labels)
        :rtype: Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
        :raises ValueError: If the specified dataset type is not supported
        """
        dataset_map = {
            DatasetType.MNIST: cls.load_mnist,
            DatasetType.FASHION_MNIST: cls.load_fashion_mnist
        }
        loader = dataset_map.get(dataset_type)
        if loader is None:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        return loader()