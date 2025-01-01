import os
import numpy as np
import tensorflow as tf
from scirex.core.model_compression.pruning import ModelPruning


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the input images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Instantiate the ModelPruning class
pruner = ModelPruning()

# Train the baseline model
pruner.train_baseline_model(train_images, train_labels)

# Evaluate the baseline model
baseline_accuracy = pruner.evaluate_baseline(test_images, test_labels)
print("Baseline test accuracy:", baseline_accuracy)

# Save the baseline model
baseline_model_path = pruner.save_baseline_model()
print("Baseline model saved at:", baseline_model_path)
