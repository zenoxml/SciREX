import numpy as np
import tensorflow as tf
from scirex.core.model_compression.pruning import ModelPruning

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize the input images and add the extra dimension (channel)
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.expand_dims(train_images, axis=-1)  # Shape (60000, 28, 28, 1)
test_images = np.expand_dims(test_images, axis=-1)  # Shape (10000, 28, 28, 1)

# Instantiate the ModelPruning class with default arguments
pruner = ModelPruning()

# Apply pruning to the model
pruned_model = pruner.apply_pruning()

# Train the pruned model
pruner.train_pruned_model(train_images, train_labels)

# Evaluate the pruned model
pruned_accuracy = pruner.evaluate_pruned_model(test_images, test_labels)
print(f"Pruned Model Accuracy: {pruned_accuracy * 100:.2f}%")
