import numpy as np
import tensorflow as tf
from scirex.core.model_compression.pruning import ModelPruning

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize the input images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize ModelPruning using the same configuration as baseline
model_handler = ModelPruning(
    input_shape=(28, 28),
    num_classes=10,
    epochs=10,
    batch_size=35,  # This will give ~1688 steps per epoch
    validation_split=0.1,
)

# Apply pruning to the model
pruned_model = model_handler.apply_pruning()

# Train the pruned model
model_handler.train_pruned_model(train_images, train_labels)

# Evaluate the pruned model
pruned_accuracy = model_handler.evaluate_pruned_model(test_images, test_labels)
print(f"Pruned Model Accuracy: {pruned_accuracy}")
