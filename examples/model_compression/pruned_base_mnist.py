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

# Initialize ModelPruning
model_handler = ModelPruning(
    input_shape=(28, 28),
    num_classes=10,
    epochs=10,
    batch_size=35,  # This will give ~1688 steps per epoch
    validation_split=0.1,
)

# Train the baseline model
model_handler.train_baseline_model(train_images, train_labels)

# Evaluate baseline model
baseline_accuracy = model_handler.evaluate_baseline(test_images, test_labels)
print(f"Baseline test accuracy: {baseline_accuracy}")

# Save the baseline model
model_path = model_handler.save_baseline_model()
print(f"Baseline model saved at: {model_path}")
