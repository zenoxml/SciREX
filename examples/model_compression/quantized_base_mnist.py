import numpy as np
import tensorflow as tf
from scirex.core.model_compression.quantization import QuantizationAwareTraining

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize QAT with model architecture
qat = QuantizationAwareTraining(input_shape=(28, 28), num_classes=10)

# Train base model
print("Training base model...")
qat.train(train_images, train_labels, epochs=10)

# Evaluate base model directly
baseline_accuracy = qat.model.evaluate(test_images, test_labels, verbose=0)[
    1
]  # Gets accuracy
print(f"\nBaseline Model Accuracy: {baseline_accuracy:.4f}")
