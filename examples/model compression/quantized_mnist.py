import numpy as np
import tensorflow as tf
from scirex.core.model_compression.quantization import (
    QuantizationAwareTraining,
)  # Import the class from the specified module
import os

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize QuantizationAwareTraining class for the MNIST dataset
qat = QuantizationAwareTraining(input_shape=(28, 28), num_classes=10)

# Train the base model for 10 epochs
qat.train(train_images, train_labels, epochs=10)

# Apply Quantization-Aware Training
qat.apply_quantization_aware_training()

# Train the quantization-aware model for 10 epochs
qat.train_q_aware_model(train_images, train_labels, epochs=10)

# Evaluate the quantized model
q_aware_accuracy = qat.evaluate(test_images, test_labels)[
    1
]  # Get only the quantized model accuracy

print("Quantization-aware model accuracy:", q_aware_accuracy)

# Convert the quantization-aware model to TFLite format
quantized_tflite_model = qat.convert_to_tflite()

# Save the quantized TFLite model
tflite_model_filename = "quantized_mnist_model.tflite"
qat.save_model(quantized_tflite_model, tflite_model_filename)

# Measure and print the size of the quantized TFLite model
model_size = qat.measure_model_size(tflite_model_filename)
print(f"Quantized model size: {model_size:.2f} MB")

# Optionally, you can also apply post-training quantization
post_quantized_tflite_model = qat.post_quantization()

# Save the post-quantized model
post_quantized_filename = "post_quantized_mnist_model.tflite"
qat.save_model(post_quantized_tflite_model, post_quantized_filename)

# Measure and print the size of the post-quantized TFLite model
post_quantized_model_size = qat.measure_model_size(post_quantized_filename)
print(f"Post-quantized model size: {post_quantized_model_size:.2f} MB")
