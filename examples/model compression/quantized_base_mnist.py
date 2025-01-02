import numpy as np
import tensorflow as tf
from scirex.core.model_compression.quantization import QuantizationAwareTraining
import os

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Initialize QuantizationAwareTraining class for the MNIST dataset
qat = QuantizationAwareTraining(input_shape=(28, 28), num_classes=10)

# Ensure that the model is properly initialized
if qat.model is None:
    print("Initializing the model...")
    qat.initialize_model()  

# Train the base model for 10 epochs
qat.train(train_images, train_labels, epochs=10)

# Evaluate the base model (before quantization)
base_accuracy = qat.model.evaluate(test_images, test_labels)  

# Print base model accuracy
print("Base model accuracy:", base_accuracy[1])  
