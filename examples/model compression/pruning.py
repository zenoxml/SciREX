import tempfile
import os
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
    ]
)

# Train the digit classification model
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_images,
    train_labels,
    epochs=4,
    validation_split=0.1,
)

# Evaluate baseline test accuracy and save the model for later usage.
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

print("Baseline test accuracy:", baseline_model_accuracy)

_, keras_file = tempfile.mkstemp(".h5")
keras.models.save_model(model, keras_file, include_optimizer=False)
print("Saved baseline model to:", keras_file)

# Fine-tune pre-trained model with pruning
# Define pruning parameters
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1  # 10% of training set will be used for validation set.
num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define pruning schedule.
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step
    )
}

# Apply pruning to the model
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Recompile the model
model_for_pruning.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model_for_pruning.summary()

# Training with pruning
logdir = tempfile.mkdtemp()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    callbacks=callbacks,
)

# Evaluate pruned model
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
    test_images, test_labels, verbose=0
)

print("Baseline test accuracy:", baseline_model_accuracy)
print("Pruned test accuracy:", model_for_pruning_accuracy)
