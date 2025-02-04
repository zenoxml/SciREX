# Model Compression Using Pruning and Quantization
This repository demonstrates two popular model compression techniques - Pruning and Quantization are implemented on the MNIST dataset using TensorFlow.

## What is Model Compression?

Model compression is a technique used to reduce the size of neural networks while maintaining their performance. This is particularly important for deploying models on devices with limited resources like mobile phones or edge devices.

## Techniques Implemented 

## 1. Pruning
Network pruning involves systematically removing redundant or less important parameters (weights and connections) from a trained neural network while preserving its performance. It works by identifying and eliminating weights that have minimal impact on the model's output, effectively reducing the network's size and computational requirements without significantly degrading accuracy.

### Overview
This code implements a reusable `ModelPruning` class for neural network pruning using TensorFlow. It uses a polynomial decay schedule to gradually increase sparsity (the proportion of weights removed from the model) by removing weights over time, optimizing the model's size and speed without compromising accuracy, making it more efficient for inference tasks.
## Key Components

### Class: ModelPruning
A comprehensive class that handles the entire pruning workflow for TensorFlow models.

## Main Features

### 1. Model Initialization
- Creates a base CNN model with configurable input shape and number of classes
- Default configuration:
  - Input shape: (28, 28)
  - Number of classes: 10
  - Epochs: 10
  - Batch size: 35
  - Validation split: 0.1

### 2. Model Architecture
The default model consists of:
- Input layer
- Reshape layer
- Conv2D layer (12 filters, 3x3 kernel)
- MaxPooling2D layer
- Flatten layer
- Dense output layer

### 3. Key Methods

#### Training and Evaluation
- `train_baseline_model()`: Trains the initial model
- `evaluate_baseline()`: Evaluates the base model performance
- `save_baseline_model()`: Saves the model in .keras format

#### Pruning Operations
- `apply_pruning()`: Implements the pruning logic
  - Uses polynomial decay schedule
  - Initial sparsity: 50%
  - Final sparsity: 80%
- `train_pruned_model()`: Trains the pruned model
- `evaluate_pruned_model()`: Evaluates the pruned model's performance

### Example Usage

1. **Dataset Preparation**
   - Loads the MNIST dataset and normalizes the images.

2. **Model Pruning**
   - Initializes the `ModelPruning` class and applies pruning to the model.

3. **Model Training**
   - Trains the pruned model on the MNIST dataset.

4. **Model Evaluation**
   - Evaluates the pruned model on the test dataset and reports accuracy.


### Results
- The baseline model will show its initial accuracy
- The pruned model will show:
  - Final accuracy (usually close to baseline)
  - Reduction in model size

Typical output looks like:
```
Baseline Test Accuracy: ~0.9818

Pruned Model Accuracy: ~0.9513

```

## 2. Quantization
Quantization is like compressing a high-resolution image to a lower resolution while trying to maintain its quality. It reduces the precision of the numbers used in the model (e.g., from 32-bit to 8-bit).


### Overview
This code implements a reusable class `QuantizationAwareTraining` for performing quantization-aware training on neural networks using TensorFlow. The implementation focuses on model optimization through quantization, which helps reduce model size and improve inference speed while maintaining accuracy.

## Key Components
- Base model creation and training
- Quantization-aware training implementation
- Model evaluation capabilities
- TFLite conversion support
- Post-training quantization
- Model size measurement utilities


### Class: `QuantizationAwareTraining`
This class encapsulates all the functionality for building, training, applying quantization, and saving the models.

### Constructor Parameters

- `input_shape`: Shape of input data (e.g., `(28, 28)` for MNIST).
- `num_classes`: Number of classes for classification tasks.
- `filters`, `kernel_size`, `pool_size`: Configurations for the Conv2D and MaxPooling2D layers.

### Model Architecture

The base model consists of:
- Input Layer
- Reshape Layer
- Conv2D Layer with ReLU activation
- MaxPooling2D Layer
- Flatten Layer
- Dense Output Layer

### Key Methods

1. **`train(train_data, train_labels, epochs, validation_split)`**
   - Trains the base model with early stopping.
   - Uses the Adam optimizer and sparse categorical cross-entropy loss.

2. **`apply_quantization_aware_training()`**
   - Prepares the model for quantization-aware training by simulating lower precision during training.

3. **`train_q_aware_model(train_data, train_labels, batch_size, epochs, validation_split)`**
   - Trains the quantization-aware model with early stopping.

4. **`evaluate(test_data, test_labels)`**
   - Evaluates both the base and quantized models.
   - Returns accuracy for comparison.

5. **`convert_to_tflite()`**
   - Converts the quantized model to TensorFlow Lite format.
   - Applies optimizations to further reduce size.

6. **`post_quantization()`**
   - Implements post-training quantization for size reduction.
   - Outputs a quantized TFLite model.


### Example Usage

1. **Dataset Preparation**
   - Loads the MNIST dataset and normalizes images.

2. **Base Model Training**
   - Initializes the `QuantizationAwareTraining` class and trains the base model.
   - Measures the original model size.

3. **Quantization-Aware Training**
   - Applies QAT and trains the quantized model for additional epochs.
   - Evaluates accuracy after QAT.

4. **TFLite Conversion**
   - Converts the quantized model to TFLite format.
   - Compares model sizes before and after quantization.

5. **Post-Training Quantization**
   - Further reduces model size using post-training quantization.


### Results

- Original model size (in MB)
- QAT (Quantization-Aware Training) model size
- Post-training quantized model size
- Size reduction percentages
- Accuracy comparisons

Typical output looks like:
```
Baseline Model Accuracy: ~0.9814

Quantization-Aware Model Accuracy: ~98.33%
Original Model Size: ~0.08 MB
QAT Model Size: ~0.02 MB
Post-Training Quantized Size: ~0.02 MB
Size Reduction from QAT: ~70.7%
Size Reduction from Post-Training: ~71.7%

```


## 3. SVD (Singular Value Decomposition)

SVD (Singular Value Decomposition) is a matrix factorization technique that decomposes a matrix A into three components: A = U ∑ V^T, where U and V are orthogonal matrices containing left and right singular vectors, and ∑ is a diagonal matrix containing singular values. In simpler terms, it breaks down a complex matrix into simpler, meaningful components that capture the most important patterns in the data.

### matrix notation:

    A (m×n) = U (m×m) × Σ (m×n) × V^T (n×n)

- A is any m×n matrix 
- U and V are orthogonal matrices 
- Σ contains singular values in descending order.

### Overview
The code demonstrates matrix compression using Singular Value Decomposition (SVD) on a 50x40 matrix, testing different compression ranks while analyzing the trade-off between accuracy and size reduction.

## Key Components:

- Original Matrix: 50x40 matrix (2000 elements)
- SVD Components: U matrix, singular values (s), and Vt matrix
- Testing ranks: 5, 10, 15, and 20

## Main Features:

### Model Initialization:

- Creates a random 50x40 matrix
- Original size: 2000 elements
- Baseline accuracy: 1.0 (perfect)

### Model Architecture:
- Uses SVD decomposition with three components:

         - U matrix: Captures row patterns
         - s vector: Contains singular values
         - Vt matrix: Captures column patterns


## Key Methods:

- SVD decomposition using numpy.linalg.svd
- Matrix reconstruction from compressed components
- Accuracy calculation using normalized error
- Compression ratio calculation

### Results

Typical output looks like:
```
Original matrix shape: (50, 40)
Original matrix size: 2000 elements
Original matrix accuracy: 1.0000

Compression Details:
----------------------------------------------------------------------
Rank  5:
  - Matrix Shapes after SVD:
    * U matrix: (50, 5) = 250 elements
    * s vector: (5,) = 5 elements
    * Vt matrix: (5, 40) = 200 elements
  - Accuracy: 0.8281
  - Original Size: 2000 elements
  - Compressed Size: 455 elements
  - Compression Ratio: 0.23 (model is 22.8% of original size)
  - Size Reduction: 77.2%
----------------------------------------------------------------------
Rank 10:
  - Matrix Shapes after SVD:
    * U matrix: (50, 10) = 500 elements
    * s vector: (10,) = 10 elements
    * Vt matrix: (10, 40) = 400 elements
  - Accuracy: 0.8921
  - Original Size: 2000 elements
  - Compressed Size: 910 elements
  - Compression Ratio: 0.46 (model is 45.5% of original size)
  - Size Reduction: 54.5%
----------------------------------------------------------------------
Rank 15:
  - Matrix Shapes after SVD:
    * U matrix: (50, 15) = 750 elements
    * s vector: (15,) = 15 elements
    * Vt matrix: (15, 40) = 600 elements
  - Accuracy: 0.9346
  - Original Size: 2000 elements
  - Compressed Size: 1365 elements
  - Compression Ratio: 0.68 (model is 68.2% of original size)
  - Size Reduction: 31.8%
----------------------------------------------------------------------
Rank 20:
  - Matrix Shapes after SVD:
    * U matrix: (50, 20) = 1000 elements
    * s vector: (20,) = 20 elements
    * Vt matrix: (20, 40) = 800 elements
  - Accuracy: 0.9635
  - Original Size: 2000 elements
  - Compressed Size: 1820 elements
  - Compression Ratio: 0.91 (model is 91.0% of original size)
  - Size Reduction: 9.0%
----------------------------------------------------------------------

```
- A visual graph will be displayed comparing the accuracy of compressed vs original matrix across different ranks, helping visualize the compression-accuracy trade-off.

## Running the Code

To run any of the examples, use the following command pattern:

```bash
pip install tensorflow-model-optimization
```

```bash
PYTHONPATH=$(pwd) python3 "examples/model compression/<filename>"
```

For example:
```bash
# For pruning
PYTHONPATH=$(pwd) python3 "examples/model compression/pruned_base_mnist.py"
PYTHONPATH=$(pwd) python3 "examples/model compression/pruned_mnist.py"

# For quantization
PYTHONPATH=$(pwd) python3 "examples/model compression/quantized_base_mnist.py"
PYTHONPATH=$(pwd) python3 "examples/model compression/quantized_mnist.py"

# For SVD
PYTHONPATH=$(pwd) python3 "examples/model compression/tensor_svd.py"
```

## Dependencies
- TensorFlow
- NumPy
- matplotlib

