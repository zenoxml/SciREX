# Model Compression Using Pruning and Quantization
This repository demonstrates two popular model compression techniques - Pruning and Quantization - implemented on the MNIST dataset using TensorFlow.

## What is Model Compression?

Model compression is a technique used to reduce the size of neural networks while maintaining their performance. This is particularly important for deploying models on devices with limited resources like mobile phones or edge devices.

## Techniques Implemented 

### 1. Pruning
Network pruning involves systematically removing redundant or less important parameters (weights and connections) from a trained neural network while preserving its performance. It works by identifying and eliminating weights that have minimal impact on the model's output, effectively reducing the network's size and computational requirements without significantly degrading accuracy.

#### How Pruning Works in Our Code:
- `pruned_base_mnist.py`: Creates and trains the baseline model
  - Loads and normalizes MNIST dataset
  - Trains the original model
  - Saves the baseline model for comparison

- `pruned_mnist.py`: Implements the pruning
  - Takes the trained model
  - Removes less important connections
  - Retrains the pruned model to maintain accuracy
  - Reports the final accuracy

### 2. Quantization
Quantization is like compressing a high-resolution image to a lower resolution while trying to maintain its quality. It reduces the precision of the numbers used in the model (e.g., from 32-bit to 8-bit).

#### How Quantization Works in Our Code:
- `quantized_base_mnist.py`: Prepares the base model
  - Sets up the model architecture
  - Trains the initial model
  - Evaluates baseline performance

- `quantized_mnist.py`: Implements quantization
  - Trains the base model
  - Applies quantization-aware training
  - Performs post-training quantization
  - Compares model sizes and accuracies

## Running the Code

To run any of the examples, use the following command pattern:
```bash
PYTHONPATH=$(pwd) python3 examples/model_compression/<filename>
```

For example:
```bash
# For pruning
PYTHONPATH=$(pwd) python3 examples/model_compression/pruned_base_mnist.py
PYTHONPATH=$(pwd) python3 examples/model_compression/pruned_mnist.py

# For quantization
PYTHONPATH=$(pwd) python3 examples/model_compression/quantized_base_mnist.py
PYTHONPATH=$(pwd) python3 examples/model_compression/quantized_mnist.py
```

## Expected Results

### Pruning Results
- The baseline model will show its initial accuracy
- The pruned model will show:
- Final accuracy (usually close to baseline)
- Reduction in model size

Typical output looks like:
```
Baseline Test Accuracy: ~0.9818

Pruned Model Accuracy: ~0.9513

```

### Quantization Results

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

## Dependencies
- TensorFlow
- NumPy


## Tips for Best Results
1. Always run the base model first before running the compressed version
2. Make sure your PYTHONPATH is set correctly
3. Keep track of the original model's performance for comparison
4. Monitor both accuracy and size reduction to ensure optimal compression


## Understanding the Results
A successful compression should:
- Maintain accuracy within 1-2% of the original model
- Significantly reduce model size (50-80% reduction is common)
- Keep the inference time similar or improved