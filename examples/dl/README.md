# SciREX Deep Learning Examples

This directory contains implementations of various deep learning models using the SciREX library. The examples demonstrate how to build, train, and evaluate neural networks on a dataset.

## File Structure

```
.
├── README.md
├── cnn_mnist.py  # Convolutional Neural Network implementation
├── gcn_cora.py   # Graph Convolutional Network implementation
└── vae_mnist.py  # Variational Autoencoder implementation
```

## Table of Contents

- [Installation](#installation)
- [Models](#models)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [Graph Convolution Network (GCN)](#graph-convolution-network-gcn)
  - [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Requirements](#requirements)

## Installation

```bash
# Install required packages
pip install jax jaxlib
pip install optax
pip install tensorflow  # For MNIST dataset
pip install torch, torch_geometric  # For CORA dataset
pip install matplotlib  # For plotting
# Install SciREX (assuming it's available on PyPI)
pip install scirex
```

## Models

### Convolutional Neural Network (CNN)

A classification model that takes MNIST digit images and predicts their corresponding digit class (0-9).

```python
from scirex.core.dl import Model, Network
import scirex.core.dl.nn as nn

class CNN(Network):
    def __init__(self):
        self.layers = [
            nn.Conv2d(1, 4, kernel_size=4),
            nn.MaxPool2d(2, 2),
            nn.relu,
            nn.Conv2d(4, 8, kernel_size=4),
            nn.MaxPool2d(2, 2),
            nn.relu,
            jnp.ravel,
            nn.Linear(8 * 4 * 4, 10),
            nn.log_softmax,
        ]
```

Features:
- 2 convolutional layers with max pooling
- Dense classification layer
- Cross-entropy loss
- Accuracy metric
- Adam optimizer

### Graph Convolution Neural Network (GCN)

A classification model that takes the citation network of scientific publications from the CORA dataset.
Find more explanation in `gcn_cora.py`.

### Variational Autoencoder (VAE)

An unsupervised learning model that learns to encode and decode MNIST digits through a compressed latent space.

```python
class VAE(Network):
    def __init__(self, encoderLayers, decoderLayers):
        self.encoder = FCNN(encoderLayers)
        self.decoder = FCNN(decoderLayers)

    def __call__(self, x):
        x = self.encoder(x)
        mean, stddev = x[:-1], jnp.exp(x[-1])
        z = mean + stddev * jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        return self.decoder(z)
```

Features:
- Encoder-decoder architecture
- Latent space sampling
- KL divergence loss
- Reconstruction capability
- Adam optimizer

## Usage

### Training a New Model

#### CNN (`cnn-mnist.py`)
```python
# Import required modules
from scirex.core.dl import Model, Network
from scirex.core.dl.nn.loss import cross_entropy_loss
from scirex.core.dl.nn.metrics import accuracy

# Create and train model
model = Model(CNN(), optax.adam(learning_rate), cross_entropy_loss, [accuracy])
history = model.fit(train_images, train_labels, num_epochs, batch_size)
model.save_net("mnist-cnn.dl")
```

#### VAE (`vae-mnist.py`)
```python
# Import required modules
from scirex.core.dl import Model, Network, FCNN

# Create and train model
model = Model(VAE(encoderLayers, decoderLayers), optax.adam(learning_rate), loss_fn)
model.fit(x_train, x_train, num_epochs=100, batch_size=64)
model.save_net("mnist_vae.dl")
```

### Loading a Pretrained Model

Both example files include model persistence:

#### CNN (`cnn-mnist.py`)
```python
if path.exists("mnist-cnn.dl"):
    print("Loading the model from mnist-cnn.dl")
    model.load_net("mnist-cnn.dl")
```

#### VAE (`vae-mnist.py`)
```python
if path.exists("mnist_vae.dl"):
    print("Loading model from disk")
    model.load_net("mnist_vae.dl")
```

### Evaluation

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc[0]:.4f}")
```

### Visualization

Each example includes visualization of training history:

#### CNN (`cnn-mnist.py`)
```python
model.plot_history("mnist-cnn.png")
```

#### VAE (`vae-mnist.py`)
```python
model.plot_history("mnist_vae.png")
```

## Model Architectures

### CNN Architecture
1. Input Layer (28x28x1)
2. Conv2D (4 filters, 4x4 kernel) → MaxPool2D (2x2) → ReLU
3. Conv2D (8 filters, 4x4 kernel) → MaxPool2D (2x2) → ReLU
4. Flatten
5. Dense (10 units) → LogSoftmax

### VAE Architecture
Encoder:
1. Conv2D (4 filters, 4x4 kernel) → MaxPool2D (2x2) → ReLU
2. Conv2D (8 filters, 4x4 kernel) → MaxPool2D (2x2) → ReLU
3. Flatten
4. Dense (5 units) → LogSoftmax

Decoder:
1. Dense (64 units) → ReLU
2. Dense (128 units) → ReLU
3. Dense (784 units)
4. Reshape to (28, 28)

## Training

Both models include:
- Automatic model checkpointing
- Training history plotting
- Performance metrics tracking
- Early stopping capability

### Hyperparameters

#### CNN (`cnn_mnist.py`)
```python
batch_size = 10
learning_rate = 0.001
num_epochs = 10
```

#### GCN (`cnn_mnist.py`)
```python
learning_rate = 0.05
num_epochs = 50
```

#### VAE (`vae_mnist.py`)
```python
batch_size = 64
learning_rate = 0.001
num_epochs = 100
```

## Requirements

- Python 3.9+
- JAX and JAX NumPy
- Optax
- TensorFlow (for MNIST dataset)
- PyTorch (for CORA dataset)
- SciREX
- Matplotlib
