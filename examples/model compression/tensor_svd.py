# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

"""
Example Script: tensor_svd.py
    Implements matrix compression using SVD (Singular Value Decomposition) to demonstrate 
    dimensionality reduction. Compresses a 50x40 matrix using different ranks and 
    analyzes compression ratio and accuracy.

    Features:
        - SVD-based matrix compression
        - Compression ratio analysis
        - Accuracy visualization
        
    Dependencies: NumPy, Matplotlib
        
    Version: 01/27/2025
"""

import numpy as np
import matplotlib.pyplot as plt

# matrix (50x40)
original_matrix = np.random.random((50, 40))


def compress_matrix(matrix, rank):
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Truncate to rank
    U_compressed = U[:, :rank]
    s_compressed = s[:rank]
    Vt_compressed = Vt[:rank, :]

    # Reconstruct matrix
    reconstructed = U_compressed @ np.diag(s_compressed) @ Vt_compressed

    # Calculate reconstruction accuracy (1 - normalized error)
    error = np.mean((matrix - reconstructed) ** 2)
    max_possible_error = np.mean(matrix**2)
    accuracy = 1 - (error / max_possible_error)

    # Calculate compression size
    original_size = matrix.size
    compressed_size = U_compressed.size + s_compressed.size + Vt_compressed.size
    compression_ratio = compressed_size / original_size

    return (
        accuracy,
        compressed_size,
        compression_ratio,
        U_compressed.shape,
        s_compressed.shape,
        Vt_compressed.shape,
    )


# Calculate original matrix accuracy and size
original_accuracy = 1.0
original_size = original_matrix.size
print(f"Original matrix shape: {original_matrix.shape}")
print(f"Original matrix size: {original_size} elements")
print(f"Original matrix accuracy: {original_accuracy:.4f}")

# Test different ranks
ranks = [5, 10, 15, 20]
accuracies = []
sizes = []
ratios = []

print("\nCompression Details:")
print("-" * 70)
for rank in ranks:
    accuracy, compressed_size, ratio, U_shape, s_shape, Vt_shape = compress_matrix(
        original_matrix, rank
    )
    accuracies.append(accuracy)
    sizes.append(compressed_size)
    ratios.append(ratio)

    print(f"Rank {rank:2d}:")
    print(f"  - Matrix Shapes after SVD:")
    print(f"    * U matrix: {U_shape} = {U_shape[0] * U_shape[1]} elements")
    print(f"    * s vector: {s_shape} = {s_shape[0]} elements")
    print(f"    * Vt matrix: {Vt_shape} = {Vt_shape[0] * Vt_shape[1]} elements")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Original Size: {original_size} elements")
    print(f"  - Compressed Size: {compressed_size} elements")
    print(
        f"  - Compression Ratio: {ratio:.2f} (model is {ratio*100:.1f}% of original size)"
    )
    print(f"  - Size Reduction: {(1-ratio)*100:.1f}%")
    print("-" * 70)

# Plot accuracy comparison
plt.figure(figsize=(10, 6))

# Plot compressed accuracies
plt.plot(ranks, accuracies, "bo-", linewidth=2, markersize=8, label="Compressed")

# Plot original accuracy
plt.axhline(
    y=original_accuracy,
    color="r",
    linestyle="--",
    label=f"Original ({original_accuracy:.4f})",
)

# Add points for original accuracy at each rank for better comparison
plt.plot(ranks, [original_accuracy] * len(ranks), "ro", markersize=8, alpha=0.3)

plt.xlabel("Rank")
plt.ylabel("Accuracy")
plt.title("Original vs Compressed Matrix Accuracy")
plt.grid(True)
plt.legend()
plt.xticks(ranks)
plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()
