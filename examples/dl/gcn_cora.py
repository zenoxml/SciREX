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
    Example Script: gcn_cora.py

    This script demonstrates how to use the graph convolution network
    implementation from the SciREX library to perform classification
    on the CORA dataset from pytorch geometirc.

    This example includes:
        - Loading the CORA dataset using pytorch.geometric
        - Training Graph Convolutional Neural Networks
        - Evaluating the results

    Key Features:
        - Uses cross-entropy loss for training
        - Implements accuracy metric for evaluation

    Authors:
        - Rajarshi Dasgupta (rajarshid@iisc.ac.in)

    Version Info:
        - 10/01/2025: Initial version

"""

import jax
import jax.numpy as jnp

from scirex.core.dl.gcn import GCN, GCNModel

from torch_geometric.datasets import Planetoid

# Load the CORA dataset
data = Planetoid(root='/tmp/Cora', name='Cora')[0]

"""
    From the CORA dataset from pytorch geometric we obtain
        - node feature vectors
            There are 2,708 nodes.
            Each node represents a scientific publication.
            The size of the node feature vector is 1,433
            and has values 0 or 1.
            A dictionary of 1,433 unique words
            is used to assign the node feature vector.
            The i-th value of the node feature vector is 1,
            if the i-th word in the dictionary is present
            in the research paper represented by the node and 0 otherwise.
        - node labels
            There are 7 classes indicating the topics of the research paper
            and the nodes have labels 0 to 6.
        - edges
            Research publication contain citations
            to other scientific works.
            The dataset provides edges
            which are a set of pairs of nodes connected by citation.
            The citation network is represented by a directed graph.
"""

print("No. of nodes = ", data['x'].shape[0])
print("No. of edges = ", data['edge_index'].shape[1])
print("Classification labels range from ", data['y'].min(), " to ", data['y'].max())


# To work with the jax ecosystem we change the data type to a jax ndarray
x = jnp.asarray(data['x'].numpy())
edge_index = jnp.asarray(data['edge_index'].numpy())
y = jnp.asarray(data['y'].numpy())

num_nodes = x.shape[0]
node_vector_size = x.shape[1]
num_classes = int(y.max()) + 1


# Adjacency matrix and degree array are formed
A = jnp.zeros((num_nodes,num_nodes))
A = A.at[edge_index[0], edge_index[1]].set(1)
degree = A.sum(axis=0) # Check axis

print("Adjacency matrix of dimensions ", A.shape)
print("Min degree ", degree.min())
print("Max degree ", degree.max())

# The GCN class is used to transform the input for classification
key = jax.random.PRNGKey(42)
model_key, key = jax.random.split(key)

gcn = GCN(
        [node_vector_size, 100, 100, num_classes],
        [jnp.tanh]*3,
        model_key
        )

# One-hot encodings are generated for the training set
num_of_nodes_in_training_set = int(0.5 * num_nodes)
training_set = jax.random.choice(
        key,
        num_nodes,
        (num_of_nodes_in_training_set,),
        replace=False
        )

Y = jnp.zeros((num_nodes, num_classes), dtype=int)
Y = Y.at[training_set,y[training_set]].set(1)


# The loss function is evaluated by the cross entropy loss on the gcn output
def loss_fn(output, target):
    predicted_probs =  jax.nn.softmax(output, axis = 1)
    cross_entropy_terms = - target * jnp.log(predicted_probs)
    return cross_entropy_terms.sum()


# Functions for calculating the training and overall accuracy are defined
def accuracy_fn(labels, node_indices):
    def subset_accuracy(output):
        predicted_probs =  jax.nn.softmax(output, axis = 1)[node_indices]
        predicted_labels = predicted_probs.argmax(axis = 1)
        predicted_correct = jnp.where(
                labels[node_indices] == predicted_labels,
                1,
                0)
        return predicted_correct.sum() / predicted_correct.shape[0]
    return subset_accuracy

training_accuracy = accuracy_fn(y, training_set)
overall_accuracy = accuracy_fn(y, jnp.arange(num_nodes))

# The model is defined and trained

model = GCNModel(
        gcn,
        loss_fn,
        [training_accuracy, overall_accuracy]
        )

gcn = model.fit(
        x,
        A,
        degree,
        Y,
        learning_rate = 5e-2,
        num_iters = 50)

# Results
final_node_embeddings = gcn(x, A, degree)
final_training_accuracy_percentage = 100.0 * training_accuracy(
        final_node_embeddings
        )
final_overall_accuracy_percentage = 100.0 * overall_accuracy(
        final_node_embeddings
        )

print(f"Final training accuracy = {final_training_accuracy_percentage:.2f}%")
print(f"Final overall accuracy = {final_overall_accuracy_percentage:.2f}%")

