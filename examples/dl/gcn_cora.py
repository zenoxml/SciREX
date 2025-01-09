#!/usr/bin/env python
# coding: utf-8

# # Graph Convolution Network (GCN) for a classification problem
# 
# We will work with the CORA dataset from pytorch geometirc.
# Each paper is assingned a high dimensional vector based on its contents
# and we will work with the directed graph formed by the citations.
# Each paper or node is classified based on research topic.

# In[1]:


from torch_geometric.datasets import Planetoid

# Load the CORA dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Access the first graph object
data = dataset[0]


# In[2]:


print("No. of nodes = ", data['x'].shape[0])
print("No. of edges = ", data['edge_index'].shape[1])
print("Classification labels range from ", data['y'].min(), " to ", data['y'].max())


# To work with the jax ecosystem we will change the data type to a jax array

# In[3]:


import jax.numpy as jnp

x = jnp.asarray(data['x'].numpy())
edge_index = jnp.asarray(data['edge_index'].numpy())
y = jnp.asarray(data['y'].numpy())

num_nodes = x.shape[0]
node_vector_size = x.shape[1]
num_classes = int(y.max()) + 1


# We will use the edge index information to form the adjacency matrix which will be 2D jax array. Ideally we should use sparse matrices for efficiency.

# In[4]:


A = jnp.zeros((num_nodes,num_nodes))
A = A.at[edge_index[0], edge_index[1]].set(1)


degree = A.sum(axis=0) # Check axis

print("Adjacency matrix of dimensions ", A.shape)
print("Min degree ", degree.min())
print("Max degree ", degree.max())


# Now we will use the gcn class defined in gcn.py to transform the node features to vectors of size = number of classes. Initially the parameters of the transformation are set randomly.

# In[5]:


import jax
from scirex.core.dl.gcn import GCN, GCNModel

key = jax.random.PRNGKey(42)

model_key, key = jax.random.split(key)
gcn_transformation = GCN([node_vector_size, 100, 100, num_classes], model_key)


# We will randomly select nodes for training. We will use one-hot encodings as target vectors.

# In[6]:


num_of_nodes_in_training_set = int(0.5 * num_nodes)
training_set = jax.random.choice(key, num_nodes, (num_of_nodes_in_training_set,), replace=False)

Y = jnp.zeros((num_nodes, num_classes), dtype=int)
Y = Y.at[training_set,y[training_set]].set(1)


# Now we can define our cross entropy loss function for the training set. To define the model, we apply the softmax function to the node embeddings we get as output. This is useful as we will generate one-hot encodings as our target output and use the cross-entropy function.

# In[7]:


def loss_fn(output, target):
    predicted_probs =  jax.nn.softmax(output, axis = 1)
    cross_entropy_terms = - target * jnp.log(predicted_probs)
    return cross_entropy_terms.sum()


# We define functions for calculating the training and overall accuracy

# In[8]:


def accuracy_fn(labels, node_indices):
    def subset_accuracy(output):
        predicted_probs =  jax.nn.softmax(output, axis = 1)[node_indices]
        predicted_labels = predicted_probs.argmax(axis = 1)
        predicted_correct = jnp.where(labels[node_indices] == predicted_labels, 1, 0)
        return predicted_correct.sum() / predicted_correct.shape[0]
    return subset_accuracy

training_accuracy = accuracy_fn(y, training_set)
overall_accuracy = accuracy_fn(y, jnp.arange(num_nodes))

model = GCNModel(gcn_transformation, loss_fn, [training_accuracy, overall_accuracy])

# Now we begin the training process

# In[9]:

gcn_transformation = model.fit(x, A, degree, Y, learning_rate = 5e-2, num_iters = 50)

# # Now let us plot the loss plot

# import matplotlib.pyplot as plt
# 
# plt.plot(loss_values)
# plt.xlabel("optim iteration")
# plt.ylabel("loss value")
# plt.grid()
# plt.show()
# 
# 
# # In[11]:
# 
# 
# plt.plot(train_acc_list, label="train")
# plt.plot(overall_acc_list, label="overall")
# plt.legend()
# plt.xlabel("optim iteration")
# plt.ylabel("accuracy")
# plt.grid()
# plt.show()
# 
# 
# # In[14]:
# 
# 

final_node_embeddings = gcn_transformation(x, A, degree)
final_training_accuracy_percentage = training_accuracy(final_node_embeddings) * 100
final_overall_accuracy_percentage = overall_accuracy(final_node_embeddings) * 100

print(f"Final training accuracy = {final_training_accuracy_percentage:.2f}%")
print(f"Final overall accuracy = {final_overall_accuracy_percentage:.2f}%")

