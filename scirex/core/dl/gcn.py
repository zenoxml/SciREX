import jax
import jax.numpy as jnp

import equinox as eqx
import optax

from tqdm import tqdm

class GCN(eqx.Module):
    num_layers: int
    W_list: list
    B_list: list

    # activations: list[callable]

    def __init__(
            self,
            layers,
            # activations,
            key):
        """
        Inputs:
            layers is a python list indicating the size of the node embeddings at each layer
            key is used to generate random numbers for initialising the W and B matrices
        """
        
        self.num_layers = len(layers)
        self.W_list = []
        self.B_list = []

        # self.activations = activations

        for i in range(self.num_layers-1):
            weights_key, bias_key, key = jax.random.split(key, num=3)
            W = jax.random.normal(weights_key, (layers[i], layers[i+1]))
            B = jax.random.normal(bias_key, (layers[i], layers[i+1]))

            self.W_list.append(W)
            self.B_list.append(B)

    def __call__(self, z, adj_mat, degree):
        """
        Inputs:
            z is a jnp array for which the i-th row is the i-th node embedding
            adj_mat is the adjacency matrix. Ideally it should be a sparse matrix
            degree is a jnp array where the i-th element is the degree of the i-th node

        Output:
            Similar to z. The node embeddings of the output
        """

        activation = jnp.tanh
        # for activation,W,B in zip(self.activations,self.W_list,self.B_list):
        for W,B in zip(self.W_list,self.B_list):
            z = activation(jnp.diagflat(1.0/degree) @ adj_mat @ z @ W + z @ B)
        return z

class GCNModel():

    def __init__(
        self,
        gcn_transformation: GCN,
        loss_fn: callable,
        metrics: list[callable] = [],
        ):

        self.gcn_transformation = gcn_transformation
        self.loss_fn = loss_fn
        self.metrics = metrics

    def fit(
        self,
        features: jnp.ndarray,
        adjacency_matrix: jnp.ndarray,
        degree_array: jnp.ndarray,
        target: jnp.ndarray,
        learning_rate: float,
        num_iters: int = 10,
        num_check_points: int = 5
        ):
        """
        Train the gcn

        Args:
            features: jnp.ndarray,
            adjacency_matrix: jnp.ndarray,
            degree_array: jnp.ndarray,
            target: jnp.ndarray,
            learning_rate: float,
            num_iters: int = 10,
            num_check_points: int = 5
        """
        check_point_gap = num_iters / num_check_points

        optim = optax.adam(learning_rate = learning_rate)
        opt_state = optim.init(self.gcn_transformation)

        for iter_id in tqdm(range(num_iters), desc="Training", total=num_iters):
            loss, grads = eqx.filter_value_and_grad(self._loss_fn)(
                    self.gcn_transformation,
                    features,
                    adjacency_matrix,
                    degree_array,
                    target
                    )

            updates, opt_state = optim.update(grads, opt_state)
            self.gcn_transformation = eqx.apply_updates(self.gcn_transformation, updates)

            if iter_id % check_point_gap == 0:
                output = self.gcn_transformation(features, adjacency_matrix, degree_array)
                metric_vals = [ m(output) for m in self.metrics ]
                print(f"Iter: {iter_id} | Loss: {loss:.2e} | Metrics {metric_vals}")

        return self.gcn_transformation

    def _loss_fn(
            self,
            gcn_transformation: GCN,
            features: jnp.ndarray,
            adjacency_matrix,
            degree_array,
            target: jnp.ndarray
            ):
        """
        Compute loss for the given input data.
        Required for getting gradients during training and JIT.

        Args:
            net (Network): Neural network to compute loss.
            x (jnp.ndarray): Input features for loss computation.
            y (jnp.ndarray): Target values for loss computation.

        Returns:
            jnp.ndarray: Loss value.
        """
        # Check why the vmap does not work properly
        # return jax.vmap(self.loss_fn)(jax.vmap(gcn_transformation)(features, adjacency_matrix, degree_array), target).mean()
        return self.loss_fn(gcn_transformation(features, adjacency_matrix, degree_array), target).mean()


