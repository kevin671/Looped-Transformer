# %%
# import sys

import jax.nn as jnn
import jax.numpy as jnp
import networkx as nx

# sys.path.append("/work/gg45/g45004/Looped-Transformer")
from neural_networks_chomsky_hierarchy.tasks import task
from neural_networks_chomsky_hierarchy.tasks.graph.base import GeneratorBase


class Generator(GeneratorBase):
    def __init__(self, num_of_nodes=10, edge_probability=0.35, max_length=100):
        super().__init__(num_of_nodes, edge_probability, max_length)

    def generate(self, n=1):
        inputs = jnp.full((n, self.max_length), -1)
        outputs = jnp.zeros((n))

        for i in range(n):
            G = self.generate_graph()
            graph_array = self.to_jnp_array(G)

            # Ensure the input graph array fits into the preallocated space
            length = min(graph_array.size, self.max_length)
            inputs = inputs.at[i, :length].set(graph_array[:length])

            # Determine the output based on the presence of a cycle
            if nx.cycle_basis(G):
                outputs = outputs.at[i].set(1)
            else:
                outputs = outputs.at[i].set(0)

        return inputs, outputs


# g = Generator()
# inputs, outputs = g.generate(5)
# print(inputs.shape, outputs.shape)  # (5, 45) (5, 1)
# print(inputs[0], outputs[0])
# to one hot
# inputs = jnn.one_hot(inputs, 10)
# outputs = jnn.one_hot(outputs, 2)
# print(inputs.shape, outputs.shape)  # (5, 45, 10) (5, 1, 2)
# print(inputs[0], outputs[0])


class Cycle(task.GeneralizationTask):
    """https://arxiv.org/abs/2305.10037

    Cycle In an undirected graph G = {V, E}, a cycle is a non-empty trail (e1, e2, . . . , en)
    with a node sequence (v1, v2, . . . , vn, v1). We present an undirected graph G = {V, E} to ask
    whether there exists a cycle through true/false questions. We retain base graphs without cycles as
    the False subset, while we randomly add edges to these base graphs to generate graphs with cycles
    as the True subset. We retain a balanced set of cyclic and noncyclic graphs in the dataset.

    Examples:
        09318115219237353674944585567668967897 1
    """

    def __init__(self, n_node: int = 10, edge_prob: float = 0.16):
        self._n_node = n_node
        self._edge_prob = edge_prob
        max_length = n_node * (n_node - 1) // 2 * 4
        self._generator = Generator(
            num_of_nodes=n_node, edge_probability=edge_prob, max_length=max_length
        )

    def sample_batch(self, rng: jnp.ndarray, batch_size: int, length: int):
        """Returns a batch of inputs/outputs."""
        del rng
        inputs, outputs = self._generator.generate(n=batch_size)
        inputs = jnn.one_hot(inputs, self.input_size)
        output = jnn.one_hot(outputs, self.output_size)
        return {"input": inputs, "output": output}

    @property
    def input_size(self) -> int:
        """Returns the input size for the models."""
        # parent_left, parent_right
        return self._n_node + 2

    @property
    def output_size(self) -> int:
        """Returns the output zsize for the models."""
        # yes or no
        return 2


# %%

if __name__ == "__main__":
    c = Cycle()
    out = c.sample_batch(jnp.array([0]), 1000, 10)
    inputs, outputs = out["input"], out["output"]  # (1000, 45, 12) (1000, 2)
    print(inputs.shape, outputs.shape)
    print(inputs[0], outputs[0])
    print(jnp.mean(outputs[:, 0]))


# %%
