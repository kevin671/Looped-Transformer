# %%
# import sys
from random import shuffle

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
        outputs = jnp.full((n, self.num_of_nodes), -1)

        for i in range(n):
            G = self.generate_weighted_graph()
            graph_array = self.to_jnp_array(G)
            length = min(graph_array.size, self.max_length)
            inputs = inputs.at[i, :length].set(graph_array[:length])

            nodes = list(G.nodes())
            shuffle(nodes)
            while True:
                for u in list(nodes):
                    for v in list(nodes):
                        if u != v and not G.has_edge(u, v) and nx.has_path(G, u, v):
                            # print(u, v)
                            inputs = inputs.at[i, length : length + 3].set(
                                [self.query_sign, u, v]
                            )
                            shortest_path = nx.shortest_path(G, u, v)
                            # print(shortest_path)
                            outputs = outputs.at[i, : len(shortest_path)].set(
                                shortest_path
                            )

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

# %%


class ShortestPath(task.GeneralizationTask):
    """https://arxiv.org/abs/2305.10037

    Shortest Path The shortest path between two nodes is the path with the sum of edge
    weights minimized. Given an undirected graph G = {V, E}, a positive weight w for each edge,
    and two nodes u and v, the task is to find the shortest path between node u and node v and its
    corresponding path length.
    """

    def __init__(self, n_node: int = 10, edge_prob: float = 0.35):
        self._n_node = n_node
        self._edge_prob = edge_prob
        max_length = n_node * (n_node - 1) // 2 * 4 + 3
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
        # parent_left, parent_right, query_sign
        return self._n_node + 3

    @property
    def output_size(self) -> int:
        """Returns the output zsize for the models."""
        # TODO: add the path length
        return self._n_node

    def output_length(self, input_length: int) -> int:
        """Returns the output length for a given input length."""
        del input_length
        return self._n_node


# %%

if __name__ == "__main__":
    c = ShortestPath()
    out = c.sample_batch(jnp.array([0]), 1000, 10)
    inputs, outputs = out["input"], out["output"]  # (1000, 183, 13) (1000, 10, 10)
    print(inputs.shape, outputs.shape)
    print(inputs[0, 100:])
    print(outputs[0])

# %%
