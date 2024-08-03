# %%
import jax.nn as jnn
import jax.numpy as jnp
import networkx as nx

# import sys
# sys.path.append("/work/gg45/g45004/Looped-Transformer")
from neural_networks_chomsky_hierarchy.tasks import task
from neural_networks_chomsky_hierarchy.tasks.graph.base import GeneratorBase


# https://gist.github.com/mikkelam/ab7966e7ab1c441f947b
def hamilton(G):
    F = [(G, [list(G.nodes())[0]])]
    n = G.number_of_nodes()
    while F:
        graph, path = F.pop()
        confs = []
        neighbors = (
            node for node in graph.neighbors(path[-1]) if node != path[-1]
        )  # exclude self loops
        for neighbor in neighbors:
            conf_p = path[:]
            conf_p.append(neighbor)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g, conf_p))
        for g, p in confs:
            if len(p) == n:
                return p
            else:
                F.append((g, p))
    return None


class Generator(GeneratorBase):
    def __init__(self, num_of_nodes=10, edge_probability=0.35, max_length=100):
        super().__init__(num_of_nodes, edge_probability, max_length)

    def generate(self, n=1):
        inputs = jnp.full((n, self.max_length), -1)
        outputs = jnp.zeros((n, 1))

        for i in range(n):
            G = self.generate_graph()
            graph_array = self.to_jnp_array(G)

            # Ensure the input graph array fits into the preallocated space
            length = min(graph_array.size, self.max_length)
            inputs = inputs.at[i, :length].set(graph_array[:length])

            # Determine the output based on the presence of a cycle
            if hamilton(G):
                outputs = outputs.at[i, 0].set(1)
            else:
                outputs = outputs.at[i, 0].set(0)

        return inputs, outputs


# %%
# g = Generator()
# inputs, outputs = g.generate(5)
# print(inputs.shape, outputs.shape)  # (5, 45) (5, 1)
# print(inputs[0], outputs[0])


# %%
class HamiltonPath(task.GeneralizationTask):
    """https://arxiv.org/abs/2305.10037

    Hamilton Path In an undirected graph, a Hamilton path is a path that visits every node
    exactly once. Given an undirected graph G = {V, E}, the task is to find a valid Hamilton path.
    """

    def __init__(self, n_node: int = 10, edge_prob: float = 0.35):
        self._n_node = n_node
        self._edge_prob = edge_prob
        max_length = n_node * (n_node - 1) // 2 * 4
        self._generator = Generator(
            num_of_nodes=n_node, edge_probability=edge_prob, max_length=max_length
        )

    def sample_batch(
        self, rng: jnp.ndarray, batch_size: int, length: int
    ) -> task.Batch:
        """Returns a batch of inputs/outputs."""
        del rng
        inputs, outputs = self._generator.generate(n=batch_size)
        inputs = jnn.one_hot(inputs, self.input_size)
        output = jnn.one_hot(outputs, self.output_size)
        return {"input": inputs, "output": output}

    @property
    def input_size(self) -> int:
        """Returns the input size for the models."""
        return self._n_node + 2

    @property
    def output_size(self) -> int:
        """Returns the output zsize for the models."""
        # yes or no
        return 2


# %%
if __name__ == "__main__":
    c = HamiltonPath()
    out = c.sample_batch(jnp.array([0]), 1000, 10)
    inputs, outputs = out["input"], out["output"]  # (1000, 45, 12) (1000, 1, 2)
    print(inputs[0])
    # print rate of 1 in outputs
    print(jnp.mean(outputs[:, 0, 1]))

# %%
