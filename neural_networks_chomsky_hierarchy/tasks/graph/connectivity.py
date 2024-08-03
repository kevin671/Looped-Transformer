# %%

import jax.nn as jnn
import jax.numpy as jnp
import networkx as nx

# import sys
# sys.path.append("/work/gg45/g45004/Looped-Transformer")
from neural_networks_chomsky_hierarchy.tasks import task
from neural_networks_chomsky_hierarchy.tasks.graph.base import GeneratorBase


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
            if nx.is_connected(G):
                outputs = outputs.at[i, 0].set(1)
            else:
                outputs = outputs.at[i, 0].set(0)

        return inputs, outputs


class Connectivity(task.GeneralizationTask):
    """https://arxiv.org/abs/2305.10037

    In an undirected graph G = {V, E}, two nodes u and v are connected if
    there exists a sequence of edges from node u to node v in E. We randomly select two nodes in the
    base graphs u, v ∈ V to ask whether node u and node v are connected with a true/false question.
    We retain a balanced set of questions where half of the node pairs are connected and the other half
    are not connected by discarding additional questions.
    """

    def __init__(self, n_node: int = 10, edge_prob: float = 0.265):
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
    c = Connectivity()
    out = c.sample_batch(jnp.array([0]), 1000, 10)
    inputs, outputs = out["input"], out["output"]  # (1000, 45, 12) (1000, 1, 2)
    print(inputs[0])
    # print rate of 1 in outputs
    print(jnp.mean(outputs[:, 0, 1]))

# %%
