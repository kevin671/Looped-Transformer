from neural_networks_chomsky_hierarchy.tasks import task
import jax.nn as jnn
import jax.numpy as jnp
from random import shuffle, random
import networkx as nx


class Generator:
    def __init__(
        self,
        num_of_nodes=10,
        edge_probability=0.35,
    ):
        self.num_of_nodes = num_of_nodes
        self.edge_probability = edge_probability
        self.max_graph_length = self.num_of_nodes * (self.num_of_nodes - 1) // 2

    def generate_graph(self):
        idx = list(range(self.num_of_nodes))
        shuffle(idx)
        G = nx.Graph()
        G.add_nodes_from(range(self.num_of_nodes))
        for u in list(G.nodes()):
            for v in list(G.nodes()):
                if u < v and random() < self.edge_probability:
                    G.add_edge(idx[u], idx[v])
        return G

    def is_cycle(self, G):
        return nx.cycle_basis(G)

    def to_jnp_array(self, G):
        edge = list(G.edges())
        edges_array = []
        for u, v in edge:
            if random() < 0.5:
                edges_array.extend([u, v])
            else:
                edges_array.extend([v, u])
        return jnp.array(edges_array)

    def generate(self, n=1):
        max_graph_length = (
            self.max_graph_length
        )  # Assume this is a predefined attribute
        inputs = jnp.zeros((n, max_graph_length))
        outputs = jnp.zeros((n, 1))

        for i in range(n):
            G = self.generate_graph()
            graph_array = self.to_jnp_array(G)

            # Ensure the input graph array fits into the preallocated space
            length = min(graph_array.size, max_graph_length)
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

    def __init__(self, n_node: int = 10, edge_prob: float = 0.35):
        self._n_node = n_node
        self._edge_prob = edge_prob
        self._generator = Generator(num_of_nodes=n_node, edge_probability=edge_prob)

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
        return self._n_node

    @property
    def output_size(self) -> int:
        """Returns the output zsize for the models."""
        # yes or no
        return 1
