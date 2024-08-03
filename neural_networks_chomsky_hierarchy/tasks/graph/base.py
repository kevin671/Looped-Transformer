from random import randint, random, shuffle

import jax.numpy as jnp
import networkx as nx


class GeneratorBase:
    def __init__(
        self, num_of_nodes=10, edge_probability=0.35, max_length=100, max_weight=10
    ):
        self.num_of_nodes = num_of_nodes
        self.parent_left = num_of_nodes + 0
        self.parent_right = num_of_nodes + 1
        self.query_sign = num_of_nodes + 2
        self.edge_probability = edge_probability
        self.max_length = max_length  # self.num_of_nodes * (self.num_of_nodes - 1) // 2
        self.max_weight = max_weight

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

    def generate_weighted_graph(self):
        idx = list(range(self.num_of_nodes))
        shuffle(idx)
        G = nx.Graph()
        G.add_nodes_from(range(self.num_of_nodes))
        for u in list(G.nodes()):
            for v in list(G.nodes()):
                if u < v and random() < self.edge_probability:
                    weight = randint(1, self.max_weight)
                    G.add_edge(idx[u], idx[v], weight=weight)
        return G

    def generate_directed_graph(self):
        idx = list(range(self.num_of_nodes))
        shuffle(idx)
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_of_nodes))
        for u in list(G.nodes()):
            for v in list(G.nodes()):
                if u < v and random() < self.edge_probability:
                    G.add_edge(idx[u], idx[v])
        return G

    def to_jnp_array(self, G):
        edge = list(G.edges())
        edges_array = []
        for u, v in edge:
            if random() < 0.5:
                # edges_array.extend([u, v])
                edges_array.extend([self.parent_left, u, v, self.parent_right])
            else:
                edges_array.extend([self.parent_left, v, u, self.parent_right])
        return jnp.array(edges_array)

    def generate(self):
        raise NotImplementedError("Subclasses should implement this method.")
