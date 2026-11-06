import graph_tool as gt
import graph_tool.generation
import numpy as np

try:
    from datasets import *
except ImportError:
    pass


class DynamicSBMGraphMerge:
    def __init__(self, k, sizes, p, q):
        """Dynamic SBM with nonuniform cluster sizes."""
        self.k = k
        self.sizes = sizes
        self.p = p
        self.q = q
        self.graph, self.labels = self._generate_sbm_graph()
        gt.generation.remove_self_loops(self.graph)
        self.available_vertices = set(range(sum(sizes)))

    def _generate_sbm_graph(self):
        """Return a graph-tool SBM instance and its cluster labels."""
        import numpy as np
        import graph_tool.all as gt

        b = np.concatenate([np.full(size, i) for i, size in enumerate(self.sizes)])

        probs = np.zeros((self.k, self.k))

        for i in range(self.k):
            n_i = self.sizes[i]
            probs[i, i] = self.p * n_i * (n_i - 1) / 2

        for i in range(self.k):
            for j in range(i + 1, self.k):
                n_i, n_j = self.sizes[i], self.sizes[j]
                probs[i, j] = probs[j, i] = self.q * n_i * n_j

        g = graph_tool.generation.generate_sbm(b, probs, directed=False)

        return g, b

    def overlay_erdos_renyi_optimized(self, s):
        """Sample additional unique Erdos-Renyi edges without modifying the graph."""
        N = self.graph.num_vertices()
        total_possible_edges = N * (N - 1) // 2
        num_edges_to_add = int(np.round(s * total_possible_edges))

        new_edges = set()
        attempts = 0
        max_attempts = num_edges_to_add * 10

        while len(new_edges) < num_edges_to_add and attempts < max_attempts:
            u, v = np.random.randint(0, N, size=2)
            if u != v and not self.graph.edge(u, v):
                new_edges.add((min(u, v), max(u, v)))
            attempts += 1

        return list(new_edges)

    def merge_clusters(self, label_a, label_b, r):
        """
        Samples edges between two clusters with probability r without directly updating the graph.
        Returns the sampled edges as a 2xm numpy array and the updated labels array.
        """
        import numpy as np

        vertices_a = np.where(self.labels == label_a)[0]
        vertices_b = np.where(self.labels == label_b)[0]

        sampled_edges_list = []

        for v_a in vertices_a:
            for v_b in vertices_b:
                if np.random.rand() < r:
                    sampled_edges_list.append([v_a, v_b])

        self.labels[vertices_b] = label_a

        return sampled_edges_list, self.labels

    def get_graph(self):
        """Return the current graph-tool graph."""
        return self.graph

    def get_labels(self):
        """Return the current cluster labels."""
        return self.labels


class DynamicSBMGraph:
    def __init__(self, k, n, p, q):
        """Dynamic SBM with equal-size initial clusters."""
        self.k = k
        self.n = n
        self.p = p
        self.q = q
        self.graph, self.labels = self._generate_sbm_graph()
        gt.generation.remove_self_loops(self.graph)
        self.available_vertices = set(range(k * n))

    def _generate_sbm_graph(self):
        """Return a graph-tool SBM instance and its cluster labels."""
        b = np.repeat(np.arange(self.k), self.n)

        probs = np.full((self.k, self.k), self.q)
        np.fill_diagonal(probs, self.p)

        n = self.n
        probs = probs * (n * (n - 1) / 2) if self.k > 1 else probs * (n * (n)) / 2
        probs = probs * 2

        g = gt.generation.generate_sbm(b, probs, directed=False)

        return g, b

    def sample_internal_edges(self, n_small, r):
        """Move a random subset into a new cluster and sample its internal edges."""
        if n_small > len(self.available_vertices):
            raise ValueError("n_small is larger than the number of available vertices.")

        selected_vertices = np.random.choice(
            list(self.available_vertices), size=n_small, replace=False
        )
        self.available_vertices -= set(selected_vertices)

        new_cluster_id = max(self.labels) + 1 if len(self.labels) > 0 else 0
        self.labels[selected_vertices] = new_cluster_id

        sampled_edges = []
        for i in range(n_small):
            for j in range(i + 1, n_small):
                if np.random.rand() < r:
                    sampled_edges.append((selected_vertices[i], selected_vertices[j]))

        return sampled_edges, self.labels

    def overlay_erdos_renyi_optimized(self, s):
        """Sample additional unique Erdos-Renyi edges without modifying the graph."""
        N = self.graph.num_vertices()
        total_possible_edges = N * (N - 1) // 2
        num_edges_to_add = int(np.round(s * total_possible_edges))

        new_edges = set()
        attempts = 0
        max_attempts = num_edges_to_add * 10

        while len(new_edges) < num_edges_to_add and attempts < max_attempts:
            u, v = np.random.randint(0, N, size=2)
            if u != v and not self.graph.edge(u, v):
                new_edges.add((min(u, v), max(u, v)))
            attempts += 1

        return list(new_edges)

    def get_graph(self):
        """Return the current graph-tool graph."""
        return self.graph

    def get_labels(self):
        """Return the current cluster labels."""
        return self.labels
