import numpy as np
import time

try:
    from .sbm import DynamicSBMGraphMerge
    from .sparsifier import DynamicGraphSparsifier
    from .contractedgraph import Contracted_Graph
    from .spectral_clustering import spectral_clustering, clusters_to_labels, run_spectral_clustering
except ImportError:
    from sbm import DynamicSBMGraphMerge
    from sparsifier import DynamicGraphSparsifier
    from contractedgraph import Contracted_Graph
    from spectral_clustering import spectral_clustering, clusters_to_labels, run_spectral_clustering


class DynamicSpectralClusteringSBM:
    """
    Dynamic spectral clustering pipeline for SBM graphs.

    This follows the paper-level structure of Algorithm 6:
    1) Preprocess: build G, sparsify to H, cluster H, contract to G_tilde.
    2) Update: maintain G, H, G_tilde under edge insertions.
    3) Query: if updates since last reconstruction <= log(n)^gamma,
       cluster on contracted graph; else cluster on sparsified graph and
       reconstruct contracted graph.
    """

    def __init__(
        self,
        k,
        sizes,
        p,
        q,
        sampling_constant=100,
        degree_trigger=0.5,
        gamma=2.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.k = int(k)
        # Keep gamma fixed to 2 as requested.
        self.gamma = 2.0
        self.degree_trigger = float(degree_trigger)

        # Data structures required by Algorithm 6.
        self.sbm = DynamicSBMGraphMerge(k, sizes, p, q)
        self.G = self.sbm.get_graph()  # actual graph
        self.true_labels = np.array(self.sbm.get_labels())

        self.sparsifier = DynamicGraphSparsifier(self.G, sampling_constant=sampling_constant)
        self.sparsifier.create_sparsifier()
        self.H = self.sparsifier.get_sparsified_graph()  # sparsified graph

        # Initial clustering on sparsifier and contraction.
        init_clusters = spectral_clustering(self.H, self.k)
        init_labels = clusters_to_labels(init_clusters, num_data_points=self.H.num_vertices())

        self.contracted = Contracted_Graph(self.G, self.H, degree_trigger=self.degree_trigger)
        self.contracted.initialize(init_clusters, np.array(init_labels))

        # Degree vector (maintained explicitly as requested).
        self.degree_vector = self.G.degree_property_map("total").a.copy()

        self.total_added_edges = 0

        # Algorithm 6 trigger is maintained in update batches / time steps.
        self.total_update_steps = 0
        self.update_steps_since_rebuild = 0

        # Threshold uses log(n_r)^gamma where r is last reconstruction time.
        self.n_at_last_rebuild = int(self.H.num_vertices())

    def _query_threshold(self):
        n = max(2, int(self.n_at_last_rebuild))
        return max(1, int(np.ceil((np.log(n) ** self.gamma))))

    def _ensure_vertices_exist(self, max_vertex_idx, new_vertex_labels=None):
        """
        Ensure that vertices [0, ..., max_vertex_idx] exist in G.

        Any newly created vertex gets a true label from `new_vertex_labels` if provided,
        otherwise label -1.
        """
        if max_vertex_idx < 0:
            return []

        if new_vertex_labels is None:
            new_vertex_labels = {}

        current_n = int(self.G.num_vertices())
        if max_vertex_idx < current_n:
            return []

        created_vertices = []
        for new_idx in range(current_n, max_vertex_idx + 1):
            self.G.add_vertex()
            label = int(new_vertex_labels.get(new_idx, -1))
            self.true_labels = np.append(self.true_labels, label)
            created_vertices.append(new_idx)

        return created_vertices

    def add_edge_updates(self, new_edges, new_vertex_labels=None):
        """Add a batch of edges and update all maintained structures.

        `new_edges` may include previously unseen vertex IDs. Missing vertices are
        created before edges are inserted.
        """
        self.total_update_steps += 1
        self.update_steps_since_rebuild += 1

        if new_edges is None:
            return []

        if new_vertex_labels is None:
            new_vertex_labels = {}

        # Allow incoming edges to introduce new vertices by index.
        if len(new_edges) > 0:
            max_idx = max(max(int(u), int(v)) for u, v in new_edges)
            added_vertices = self._ensure_vertices_exist(max_idx, new_vertex_labels)
        else:
            added_vertices = []

        added_edges = []
        for u, v in new_edges:
            u = int(u)
            v = int(v)
            if u == v:
                continue
            if self.G.edge(u, v) is None:
                self.G.add_edge(u, v)
                added_edges.append((min(u, v), max(u, v)))

        if len(added_edges) == 0 and len(added_vertices) == 0:
            return []

        # Keep H up-to-date.
        if len(added_edges) == 0:
            incoming = np.empty((0, 2), dtype=int)
        else:
            incoming = np.array(added_edges, dtype=int)
        self.sparsifier.update_sparsifier(incoming, verbose=False)
        self.H = self.sparsifier.get_sparsified_graph()

        # Keep contracted graph up-to-date.
        self.contracted.sparsified_graph = self.H
        self.contracted.update(incoming)

        # Maintain degree vector.
        self.degree_vector = self.G.degree_property_map("total").a.copy()

        self.total_added_edges += len(added_edges)

        return added_edges

    def add_random_overlay_updates(self, s):
        """Sample edge insertions via ER overlay and apply updates."""
        sampled = self.sbm.overlay_erdos_renyi_optimized(float(s))
        return self.add_edge_updates(sampled)

    def _rebuild_contracted_from_sparsifier(self, num_clusters):
        clusters = spectral_clustering(self.H, num_clusters)
        labels = clusters_to_labels(clusters, num_data_points=self.H.num_vertices())

        self.contracted = Contracted_Graph(self.G, self.H, degree_trigger=self.degree_trigger)
        self.contracted.initialize(clusters, np.array(labels))

        self.update_steps_since_rebuild = 0
        self.n_at_last_rebuild = int(self.H.num_vertices())

    def query(self, num_clusters=None, true_labels=None):
        """
        Query clusters using Algorithm-6 style branching.

        Returns a dict with clustering source, clusters/labels and timing.
        """
        if num_clusters is None:
            num_clusters = self.k

        if true_labels is None:
            true_labels = self.true_labels

        threshold = self._query_threshold()
        use_contracted = self.update_steps_since_rebuild <= threshold

        if use_contracted:
            clusters, labels, ari, runtime = self.contracted.spectral_clustering_on_contracted(
                num_clusters, true_labels
            )
            return {
                "source": "contracted",
                "threshold": threshold,
                "updates_since_rebuild": self.update_steps_since_rebuild,
                "clusters": clusters,
                "labels": labels,
                "ari": ari,
                "runtime": runtime,
            }

        clusters, labels, ari, runtime = run_spectral_clustering(self.H, true_labels, num_clusters)
        self._rebuild_contracted_from_sparsifier(num_clusters)

        return {
            "source": "sparsified",
            "threshold": threshold,
            "updates_since_rebuild": self.update_steps_since_rebuild,
            "clusters": clusters,
            "labels": labels,
            "ari": ari,
            "runtime": runtime,
        }


def run_algorithm6_sbm_demo():
    """
    Lightweight demo run on SBM.

    Parameters are SBM-style and can be tuned to match experiments in the paper.
    """
    sizes = [1000, 500, 2000]
    k = len(sizes)
    p = 0.5
    q = 0.1 

    dyn = DynamicSpectralClusteringSBM(
        k=k,
        sizes=sizes,
        p=p,
        q=q,
        sampling_constant=50,
        degree_trigger=0.5,
        gamma=2.0,
        seed=7,
    )

    print(f"Initial graph: n={dyn.G.num_vertices()}, m={dyn.G.num_edges()}")
    print(f"Initial sparsifier: n={dyn.H.num_vertices()}, m={dyn.H.num_edges()}")

    sep = (
        "+-----+-------+-----------+-------------+-----------------+---------+---------------+---------------+"
    )
    header = (
        "|  t  | added | supernode | upd/limit   |   query_graph   |   ari   | query_time(s) | update_time(s) |"
    )
    print(sep)
    print(header)
    print(sep)

    # stream updates and query periodically
    n = dyn.G.num_vertices()
    edges_per_round = 50
    rounds = 800

    for t in range(1, rounds + 1):
        s = edges_per_round / (n * (n - 1) / 2)
        update_start = time.time()
        added = dyn.add_random_overlay_updates(s)
        update_wall = time.time() - update_start

        query_start = time.time()
        result = dyn.query(num_clusters=k)
        query_wall = time.time() - query_start

        print(
            f"| {t:3d} | "
            f"{len(added):5d} | "
            f"{len(dyn.contracted.contracted_node_to_full_node_map):9d} | "
            f"{result['updates_since_rebuild']:5d}/{result['threshold']:<5d} | "
            f"{result['source']:<15s} | "
            f"{result['ari']:7.4f} | "
            f"{query_wall:13.3f} | "
            f"{update_wall:13.3f} |"
        )
    print(sep)


def run_vertex_arrival_new_cluster_demo():
    """
    Demo: each update inserts one new vertex; after `birth_round`, new arrivals
    belong to a brand-new cluster.
    """
    sizes = [500,100]
    base_k = len(sizes)
    new_cluster_label = base_k

    dyn = DynamicSpectralClusteringSBM(
        k=base_k,
        sizes=sizes,
        p=0.6,
        q=0.1,
        sampling_constant=100,
        degree_trigger=0.5,
        gamma=2.0,
        seed=13,
    )

    rounds = 100
    birth_round = 18
    query_with_new_k_round = 24

    print(
        "Vertex-arrival demo:\n"
        f"start_n={dyn.G.num_vertices()}, base_clusters={base_k}, "
        f"birth_round={birth_round}, query_with_{base_k + 1}_clusters_from={query_with_new_k_round}"
    )

    sep = (
        "+-----+-------+-------+-------+-------+-----------+-------------+-----------------+---------+---------------+---------------+"
    )
    header = (
        "|  t  |   n   | new_v | label | added | query_k   | upd/limit   |   query_graph   |   ari   | query_time(s) | update_time(s) |"
    )
    print(sep)
    print(header)
    print(sep)

    for t in range(1, rounds + 1):
        # Next vertex ID (graph-tool keeps integer IDs 0..n-1).
        new_vertex = int(dyn.G.num_vertices())

        # Before birth_round, new vertices join existing clusters.
        # After birth_round, they join a new cluster.
        if t < birth_round:
            assigned_label = int(np.random.randint(0, base_k))
        else:
            assigned_label = int(new_cluster_label)

        current_labels = dyn.true_labels
        same_cluster = np.where(current_labels == assigned_label)[0]
        other_cluster = np.where(current_labels != assigned_label)[0]

        if assigned_label == new_cluster_label:
            # Encourage a coherent new cluster with sparse cross-cluster links.
            intra_edges = min(8, len(same_cluster))
            inter_edges = min(1, len(other_cluster))
        else:
            intra_edges = min(8, len(same_cluster))
            inter_edges = min(2, len(other_cluster))

        candidate_edges = []
        if intra_edges > 0:
            intra_neighbours = np.random.choice(same_cluster, size=intra_edges, replace=False)
            candidate_edges.extend([(new_vertex, int(v)) for v in intra_neighbours if int(v) != new_vertex])

        if inter_edges > 0:
            inter_neighbours = np.random.choice(other_cluster, size=inter_edges, replace=False)
            candidate_edges.extend([(new_vertex, int(v)) for v in inter_neighbours if int(v) != new_vertex])

        # Deduplicate and normalize edge orientation.
        edge_batch = sorted(set((min(u, v), max(u, v)) for u, v in candidate_edges))

        update_start = time.time()
        added_edges = dyn.add_edge_updates(
            edge_batch,
            new_vertex_labels={new_vertex: assigned_label},
        )
        update_wall = time.time() - update_start

        query_k = base_k if t < query_with_new_k_round else base_k + 1
        query_start = time.time()
        result = dyn.query(num_clusters=query_k, true_labels=dyn.true_labels)
        query_wall = time.time() - query_start

        print(
            f"| {t:3d} | "
            f"{dyn.G.num_vertices():5d} | "
            f"{new_vertex:5d} | "
            f"{assigned_label:5d} | "
            f"{len(added_edges):5d} | "
            f"{query_k:9d} | "
            f"{result['updates_since_rebuild']:5d}/{result['threshold']:<5d} | "
            f"{result['source']:<15s} | "
            f"{result['ari']:7.4f} | "
            f"{query_wall:13.3f} | "
            f"{update_wall:13.3f} |"
        )
    print(sep)


if __name__ == "__main__":
    #run_algorithm6_sbm_demo()
    run_vertex_arrival_new_cluster_demo()  # run twice to see variability
