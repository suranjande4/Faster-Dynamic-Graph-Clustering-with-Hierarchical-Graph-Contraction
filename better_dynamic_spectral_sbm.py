import time
import numpy as np
import graph_tool as gt

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


class BetterDynamicSpectralClusteringSBM:
    """
    Algorithm-3 style dynamic spectral clustering with a semi-contracted graph.

    Core idea from the provided "Better Dynamic Clustering" draft:
    1) Keep the standard dynamic sparsifier H_t.
    2) Keep a dynamic contracted graph G_t~.
    3) Between global reconstructions, unfold "small" supernodes so G_t~ becomes
       semi-contracted: large clusters stay as supernodes, small clusters are
       represented by original vertices.

    Notes on implementation choices from the draft pseudocode:
    - The draft's Algorithm 5 checks `vol(P_i) = alpha`; in practice we use
      `vol(P_i) <= alpha` to define "small clusters" robustly.
    - Unfolding is done during updates (AddNewEdge), matching the intent in the
      comments/prose around Algorithm 3/5.
    """

    def __init__(
        self,
        k,
        sizes,
        p,
        q,
        beta=0.5,
        sampling_constant=100,
        degree_trigger=0.5,
        gamma=2.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.k = int(k)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.degree_trigger = float(degree_trigger)

        # Original graph + labels (SBM generator).
        self.sbm = DynamicSBMGraphMerge(k, sizes, p, q)
        self.G = self.sbm.get_graph()
        self.true_labels = np.array(self.sbm.get_labels())

        # Dynamic sparsifier H_t.
        self.sparsifier = DynamicGraphSparsifier(self.G, sampling_constant=sampling_constant)
        self.sparsifier.create_sparsifier()
        self.H = self.sparsifier.get_sparsified_graph()

        # Initial contracted graph from spectral clustering on H_1.
        init_clusters = spectral_clustering(self.H, self.k)
        init_labels = clusters_to_labels(init_clusters, num_data_points=self.H.num_vertices())
        self.contracted = Contracted_Graph(self.G, self.H, degree_trigger=self.degree_trigger)
        self.contracted.initialize(init_clusters, np.array(init_labels))

        # Update/reconstruction counters.
        self.total_update_steps = 0
        self.update_steps_since_rebuild = 0
        self.total_added_edges = 0
        self.n_at_last_rebuild = int(self.G.num_vertices())

        # Rolling unfolding budget:
        # alpha starts at 0 and increases with each added edge.
        self.alpha = 0
        # Rebuild budget limit for the current epoch.
        self.alpha_limit = self._alpha_threshold(self.n_at_last_rebuild)

    @staticmethod
    def _safe_weighted_edges(graph: gt.Graph):
        """Return np.array([u, v, w]) for all edges; default w=1 when missing."""
        if "edge_weight" in graph.edge_properties:
            ep = graph.ep["edge_weight"]
            try:
                return graph.get_edges([ep])
            except Exception:
                return np.array([[int(e.source()), int(e.target()), float(ep[e])] for e in graph.edges()])

        edges = np.array(list(graph.get_edges()))
        if edges.size == 0:
            return np.zeros((0, 3))
        return np.column_stack((edges[:, 0].astype(int), edges[:, 1].astype(int), np.ones(len(edges))))

    def _alpha_threshold(self, n):
        return max(1, int(np.ceil(float(max(1, n)) ** self.beta)))

    def _query_threshold(self):
        # In one epoch, alpha is compared to the fixed limit n_r^beta.
        return self.alpha_limit

    def _ensure_vertices_exist(self, max_vertex_idx, new_vertex_labels=None):
        """
        Ensure vertices [0, ..., max_vertex_idx] exist in G.

        Any newly created vertex gets label from new_vertex_labels if available,
        otherwise -1.
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

    def _cluster_volumes(self):
        """Compute weighted volume (incident edge-weight sum) per node in contracted graph."""
        cg = self.contracted.contracted_graph
        m = len(self.contracted.contracted_node_to_full_node_map)
        volumes = np.zeros(m, dtype=float)

        edges = self._safe_weighted_edges(cg)
        if edges.size == 0:
            return volumes

        for cid in range(m):
            mask = (edges[:, 0].astype(int) == cid) | (edges[:, 1].astype(int) == cid)
            volumes[cid] = float(np.sum(edges[mask, 2]))

        return volumes

    def _rebuild_contracted_from_mapping(self, updated_clusters):
        """
        Build semi-contracted graph from cluster mapping and sparsifier H.

        updated_clusters: list[set[int]] where each set is one node in the
        semi-contracted graph (either a supernode or singleton original vertex).
        """
        num_new = len(updated_clusters)
        n_h = int(self.H.num_vertices())

        vertex_to_cluster = np.full(n_h, -1, dtype=int)
        for cid, cluster in enumerate(updated_clusters):
            for v in cluster:
                iv = int(v)
                if 0 <= iv < n_h:
                    vertex_to_cluster[iv] = cid

        edges_h = self._safe_weighted_edges(self.H)

        new_g = gt.Graph(directed=False)
        new_g.add_vertex(num_new)
        new_ep = new_g.new_edge_property("double")

        if edges_h.size > 0:
            src = vertex_to_cluster[edges_h[:, 0].astype(int)]
            tgt = vertex_to_cluster[edges_h[:, 1].astype(int)]
            valid = (src >= 0) & (tgt >= 0)

            src = src[valid]
            tgt = tgt[valid]
            w = edges_h[valid, 2].astype(float)

            if src.size > 0:
                pairs = np.column_stack((src, tgt, w))
                unique_pairs, inv = np.unique(pairs[:, :2].astype(int), axis=0, return_inverse=True)
                total_w = np.zeros(unique_pairs.shape[0], dtype=float)
                np.add.at(total_w, inv, pairs[:, 2])

                edges_and_weights = np.hstack((unique_pairs, total_w[:, None]))
                new_g.add_edge_list(edges_and_weights, eprops=[new_ep])

        new_g.edge_properties["edge_weight"] = new_ep

        self.contracted.contracted_graph = new_g
        self.contracted.contracted_node_to_full_node_map = updated_clusters

        # Keep update metadata in sync for subsequent UpdateContractedGraph calls.
        self.contracted.og_num_contract_nodes = len(updated_clusters)
        self.contracted.degree_array = self.G.degree_property_map("total").a.copy()
        self.contracted.degree_array_at_construction = self.contracted.degree_array.copy()

    def unfold_small_clusters(self, alpha=None):
        """
        Algorithm-5 style unfolding for small clusters.

        Unfold each current supernode whose contracted-graph volume is <= alpha,
        replacing it with singleton original vertices.
        """
        if alpha is None:
            alpha = self.alpha

        current_map = self.contracted.contracted_node_to_full_node_map
        if len(current_map) == 0:
            return 0

        volumes = self._cluster_volumes()

        # Only unfold proper supernodes (size > 1).
        to_unfold = [
            i
            for i, cluster in enumerate(current_map)
            if len(cluster) > 1 and volumes[i] <= float(alpha)
        ]

        if len(to_unfold) == 0:
            return 0

        to_unfold_set = set(to_unfold)
        updated = []

        for idx, cluster in enumerate(current_map):
            if idx in to_unfold_set:
                for v in sorted(list(cluster)):
                    updated.append(set([int(v)]))
            else:
                updated.append(set(int(v) for v in cluster))

        self._rebuild_contracted_from_mapping(updated)
        return len(to_unfold)

    def _global_reconstruct(self, num_clusters):
        """Rebuild contracted graph from spectral clustering on the current sparsifier H."""
        clusters = spectral_clustering(self.H, num_clusters)
        labels = clusters_to_labels(clusters, num_data_points=self.H.num_vertices())

        self.contracted = Contracted_Graph(self.G, self.H, degree_trigger=self.degree_trigger)
        self.contracted.initialize(clusters, np.array(labels))

        self.update_steps_since_rebuild = 0
        self.n_at_last_rebuild = int(self.G.num_vertices())
        self.alpha = 0
        self.alpha_limit = self._alpha_threshold(self.n_at_last_rebuild)

    def add_edge_updates(self, new_edges, new_vertex_labels=None):
        """
        Add edge updates (can include unseen vertex IDs) and update all maintained structures.

        This is the Algorithm-3 AddNewEdge stage:
        - Update H_t via UpdateSparsifier
        - Update contracted graph via UpdateContractedGraph
        - Unfold small clusters via UnfoldCluster
        """
        self.total_update_steps += 1
        self.update_steps_since_rebuild += 1

        if new_edges is None:
            return {"added_edges": [], "created_vertices": [], "unfolded_clusters": 0}

        if new_vertex_labels is None:
            new_vertex_labels = {}

        if len(new_edges) > 0:
            max_idx = max(max(int(u), int(v)) for u, v in new_edges)
            created_vertices = self._ensure_vertices_exist(max_idx, new_vertex_labels)
        else:
            created_vertices = []

        added_edges = []
        for u, v in new_edges:
            u = int(u)
            v = int(v)
            if u == v:
                continue
            if self.G.edge(u, v) is None:
                self.G.add_edge(u, v)
                added_edges.append((min(u, v), max(u, v)))

        # If only isolated vertices were added, we still need to refresh H and contracted graph sizes.
        if len(added_edges) == 0:
            incoming = np.empty((0, 2), dtype=int)
        else:
            incoming = np.array(added_edges, dtype=int)

        self.sparsifier.update_sparsifier(incoming, verbose=False)
        self.H = self.sparsifier.get_sparsified_graph()

        self.contracted.sparsified_graph = self.H
        self.contracted.update(incoming)

        # Increase rolling alpha by number of actually inserted edges.
        self.alpha += len(added_edges)

        unfolded = self.unfold_small_clusters(self.alpha)

        self.total_added_edges += len(added_edges)

        return {
            "added_edges": added_edges,
            "created_vertices": created_vertices,
            "unfolded_clusters": unfolded,
        }

    def add_random_overlay_updates(self, s):
        sampled = self.sbm.overlay_erdos_renyi_optimized(float(s))
        return self.add_edge_updates(sampled)

    def query(self, num_clusters=None, true_labels=None):
        """
        Algorithm-3 QuerySpecClustering branch:
        - if alpha <= n_r^beta: cluster semi-contracted graph
        - else: cluster sparsifier and globally reconstruct contracted state,
          then reset alpha to 0 for the next epoch.
        """
        if num_clusters is None:
            num_clusters = self.k

        if true_labels is None:
            true_labels = self.true_labels

        threshold = self._query_threshold()
        alpha_before_query = int(self.alpha)
        use_semi_contracted = alpha_before_query <= threshold

        if use_semi_contracted:
            clusters, labels, ari, runtime = self.contracted.spectral_clustering_on_contracted(
                num_clusters, true_labels
            )
            return {
                "source": "semi_contracted",
                "threshold": threshold,
                "updates_since_rebuild": self.update_steps_since_rebuild,
                "alpha": alpha_before_query,
                "alpha_limit": threshold,
                "num_supernodes": len(self.contracted.contracted_node_to_full_node_map),
                "clusters": clusters,
                "labels": labels,
                "ari": ari,
                "runtime": runtime,
            }

        clusters, labels, ari, runtime = run_spectral_clustering(self.H, true_labels, num_clusters)
        self._global_reconstruct(num_clusters)

        return {
            "source": "sparsified",
            "threshold": threshold,
            "updates_since_rebuild": self.update_steps_since_rebuild,
            "alpha": alpha_before_query,
            "alpha_limit": threshold,
            "num_supernodes": len(self.contracted.contracted_node_to_full_node_map),
            "clusters": clusters,
            "labels": labels,
            "ari": ari,
            "runtime": runtime,
        }


def run_better_algorithm3_sbm_demo():
    """Small demo of Algorithm-3 style semi-contracted maintenance on SBM."""
    # Larger graph setup with one comparatively small, low-volume cluster.
    # beta is kept at 0.5 as required.
    sizes = [500,5]
    k = len(sizes)

    dyn = BetterDynamicSpectralClusteringSBM(
        k=k,
        sizes=sizes,
        p=0.4,
        q=0.01,
        beta=0.5,
        sampling_constant=60,
        degree_trigger=0.5,
        seed=11,
    )

    print(
        f"Initial: n={dyn.G.num_vertices()}, m={dyn.G.num_edges()}, "
        f"|H|={dyn.H.num_edges()}, supernodes={len(dyn.contracted.contracted_node_to_full_node_map)}, "
        f"alpha={dyn.alpha}, alpha_limit={dyn.alpha_limit}"
    )

    sep = (
        "+-----+-------+--------+-----------+--------+--------+-----------------+---------+---------------+---------------+"
    )
    header = (
        "|  t  | added | unfold | supernode | alpha  | limit  |   query_graph   |   ari   | query_time(s) | update_time(s) |"
    )
    print(sep)
    print(header)
    print(sep)

    rounds = 120
    edges_per_round = 1
    for t in range(1, rounds + 1):
        n = dyn.G.num_vertices()
        s = edges_per_round / max(1, (n * (n - 1) / 2))

        update_start = time.time()
        update_info = dyn.add_random_overlay_updates(s)
        update_wall = time.time() - update_start

        query_start = time.time()
        res = dyn.query(num_clusters=k)
        query_wall = time.time() - query_start

        print(
            f"| {t:3d} | "
            f"{len(update_info['added_edges']):5d} | "
            f"{update_info['unfolded_clusters']:6d} | "
            f"{res['num_supernodes']:9d} | "
            f"{res['alpha']:6d} | "
            f"{res['alpha_limit']:6d} | "
            f"{res['source']:<15s} | "
            f"{res['ari']:7.4f} | "
            f"{query_wall:13.3f} | "
            f"{update_wall:13.3f} |"
        )
    print(sep)


if __name__ == "__main__":
    run_better_algorithm3_sbm_demo()
