import csv
import os
import time
from datetime import datetime
import graph_tool as gt
import numpy as np
from contractedgraph import Contracted_Graph
from sbm import DynamicSBMGraphMerge
from semicontracted import SemiContractedGraph
from sparsifier import DynamicGraphSparsifier
from sklearn.metrics import adjusted_rand_score
from spectral_clustering import (
    clusters_to_labels,
    run_spectral_clustering,
    spectral_clustering,
)
from hierarchy_contracted import HierarchicalContractedStructure as _HierarchyClass


    
def _norm_edge(u, v):
    u = int(u)
    v = int(v)
    return (u, v) if u <= v else (v, u)


def _edge_set_from_graph(graph):
    edges = np.array(list(graph.get_edges()))
    if edges.size == 0:
        return set()

    out = set()
    for row in edges:
        out.add(_norm_edge(int(row[0]), int(row[1])))
    return out


def _sample_missing_edge_within(nodes, existing_edges, rng, max_attempts=3000):
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size < 2:
        return None

    for _ in range(max_attempts):
        u, v = rng.choice(nodes, size=2, replace=False)
        edge = _norm_edge(int(u), int(v))
        if edge[0] != edge[1] and edge not in existing_edges:
            return edge
    return None


def _sample_missing_edge_between(nodes_a, nodes_b, existing_edges, rng, max_attempts=3000):
    nodes_a = np.asarray(nodes_a, dtype=int)
    nodes_b = np.asarray(nodes_b, dtype=int)
    if nodes_a.size == 0 or nodes_b.size == 0:
        return None

    for _ in range(max_attempts):
        u = int(rng.choice(nodes_a))
        v = int(rng.choice(nodes_b))
        if u == v:
            continue
        edge = _norm_edge(u, v)
        if edge not in existing_edges:
            return edge
    return None


def build_single_edge_update_stream(base_graph, true_labels, num_updates, seed):
    """
    Build one deterministic stream of unique single-edge insertions.

    Stream is skewed toward the largest cluster so alpha increases while
    preserving overall cluster structure.
    """
    rng = np.random.default_rng(seed)
    existing_edges = _edge_set_from_graph(base_graph)

    labels = np.array(true_labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_label = int(unique_labels[np.argmax(counts)])
    smallest_label = int(unique_labels[np.argmin(counts)])

    largest_nodes = np.where(labels == largest_label)[0]
    smallest_nodes = np.where(labels == smallest_label)[0]
    all_nodes = np.arange(base_graph.num_vertices())

    updates = []
    for _ in range(int(num_updates)):
        r = float(rng.random())
        if r < 0.60:
            edge = _sample_missing_edge_within(largest_nodes, existing_edges, rng)
        elif r < 0.90:
            edge = _sample_missing_edge_between(largest_nodes, smallest_nodes, existing_edges, rng)
        else:
            edge = _sample_missing_edge_within(all_nodes, existing_edges, rng)

        if edge is None:
            edge = _sample_missing_edge_within(all_nodes, existing_edges, rng, max_attempts=10000)

        if edge is None:
            break

        existing_edges.add(edge)
        updates.append(edge)

    return updates


def _chunk_updates(updates, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [updates[i : i + batch_size] for i in range(0, len(updates), batch_size)]


class LS24Runner:
    """LS24 algorithm: sparsifier + contracted graph."""

    def __init__(self, graph, true_labels, k, sampling_constant=80, degree_trigger=0.5, gamma=2.0):
        self.graph = graph
        self.true_labels = np.array(true_labels)
        self.k = int(k)
        self.gamma = float(gamma)
        self.degree_trigger = float(degree_trigger)

        self.sparsifier = DynamicGraphSparsifier(self.graph, sampling_constant=sampling_constant)
        self.sparsifier.create_sparsifier()
        self.H = self.sparsifier.get_sparsified_graph()

        init_clusters = spectral_clustering(self.H, self.k)
        init_labels = clusters_to_labels(init_clusters, num_data_points=self.H.num_vertices())
        self.contracted = Contracted_Graph(self.graph, self.H, degree_trigger=self.degree_trigger)
        self.contracted.initialize(init_clusters, np.array(init_labels))

        self.updates_since_rebuild = 0
        self.n_at_last_rebuild = int(self.H.num_vertices())

    def _threshold(self):
        n = max(2, int(self.n_at_last_rebuild))
        return max(1, int(np.ceil(np.log(n) ** self.gamma)))

    def add_edge_batch(self, edges):
        applied = []
        for edge in edges:
            u, v = _norm_edge(edge[0], edge[1])
            if u == v:
                continue
            if self.graph.edge(u, v) is None:
                self.graph.add_edge(u, v)
                applied.append((u, v))

        if len(applied) == 0:
            return {"num_applied": 0}

        incoming = np.array(applied, dtype=int)
        self.sparsifier.update_sparsifier(incoming, verbose=False)
        self.H = self.sparsifier.get_sparsified_graph()
        self.contracted.sparsified_graph = self.H
        self.contracted.update(incoming)

        self.updates_since_rebuild += len(applied)
        return {"num_applied": len(applied)}

    def _rebuild_from_sparsifier(self):
        clusters = spectral_clustering(self.H, self.k)
        labels = clusters_to_labels(clusters, num_data_points=self.H.num_vertices())
        self.contracted = Contracted_Graph(self.graph, self.H, degree_trigger=self.degree_trigger)
        self.contracted.initialize(clusters, np.array(labels))
        self.updates_since_rebuild = 0
        self.n_at_last_rebuild = int(self.H.num_vertices())

    def query(self):
        threshold = self._threshold()
        use_contracted = self.updates_since_rebuild <= threshold

        if use_contracted:
            _, _, ari, runtime = self.contracted.spectral_clustering_on_contracted(
                self.k, self.true_labels
            )
            return {
                "source": "contracted",
                "ari": float(ari),
                "runtime": float(runtime),
                "threshold": threshold,
                "updates_since_rebuild": self.updates_since_rebuild,
            }

        _, _, ari, runtime = run_spectral_clustering(self.H, self.true_labels, self.k)
        self._rebuild_from_sparsifier()
        return {
            "source": "sparsified",
            "ari": float(ari),
            "runtime": float(runtime),
            "threshold": threshold,
            "updates_since_rebuild": self.updates_since_rebuild,
        }


class SemiContractedRunner:
    """Semi-contracted algorithm with batch alpha updates."""

    def __init__(self, graph, true_labels, k, beta=0.5, sampling_constant=80, degree_trigger=0.5):
        self.graph = graph
        self.true_labels = np.array(true_labels)
        self.k = int(k)
        self.beta = float(beta)

        self.sparsifier = DynamicGraphSparsifier(self.graph, sampling_constant=sampling_constant)
        self.sparsifier.create_sparsifier()

        self.semi = SemiContractedGraph(
            original_graph=self.graph,
            sparsifier=self.sparsifier,
            degree_trigger=degree_trigger,
        )
        self.semi.full_reconstruct(num_clusters=self.k)
        self.n_at_last_rebuild = int(self.graph.num_vertices())

    def _alpha_limit(self):
        return max(1, int(np.ceil(max(1, self.n_at_last_rebuild) ** self.beta)))

    def add_edge_batch(self, edges):
        result = self.semi.update(new_edges=edges)
        return {
            "num_applied": int(result.get("num_applied", 0)),
            "update_runtime_s": float(result.get("update_runtime_s", 0.0)),
            "unfold_runtime_s": float(result.get("unfold_runtime_s", 0.0)),
        }

    def query(self):
        alpha_limit = self._alpha_limit()
        alpha_now = int(self.semi.alpha)
        use_semi_contracted = alpha_now <= alpha_limit

        if use_semi_contracted:
            _, _, ari, runtime = self.semi.spectral_clustering_on_semicontracted(
                self.k, self.true_labels
            )
            return {
                "source": "semi_contracted",
                "ari": float(ari),
                "runtime": float(runtime),
                "alpha": alpha_now,
                "alpha_limit": alpha_limit,
            }

        _, _, ari, runtime = run_spectral_clustering(
            self.semi.sparsified_graph, self.true_labels, self.k
        )
        self.semi.full_reconstruct(num_clusters=self.k)
        self.n_at_last_rebuild = int(self.graph.num_vertices())
        return {
            "source": "sparsified",
            "ari": float(ari),
            "runtime": float(runtime),
            "alpha": alpha_now,
            "alpha_limit": alpha_limit,
        }


class HierarchyRunner:
    """Hierarchy algorithm using the single-sparsifier hierarchy implementation."""

    def __init__(
        self,
        graph,
        true_labels,
        k,
        num_levels=3,
        sampling_constant=80,
        degree_trigger=0.5,
    ):
        self.graph = graph
        self.true_labels = np.array(true_labels)
        self.k = int(k)
        self.hier = _HierarchyClass(
            base_graph=self.graph,
            num_clusters=int(self.k),
            num_levels=int(num_levels),
            sampling_constant=int(sampling_constant),
            degree_trigger=float(degree_trigger),
        )

    def add_edge_batch(self, edges):
        info = self.hier.update_hierarchy(edges)
        return {"num_applied": int(info.get("num_applied", 0))}

    def query(self):
        if self.hier.active_levels <= 0:
            return {"ari": float("nan"), "runtime": 0.0}

        state = self.hier.levels[self.hier.active_levels - 1]
        if state is None:
            return {"ari": float("nan"), "runtime": 0.0}

        cg = state.semi.contracted.contracted_graph
        if int(cg.num_vertices()) == 0:
            return {"ari": float("nan"), "runtime": 0.0}

        k_eff = max(1, min(int(self.k), int(cg.num_vertices())))

        # Query runtime tracks only spectral clustering on deepest contracted graph.
        t0 = time.perf_counter()
        clusters_level = spectral_clustering(cg, k_eff)
        runtime = time.perf_counter() - t0

        base_clusters = []
        for cluster in clusters_level:
            base_nodes = set()
            for node in cluster:
                idx = int(node)
                if 0 <= idx < len(state.base_node_map):
                    base_nodes.update(state.base_node_map[idx])
            base_clusters.append(sorted(base_nodes))

        labels_pred = clusters_to_labels(
            base_clusters, num_data_points=int(self.hier.base_graph.num_vertices())
        )
        m = min(len(self.true_labels), len(labels_pred))
        if m <= 0:
            ari = float("nan")
        else:
            ari = float(
                adjusted_rand_score(
                    np.array(self.true_labels[:m]), np.array(labels_pred[:m])
                )
            )

        return {"ari": float(ari), "runtime": float(runtime)}


def run_sbm_batch_comparison():
    # 4 clusters, total n=5000
    sizes = [5000, 1500, 1000, 2500]
    k = len(sizes)
    p = 0.06
    q = 0.01
    num_updates = 2000
    batch_size = 50
    seed_graph = 7
    seed_updates = 19

    sampling_constant = 80
    degree_trigger = 0.5
    gamma = 2.0
    beta = 0.5
    num_levels = 2

    np.random.seed(seed_graph)
    base_sbm = DynamicSBMGraphMerge(k=k, sizes=sizes, p=p, q=q)
    base_graph = base_sbm.get_graph()
    true_labels = np.array(base_sbm.get_labels())

    updates = build_single_edge_update_stream(
        base_graph=base_graph,
        true_labels=true_labels,
        num_updates=num_updates,
        seed=seed_updates,
    )
    if len(updates) == 0:
        raise RuntimeError("No updates could be generated. Try different parameters.")

    batches = _chunk_updates(updates, batch_size=batch_size)

    graph_ls24 = gt.Graph(base_graph)
    graph_semi = gt.Graph(base_graph)
    graph_hier = gt.Graph(base_graph)

    ls24 = LS24Runner(
        graph=graph_ls24,
        true_labels=true_labels,
        k=k,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        gamma=gamma,
    )
    semi = SemiContractedRunner(
        graph=graph_semi,
        true_labels=true_labels,
        k=k,
        beta=beta,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
    )
    hier = HierarchyRunner(
        graph=graph_hier,
        true_labels=true_labels,
        k=k,
        num_levels=num_levels,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
    )

    print("Running comparison on SBM (LS24 vs SemiContracted vs Hierarchy)")
    print(f"  sizes={sizes}, p={p}, q={q}")
    print(
        f"  total_updates={len(updates)}, batch_size={batch_size}, num_batches={len(batches)}, "
        f"num_levels={num_levels}"
    )
    print()

    sep = "+-------------+----------+----------------------+---------------------+----------+----------------------+---------------------+----------+----------------------+---------------------+"
    header = (
        "| num_updates | ls24_ari  | ls24_cum_update_s  | ls24_cum_query_s  | semi_ari  | semi_cum_update_s  | semi_cum_query_s  | hier_ari  | hier_cum_update_s  | hier_cum_query_s  |"
    )
    print(sep)
    print(header)
    print(sep)

    rows = []
    ls24_update_cum = 0.0
    semi_update_cum = 0.0
    hier_update_cum = 0.0
    ls24_query_cum = 0.0
    semi_query_cum = 0.0
    hier_query_cum = 0.0
    processed_updates = 0

    for batch in batches:
        t0 = time.perf_counter()
        ls24.add_edge_batch(batch)
        t_update_ls24 = time.perf_counter() - t0
        ls24_update_cum += t_update_ls24

        ls24_q = ls24.query()
        t_query_ls24 = float(ls24_q.get("runtime", 0.0))
        ls24_query_cum += t_query_ls24

        t0 = time.perf_counter()
        semi.add_edge_batch(batch)
        t_update_semi = time.perf_counter() - t0
        semi_update_cum += t_update_semi

        semi_q = semi.query()
        t_query_semi = float(semi_q.get("runtime", 0.0))
        semi_query_cum += t_query_semi

        t0 = time.perf_counter()
        hier.add_edge_batch(batch)
        t_update_hier = time.perf_counter() - t0
        hier_update_cum += t_update_hier

        hier_q = hier.query()
        t_query_hier = float(hier_q.get("runtime", 0.0))
        hier_query_cum += t_query_hier

        processed_updates += len(batch)
        ls24_ari = float(ls24_q["ari"])
        semi_ari = float(semi_q["ari"])
        hier_ari = float(hier_q["ari"])

        rows.append(
            {
                "num_updates": int(processed_updates),
                "ls24_ari": float(ls24_ari),
                "semi_ari": float(semi_ari),
                "hierarchy_ari": float(hier_ari),
                "ls24_cumulative_update_runtime_s": float(ls24_update_cum),
                "semi_cumulative_update_runtime_s": float(semi_update_cum),
                "hierarchy_cumulative_update_runtime_s": float(hier_update_cum),
                "ls24_cumulative_query_runtime_s": float(ls24_query_cum),
                "semi_cumulative_query_runtime_s": float(semi_query_cum),
                "hierarchy_cumulative_query_runtime_s": float(hier_query_cum),
            }
        )

        print(
            f"| {processed_updates:11d} | "
            f"{ls24_ari:8.4f} | "
            f"{ls24_update_cum:20.4f} | "
            f"{ls24_query_cum:19.4f} | "
            f"{semi_ari:8.4f} | "
            f"{semi_update_cum:20.4f} | "
            f"{semi_query_cum:19.4f} | "
            f"{hier_ari:8.4f} | "
            f"{hier_update_cum:20.4f} | "
            f"{hier_query_cum:19.4f} |"
        )

    print(sep)
    print("Summary")
    print(f"  final_num_updates={rows[-1]['num_updates']}")
    print(f"  final_ls24_ari={rows[-1]['ls24_ari']:.4f}")
    print(f"  final_semi_ari={rows[-1]['semi_ari']:.4f}")
    print(f"  final_hierarchy_ari={rows[-1]['hierarchy_ari']:.4f}")
    print(f"  final_ls24_cumulative_update_runtime_s={rows[-1]['ls24_cumulative_update_runtime_s']:.3f}")
    print(f"  final_semi_cumulative_update_runtime_s={rows[-1]['semi_cumulative_update_runtime_s']:.3f}")
    print(f"  final_hierarchy_cumulative_update_runtime_s={rows[-1]['hierarchy_cumulative_update_runtime_s']:.3f}")
    print(f"  final_ls24_cumulative_query_runtime_s={rows[-1]['ls24_cumulative_query_runtime_s']:.3f}")
    print(f"  final_semi_cumulative_query_runtime_s={rows[-1]['semi_cumulative_query_runtime_s']:.3f}")
    print(f"  final_hierarchy_cumulative_query_runtime_s={rows[-1]['hierarchy_cumulative_query_runtime_s']:.3f}")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        results_dir,
        (
            f"compare_ls24_vs_semicontracted_vs_hierarchy_sbm_"
            f"n_{sum(sizes)}_k_{k}_updates_{len(updates)}_batch_{batch_size}_{timestamp}.csv"
        ),
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  csv={out_path}")


if __name__ == "__main__":
    run_sbm_batch_comparison()
