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
from spectral_clustering import (
    clusters_to_labels,
    run_spectral_clustering,
    spectral_clustering,
)


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


def run_sbm_batch_comparison():
    # 4 clusters, total n=5000
    sizes = [5000, 1500, 1000, 2500]
    k = len(sizes)
    p = 0.04
    q = 0.01
    num_updates = 5000
    batch_size = 50
    seed_graph = 7
    seed_updates = 19

    sampling_constant = 80
    degree_trigger = 0.5
    gamma = 2.0
    beta = 0.5

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

    print("Running comparison on SBM (5000 nodes, 4 clusters, batch updates)")
    print(f"  sizes={sizes}, p={p}, q={q}")
    print(f"  total_updates={len(updates)}, batch_size={batch_size}, num_batches={len(batches)}")
    print()

    sep = "+-------------+------------+----------------+----------------+----------------+----------------+----------+----------+"
    header = "| num_updates | batch_size |     ls24_time |    semi_time   |  ls24_upd_time |  semi_upd_time | ls24_ari | semi_ari |"
    print(sep)
    print(header)
    print(sep)

    rows = []
    ls24_cum = 0.0
    semi_cum = 0.0
    ls24_update_cum = 0.0
    semi_update_cum = 0.0
    processed_updates = 0

    for batch_idx, batch in enumerate(batches, start=1):
        t0 = time.time()
        ls24_update = ls24.add_edge_batch(batch)
        t_update_ls24 = time.time() - t0
        ls24_update_cum += t_update_ls24

        t0 = time.time()
        ls24_q = ls24.query()
        t_query_ls24 = time.time() - t0
        ls24_step_time = t_update_ls24 + t_query_ls24
        ls24_cum += ls24_step_time

        semi_update = semi.add_edge_batch(batch)
        t_update_semi = float(semi_update.get("update_runtime_s", 0.0))
        semi_update_cum += t_update_semi

        t0 = time.time()
        semi_q = semi.query()
        t_query_semi = time.time() - t0
        semi_step_time = t_update_semi + t_query_semi
        semi_cum += semi_step_time

        processed_updates += len(batch)

        rows.append(
            {
                "batch": batch_idx,
                "num_updates": int(processed_updates),
                "batch_size": int(len(batch)),
                "ls24_applied": int(ls24_update["num_applied"]),
                "semi_applied": int(semi_update["num_applied"]),
                "ls24_source": ls24_q["source"],
                "semi_source": semi_q["source"],
                "ls24_ari": float(ls24_q["ari"]),
                "semi_ari": float(semi_q["ari"]),
                "ls24_step_runtime_s": float(ls24_step_time),
                "semi_step_runtime_s": float(semi_step_time),
                "ls24_cumulative_runtime_s": float(ls24_cum),
                "semi_cumulative_runtime_s": float(semi_cum),
                "ls24_cumulative_update_runtime_s": float(ls24_update_cum),
                "semi_cumulative_update_runtime_s": float(semi_update_cum),
                "ls24_updates_since_rebuild": int(ls24_q["updates_since_rebuild"]),
                "ls24_threshold": int(ls24_q["threshold"]),
                "semi_alpha": int(semi_q["alpha"]),
                "semi_alpha_limit": int(semi_q["alpha_limit"]),
            }
        )

        print(
            f"| {processed_updates:11d} | "
            f"{len(batch):10d} | "
            f"{ls24_cum:14.3f} | "
            f"{semi_cum:14.3f} | "
            f"{ls24_update_cum:14.3f} | "
            f"{semi_update_cum:14.3f} | "
            f"{ls24_q['ari']:8.4f} | "
            f"{semi_q['ari']:8.4f} |"
        )

    print(sep)
    print("Summary")
    print(f"  final_ls24_ari={rows[-1]['ls24_ari']:.4f}")
    print(f"  final_semi_ari={rows[-1]['semi_ari']:.4f}")
    print(f"  final_ls24_cumulative_runtime_s={rows[-1]['ls24_cumulative_runtime_s']:.3f}")
    print(f"  final_semi_cumulative_runtime_s={rows[-1]['semi_cumulative_runtime_s']:.3f}")
    print(f"  final_ls24_cumulative_update_runtime_s={rows[-1]['ls24_cumulative_update_runtime_s']:.3f}")
    print(f"  final_semi_cumulative_update_runtime_s={rows[-1]['semi_cumulative_update_runtime_s']:.3f}")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        results_dir,
        (
            f"compare_ls24_vs_semicontracted_sbm_"
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
