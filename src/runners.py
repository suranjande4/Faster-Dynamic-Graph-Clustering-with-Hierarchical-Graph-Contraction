import csv
import gc
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


def _sample_missing_edge_between(
    nodes_a, nodes_b, existing_edges, rng, max_attempts=3000
):
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
            edge = _sample_missing_edge_between(
                largest_nodes, smallest_nodes, existing_edges, rng
            )
        else:
            edge = _sample_missing_edge_within(all_nodes, existing_edges, rng)

        if edge is None:
            edge = _sample_missing_edge_within(
                all_nodes, existing_edges, rng, max_attempts=10000
            )

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

    def __init__(
        self,
        graph,
        true_labels,
        k,
        sampling_constant=80,
        degree_trigger=0.5,
        gamma=2.0,
        query_policy="algorithm",
    ):
        self.graph = graph
        self.true_labels = np.array(true_labels)
        self.k = int(k)
        self.gamma = float(gamma)
        self.degree_trigger = float(degree_trigger)
        self.query_policy = str(query_policy)
        if self.query_policy not in {"algorithm", "contracted", "sparsified"}:
            raise ValueError(
                "query_policy must be 'algorithm', 'contracted', or 'sparsified'."
            )

        self.sparsifier = DynamicGraphSparsifier(
            self.graph, sampling_constant=sampling_constant
        )
        self.sparsifier.create_sparsifier()
        self.H = self.sparsifier.get_sparsified_graph()

        init_clusters = spectral_clustering(self.H, self.k)
        init_labels = clusters_to_labels(
            init_clusters, num_data_points=self.H.num_vertices()
        )
        self.contracted = Contracted_Graph(
            self.graph, self.H, degree_trigger=self.degree_trigger
        )
        self.contracted.initialize(init_clusters, np.array(init_labels))

        self.updates_since_rebuild = 0
        self.n_at_last_rebuild = int(self.H.num_vertices())

    def _threshold(self):
        n = max(2, int(self.n_at_last_rebuild))
        return max(1, int(np.ceil(np.log(n) ** self.gamma)))

    def add_edge_batch(self, edges):
        raw = np.asarray(list(edges))
        if raw.size == 0:
            return {"num_applied": 0}
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        max_vertex = int(np.max(raw[:, :2]))
        current_n = int(self.graph.num_vertices())
        if max_vertex >= current_n:
            self.graph.add_vertex(max_vertex + 1 - current_n)

        # Aggregate repeated logical insertions while retaining their mass.
        incoming = {}
        for edge in raw:
            u, v = _norm_edge(edge[0], edge[1])
            if u == v:
                continue
            weight = float(edge[2]) if raw.shape[1] >= 3 else 1.0
            incoming[(u, v)] = incoming.get((u, v), 0.0) + weight

        has_weights = "edge_weight" in self.graph.edge_properties
        weight_prop = self.graph.ep["edge_weight"] if has_weights else None
        applied = []
        for (u, v), weight in incoming.items():
            if self.graph.edge(u, v) is None:
                new_edge = self.graph.add_edge(u, v)
                if has_weights:
                    weight_prop[new_edge] = weight
                applied.append((u, v, weight))

        if len(applied) == 0:
            return {"num_applied": 0}

        applied_array = np.asarray(applied, dtype=float)
        self.sparsifier.update_sparsifier(applied_array, verbose=False)
        self.H = self.sparsifier.get_sparsified_graph()
        next_updates_since_rebuild = self.updates_since_rebuild + len(applied)
        threshold = self._threshold()
        maintained_contracted = self.query_policy == "contracted" or (
            self.query_policy == "algorithm" and next_updates_since_rebuild <= threshold
        )
        if maintained_contracted:
            self.contracted.sparsified_graph = self.H
            self.contracted.update(applied_array)

        self.updates_since_rebuild = next_updates_since_rebuild
        return {
            "num_applied": len(applied),
            "threshold": threshold,
            "updates_since_rebuild": self.updates_since_rebuild,
            "maintained_contracted": maintained_contracted,
        }

    def _rebuild_from_sparsifier(self):
        clusters = spectral_clustering(self.H, self.k)
        labels = clusters_to_labels(clusters, num_data_points=self.H.num_vertices())
        self.contracted = Contracted_Graph(
            self.graph, self.H, degree_trigger=self.degree_trigger
        )
        self.contracted.initialize(clusters, np.array(labels))
        self.updates_since_rebuild = 0
        self.n_at_last_rebuild = int(self.H.num_vertices())

    def reconstruct_from_sparsifier(self):
        """Reconstruct the contracted graph once from the current H.

        This public operation mirrors the class-boundary reconstruction used
        by the released Laenen--Sun MNIST/EMNIST experiment driver.
        """
        self._rebuild_from_sparsifier()

    def query(self):
        if self.query_policy == "contracted":
            _, labels, ari, runtime = self.contracted.spectral_clustering_on_contracted(
                self.k, self.true_labels
            )
            return {
                "source": "contracted",
                "labels": np.asarray(labels),
                "ari": float(ari),
                "runtime": float(runtime),
                "threshold": self._threshold(),
                "updates_since_rebuild": self.updates_since_rebuild,
            }

        if self.query_policy == "sparsified":
            _, labels, ari, runtime = run_spectral_clustering(
                self.H, self.true_labels, self.k
            )
            return {
                "source": "sparsified",
                "labels": np.asarray(labels),
                "ari": float(ari),
                "runtime": float(runtime),
                "threshold": self._threshold(),
                "updates_since_rebuild": self.updates_since_rebuild,
            }

        threshold = self._threshold()
        use_contracted = self.updates_since_rebuild <= threshold

        if use_contracted:
            _, labels, ari, runtime = self.contracted.spectral_clustering_on_contracted(
                self.k, self.true_labels
            )
            return {
                "source": "contracted",
                "labels": np.asarray(labels),
                "ari": float(ari),
                "runtime": float(runtime),
                "threshold": threshold,
                "updates_since_rebuild": self.updates_since_rebuild,
            }

        updates_before_rebuild = self.updates_since_rebuild
        _, labels, ari, runtime = run_spectral_clustering(
            self.H, self.true_labels, self.k
        )
        rebuild_start = time.perf_counter()
        self._rebuild_from_sparsifier()
        rebuild_runtime = time.perf_counter() - rebuild_start
        return {
            "source": "sparsified",
            "labels": np.asarray(labels),
            "ari": float(ari),
            "runtime": float(runtime),
            "rebuild_runtime_s_excluded": float(rebuild_runtime),
            "threshold": threshold,
            "updates_since_rebuild": updates_before_rebuild,
        }


class SemiContractedRunner:
    """Semi-contracted algorithm with batch alpha updates."""

    def __init__(
        self,
        graph,
        true_labels,
        k,
        beta=1.0,
        sampling_constant=80,
        degree_trigger=0.5,
        query_policy="algorithm",
        verbose=False,
    ):
        self.graph = graph
        self.true_labels = np.array(true_labels)
        self.k = int(k)
        self.beta = float(beta)
        self.query_policy = str(query_policy)
        if self.query_policy not in {"algorithm", "semicontracted", "sparsified"}:
            raise ValueError(
                "query_policy must be 'algorithm', 'semicontracted', or 'sparsified'."
            )
        self.verbose = bool(verbose)

        if self.verbose:
            print("    semi: creating sparsifier...", flush=True)
        t0 = time.perf_counter()
        self.sparsifier = DynamicGraphSparsifier(
            self.graph, sampling_constant=sampling_constant
        )
        self.sparsifier.create_sparsifier()
        if self.verbose:
            print(
                f"    semi: sparsifier ready in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

        if self.verbose:
            print("    semi: creating wrapper...", flush=True)
        t0 = time.perf_counter()
        self.semi = SemiContractedGraph(
            original_graph=self.graph,
            sparsifier=self.sparsifier,
            degree_trigger=degree_trigger,
            track_existing_edges=False,
        )
        if self.verbose:
            print(
                f"    semi: wrapper ready in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

        if self.verbose:
            print(
                "    semi: full reconstruction / initial spectral clustering...",
                flush=True,
            )
        t0 = time.perf_counter()
        self.semi.full_reconstruct(num_clusters=self.k)
        if self.verbose:
            print(
                f"    semi: reconstruction ready in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        self.n_at_last_rebuild = int(self.graph.num_vertices())

    def _alpha_limit(self):
        return max(1, int(np.ceil(max(1, self.n_at_last_rebuild) ** self.beta)))

    def add_edge_batch(self, edges):
        raw = list(edges)
        next_alpha = int(self.semi.alpha) + len(raw)
        alpha_limit = self._alpha_limit()
        maintain_contracted = self.query_policy == "semicontracted" or (
            self.query_policy == "algorithm" and next_alpha <= alpha_limit
        )
        result = self.semi.update(
            new_edges=raw,
            maintain_contracted=maintain_contracted,
        )
        return {
            "num_applied": int(result.get("num_applied", 0)),
            "update_runtime_s": float(result.get("update_runtime_s", 0.0)),
            "unfold_runtime_s": float(result.get("unfold_runtime_s", 0.0)),
            "alpha": int(result.get("alpha", self.semi.alpha)),
            "alpha_limit": alpha_limit,
            "maintained_contracted": bool(result.get("maintained_contracted", False)),
        }

    def query(self):
        alpha_limit = self._alpha_limit()
        alpha_now = int(self.semi.alpha)
        use_semi_contracted = self.query_policy == "semicontracted" or (
            self.query_policy == "algorithm" and alpha_now <= alpha_limit
        )

        if use_semi_contracted:
            _, labels, ari, runtime = self.semi.spectral_clustering_on_semicontracted(
                self.k, self.true_labels
            )
            return {
                "source": "semi_contracted",
                "labels": np.asarray(labels),
                "ari": float(ari),
                "runtime": float(runtime),
                "alpha": alpha_now,
                "alpha_limit": alpha_limit,
            }

        if self.query_policy == "sparsified":
            _, labels, ari, runtime = run_spectral_clustering(
                self.semi.sparsified_graph, self.true_labels, self.k
            )
            return {
                "source": "sparsified",
                "labels": np.asarray(labels),
                "ari": float(ari),
                "runtime": float(runtime),
                "alpha": alpha_now,
                "alpha_limit": alpha_limit,
            }

        _, labels, ari, runtime = run_spectral_clustering(
            self.semi.sparsified_graph, self.true_labels, self.k
        )
        rebuild_start = time.perf_counter()
        self.semi.full_reconstruct(num_clusters=self.k)
        rebuild_runtime = time.perf_counter() - rebuild_start
        self.n_at_last_rebuild = int(self.graph.num_vertices())
        return {
            "source": "sparsified",
            "labels": np.asarray(labels),
            "ari": float(ari),
            "runtime": float(runtime),
            "rebuild_runtime_s_excluded": float(rebuild_runtime),
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
        beta=0.5,
        scheduler_update_unit="edge",
        verbose=False,
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
            semicontracted_beta=float(beta),
            batch_update_mode=True,
            scheduler_update_unit=str(scheduler_update_unit),
            verbose=bool(verbose),
        )

    def add_edge_batch(self, edges):
        self.hier.k = int(self.k)
        info = self.hier.update_hierarchy(edges)
        return {
            "num_applied": int(info.get("num_applied", 0)),
            "phase_k": info.get("phase_k"),
            "rebuilt_levels": list(info.get("rebuilt_levels", [])),
            "restarted": bool(info.get("restarted", False)),
            "updates_since_t_prime": int(info.get("updates_since_t_prime", 0)),
            "n_at_t_prime": int(info.get("n_at_t_prime", 0)),
        }

    def query(self):
        self.hier.k = int(self.k)
        if self.hier.active_levels <= 0:
            return {"ari": float("nan"), "runtime": 0.0}

        self.hier._ensure_level_base_map(self.hier.active_levels)
        state = self.hier.levels[self.hier.active_levels - 1]
        if state is None:
            return {"ari": float("nan"), "runtime": 0.0}

        alpha_now = int(state.semi.alpha)
        alpha_limit = self.hier._level_alpha_limit(state)
        use_sparsified = alpha_now > alpha_limit
        fallback_reason = "alpha" if use_sparsified else None

        def _map_covers_current_base(node_map):
            base_n = int(self.hier.base_graph.num_vertices())
            if base_n <= 0:
                return True
            covered = np.zeros(base_n, dtype=bool)
            for group in node_map:
                for v in group:
                    iv = int(v)
                    if 0 <= iv < base_n:
                        covered[iv] = True
            return bool(np.all(covered))

        if not use_sparsified:
            self.hier._ensure_level_base_map(self.hier.active_levels)
            state = self.hier.levels[self.hier.active_levels - 1]
            if state is None:
                return {"ari": float("nan"), "runtime": 0.0}
            if not _map_covers_current_base(state.base_node_map):
                use_sparsified = True
                fallback_reason = "stale_contracted_map"

        query_graph = (
            state.semi.sparsified_graph
            if use_sparsified
            else state.semi.contracted.contracted_graph
        )
        if int(query_graph.num_vertices()) == 0:
            return {"ari": float("nan"), "runtime": 0.0}

        k_eff = max(1, min(int(self.k), int(query_graph.num_vertices())))

        # Query runtime tracks only spectral clustering. Any fallback rebuild is
        # reported separately and charged to the next update by the experiment.
        t0 = time.perf_counter()
        clusters_level = spectral_clustering(query_graph, k_eff)
        runtime = time.perf_counter() - t0

        base_clusters = []
        node_map = state.input_base_node_map if use_sparsified else state.base_node_map
        for cluster in clusters_level:
            base_nodes = set()
            for node in cluster:
                idx = int(node)
                if 0 <= idx < len(node_map):
                    base_nodes.update(node_map[idx])
            base_clusters.append(sorted(base_nodes))

        rebuild_runtime = 0.0
        if use_sparsified:
            rebuild_start = time.perf_counter()
            state.semi.full_reconstruct(
                num_clusters=self.hier._cluster_count_for_graph(
                    self.k, state.semi.sparsified_graph
                )
            )
            state.base_node_map = self.hier._compose_map(
                state.semi.contracted.contracted_node_to_full_node_map,
                state.input_base_node_map,
            )
            state.base_map_dirty = False
            state.input_map_dirty = True
            state.base_vertex_to_input = None
            rebuild_runtime = time.perf_counter() - rebuild_start

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

        return {
            "source": (
                f"H_C_{self.hier.active_levels}"
                if use_sparsified
                else f"C_{self.hier.active_levels}"
            ),
            "fallback_reason": fallback_reason,
            "labels": np.asarray(labels_pred),
            "ari": float(ari),
            "runtime": float(runtime),
            "rebuild_runtime_s_excluded": float(rebuild_runtime),
            "alpha": alpha_now,
            "alpha_limit": alpha_limit,
        }


def run_sbm_batch_comparison(
    sizes=None,
    p=0.09,
    q=0.01,
    num_updates=5000,
    batch_size=100,
    seed_graph=7,
    seed_updates=19,
    seed_algo=123,
    sampling_constant=80,
    degree_trigger=0.5,
    gamma=2.0,
    beta=0.5,
    num_levels=3,
    num_runs=5,
    results_dir="results",
):
    # Default: 4 clusters, total n=10000
    if sizes is None:
        sizes = [5000, 1500, 1000, 2500]

    sizes = [int(x) for x in sizes]
    k = len(sizes)
    if k <= 0:
        raise ValueError("sizes must contain at least one cluster size.")

    p = float(p)
    q = float(q)
    num_updates = int(num_updates)
    batch_size = int(batch_size)
    seed_graph = int(seed_graph)
    seed_updates = int(seed_updates)
    seed_algo = int(seed_algo)
    sampling_constant = int(sampling_constant)
    degree_trigger = float(degree_trigger)
    gamma = float(gamma)
    beta = float(beta)
    num_levels = int(num_levels)
    num_runs = int(num_runs)

    if num_updates <= 0:
        raise ValueError("num_updates must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if num_runs <= 0:
        raise ValueError("num_runs must be >= 1.")

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

    print("Running comparison on SBM (LS24 vs SemiContracted vs Hierarchy)")
    print(f"  sizes={sizes}, p={p}, q={q}")
    print(
        f"  total_updates={len(updates)}, batch_size={batch_size}, num_batches={len(batches)}, "
        f"num_levels={num_levels}, num_runs={num_runs}"
    )
    print()
    all_rows = []
    final_rows = []

    for run_idx in range(num_runs):
        np.random.seed(seed_algo + run_idx)

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

        print(
            f"[run {run_idx + 1}/{num_runs}] starting from the same initial graph + update stream"
        )
        print(
            "+-------------+----------+------------+-------------+----------+------------+-------------+----------+------------+-------------+"
        )
        print(
            "| num_updates | ls24_ari | ls24_upd_s | ls24_total_s | semi_ari | semi_upd_s | semi_total_s | hier_ari | hier_upd_s | hier_total_s |"
        )
        print(
            "+-------------+----------+------------+-------------+----------+------------+-------------+----------+------------+-------------+"
        )

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
            ls24_total_cum = float(ls24_update_cum + ls24_query_cum)
            semi_total_cum = float(semi_update_cum + semi_query_cum)
            hier_total_cum = float(hier_update_cum + hier_query_cum)

            row = {
                "num_updates": int(processed_updates),
                "ls24_ari": float(ls24_ari),
                "semi_ari": float(semi_ari),
                "hierarchy_ari": float(hier_ari),
                "ls24_cumulative_update_runtime_s": float(ls24_update_cum),
                "semi_cumulative_update_runtime_s": float(semi_update_cum),
                "hierarchy_cumulative_update_runtime_s": float(hier_update_cum),
                "ls24_cumulative_total_runtime_s": float(ls24_total_cum),
                "semi_cumulative_total_runtime_s": float(semi_total_cum),
                "hierarchy_cumulative_total_runtime_s": float(hier_total_cum),
            }
            rows.append(row)

            row_out = {"run": int(run_idx + 1)}
            row_out.update(row)
            all_rows.append(row_out)

            print(
                f"| {processed_updates:11d} | {ls24_ari:8.4f} | {ls24_update_cum:10.4f} | {ls24_total_cum:11.4f} | "
                f"{semi_ari:8.4f} | {semi_update_cum:10.4f} | {semi_total_cum:11.4f} | "
                f"{hier_ari:8.4f} | {hier_update_cum:10.4f} | {hier_total_cum:11.4f} |"
            )

        if len(rows) == 0:
            raise RuntimeError(f"Run {run_idx + 1} produced no rows.")

        print(
            "+-------------+----------+------------+-------------+----------+------------+-------------+----------+------------+-------------+"
        )
        print(
            f"[run {run_idx + 1}] final_ls24_ari={rows[-1]['ls24_ari']:.4f}, "
            f"final_semi_ari={rows[-1]['semi_ari']:.4f}, "
            f"final_hierarchy_ari={rows[-1]['hierarchy_ari']:.4f}"
        )
        print()
        final_rows.append(rows[-1].copy())

        del rows
        del ls24_q, semi_q, hier_q
        del ls24, semi, hier
        del graph_ls24, graph_semi, graph_hier
        gc.collect()

    ls24_final = np.array([row["ls24_ari"] for row in final_rows], dtype=float)
    semi_final = np.array([row["semi_ari"] for row in final_rows], dtype=float)
    hier_final = np.array([row["hierarchy_ari"] for row in final_rows], dtype=float)
    ls24_total_final = np.array(
        [row["ls24_cumulative_total_runtime_s"] for row in final_rows], dtype=float
    )
    semi_total_final = np.array(
        [row["semi_cumulative_total_runtime_s"] for row in final_rows], dtype=float
    )
    hier_total_final = np.array(
        [row["hierarchy_cumulative_total_runtime_s"] for row in final_rows], dtype=float
    )

    print("Summary across runs")
    print(f"  final_num_updates={int(final_rows[0]['num_updates'])}")
    print(f"  final_ls24_ari_mean={np.mean(ls24_final):.4f} ± {np.std(ls24_final):.4f}")
    print(f"  final_semi_ari_mean={np.mean(semi_final):.4f} ± {np.std(semi_final):.4f}")
    print(
        f"  final_hierarchy_ari_mean={np.mean(hier_final):.4f} ± {np.std(hier_final):.4f}"
    )
    print(
        "  final_ls24_cumulative_total_runtime_s_mean="
        f"{np.mean(ls24_total_final):.3f} ± {np.std(ls24_total_final):.3f}"
    )
    print(
        "  final_semi_cumulative_total_runtime_s_mean="
        f"{np.mean(semi_total_final):.3f} ± {np.std(semi_total_final):.3f}"
    )
    print(
        "  final_hierarchy_cumulative_total_runtime_s_mean="
        f"{np.mean(hier_total_final):.3f} ± {np.std(hier_total_final):.3f}"
    )

    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        results_dir,
        (
            f"compare_ls24_vs_semicontracted_vs_hierarchy_sbm_"
            f"n_{sum(sizes)}_k_{k}_updates_{len(updates)}_batch_{batch_size}_runs_{num_runs}_{timestamp}.csv"
        ),
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  csv={out_path}")
    return out_path


if __name__ == "__main__":
    run_sbm_batch_comparison()
