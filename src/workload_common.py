from __future__ import annotations

import csv
import gc
import os
import time
from collections import Counter, defaultdict
from datetime import datetime

import graph_tool as gt
import numpy as np

from runners import HierarchyRunner, LS24Runner, SemiContractedRunner
from sbm import DynamicSBMGraph


METHOD_LABELS = {
    "ls24": "LS24",
    "semi": "Semi-Contracted",
    "hierarchy": "Hierarchy",
}
METHOD_COLORS = {
    "ls24": "#1f77b4",
    "semi": "#ff7f0e",
    "hierarchy": "#2ca02c",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def maybe_enable_fast_edge_lookup(graph: gt.Graph) -> None:
    fn = getattr(graph, "set_fast_edge_lookup", None)
    if callable(fn):
        try:
            fn(fast=True)
        except TypeError:
            fn(True)


def build_initial_equal_sbm(
    *,
    k: int,
    n_per_cluster: int,
    p: float,
    q: float,
    seed: int,
    verbose: bool = True,
) -> tuple[gt.Graph, np.ndarray]:
    np.random.seed(int(seed))
    if verbose:
        total_n = int(k) * int(n_per_cluster)
        print(
            f"[setup] generating SBM: n={total_n}, k={k}, "
            f"cluster_size={n_per_cluster}, p={p}, q={q}",
            flush=True,
        )
    t0 = time.perf_counter()
    sbm = DynamicSBMGraph(int(k), int(n_per_cluster), float(p), float(q))
    graph = sbm.get_graph()
    labels = np.array(sbm.get_labels())
    maybe_enable_fast_edge_lookup(graph)
    if verbose:
        print(
            f"[setup] graph ready: n={int(graph.num_vertices())}, "
            f"m={int(graph.num_edges())}, time={time.perf_counter() - t0:.2f}s",
            flush=True,
        )
    return graph, labels


def generate_random_nonedge_stream(
    *,
    graph: gt.Graph,
    num_updates: int,
    seed: int,
    verbose: bool = True,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    n = int(graph.num_vertices())
    updates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(10000, int(num_updates) * 200)

    if verbose:
        print(
            f"[setup] generating {int(num_updates)} random non-edge insertions...",
            flush=True,
        )
    t0 = time.perf_counter()
    while len(updates) < int(num_updates) and attempts < max_attempts:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        attempts += 1
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edge = (u, v)
        if edge in seen:
            continue
        if graph.edge(u, v) is not None:
            continue
        seen.add(edge)
        updates.append(edge)

    if len(updates) < int(num_updates):
        raise RuntimeError(
            f"Could only generate {len(updates)} non-edge updates after {attempts} attempts."
        )
    if verbose:
        print(
            f"[setup] update stream ready in {time.perf_counter() - t0:.2f}s "
            f"after {attempts} attempts",
            flush=True,
        )
    return updates


def _sample_missing_edge_within_cluster(
    *,
    graph: gt.Graph,
    nodes: np.ndarray,
    seen: set[tuple[int, int]],
    rng: np.random.Generator,
    max_attempts: int = 3000,
) -> tuple[int, int] | None:
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size < 2:
        return None

    for _ in range(int(max_attempts)):
        u, v = rng.choice(nodes, size=2, replace=False)
        u = int(u)
        v = int(v)
        if u > v:
            u, v = v, u
        edge = (u, v)
        if edge in seen:
            continue
        if graph.edge(u, v) is not None:
            continue
        return edge
    return None


def _sample_missing_edge_between_clusters(
    *,
    graph: gt.Graph,
    nodes_a: np.ndarray,
    nodes_b: np.ndarray,
    seen: set[tuple[int, int]],
    rng: np.random.Generator,
    max_attempts: int = 3000,
) -> tuple[int, int] | None:
    nodes_a = np.asarray(nodes_a, dtype=int)
    nodes_b = np.asarray(nodes_b, dtype=int)
    if nodes_a.size == 0 or nodes_b.size == 0:
        return None

    for _ in range(int(max_attempts)):
        u = int(rng.choice(nodes_a))
        v = int(rng.choice(nodes_b))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edge = (u, v)
        if edge in seen:
            continue
        if graph.edge(u, v) is not None:
            continue
        return edge
    return None


def generate_cluster_biased_nonedge_stream(
    *,
    graph: gt.Graph,
    labels: np.ndarray,
    num_updates: int,
    seed: int,
    intra_cluster_prob: float = 0.85,
    verbose: bool = True,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels, dtype=int)
    n = int(graph.num_vertices())
    if labels.shape[0] != n:
        raise ValueError("labels length must match graph.num_vertices().")

    intra_cluster_prob = float(intra_cluster_prob)
    if not (0.0 <= intra_cluster_prob <= 1.0):
        raise ValueError("intra_cluster_prob must be in [0, 1].")

    unique_labels = np.unique(labels)
    cluster_nodes = {
        int(label): np.where(labels == label)[0].astype(int) for label in unique_labels
    }
    viable_intra_labels = [
        int(label) for label, nodes in cluster_nodes.items() if int(nodes.size) >= 2
    ]

    updates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(10000, int(num_updates) * 400)

    if verbose:
        print(
            f"[setup] generating {int(num_updates)} cluster-biased non-edge insertions "
            f"(intra_cluster_prob={intra_cluster_prob:.2f})...",
            flush=True,
        )
    t0 = time.perf_counter()

    while len(updates) < int(num_updates) and attempts < max_attempts:
        attempts += 1
        edge: tuple[int, int] | None = None

        use_intra = bool(
            len(viable_intra_labels) > 0 and rng.random() < intra_cluster_prob
        )
        if use_intra:
            label = int(rng.choice(viable_intra_labels))
            edge = _sample_missing_edge_within_cluster(
                graph=graph,
                nodes=cluster_nodes[label],
                seen=seen,
                rng=rng,
                max_attempts=32,
            )
        elif unique_labels.size >= 2:
            label_a, label_b = rng.choice(unique_labels, size=2, replace=False)
            edge = _sample_missing_edge_between_clusters(
                graph=graph,
                nodes_a=cluster_nodes[int(label_a)],
                nodes_b=cluster_nodes[int(label_b)],
                seen=seen,
                rng=rng,
                max_attempts=32,
            )

        if edge is None:
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                continue
            if u > v:
                u, v = v, u
            edge = (u, v)
            if edge in seen:
                continue
            if graph.edge(u, v) is not None:
                continue

        seen.add(edge)
        updates.append(edge)

    if len(updates) < int(num_updates):
        raise RuntimeError(
            "Could only generate "
            f"{len(updates)} cluster-biased non-edge updates after {attempts} attempts."
        )
    if verbose:
        print(
            f"[setup] cluster-biased update stream ready in "
            f"{time.perf_counter() - t0:.2f}s after {attempts} attempts",
            flush=True,
        )
    return updates


def edge_chunks(updates: list[tuple[int, int]], chunk_size: int):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    for start in range(0, len(updates), chunk_size):
        yield updates[start : start + chunk_size]


def make_runner(
    *,
    method: str,
    graph: gt.Graph,
    true_labels: np.ndarray,
    k: int,
    sampling_constant: int,
    degree_trigger: float,
    gamma: float,
    beta: float,
    num_levels: int,
    ls24_query_policy: str = "algorithm",
    verbose: bool = False,
):
    method = str(method)
    if method == "ls24":
        return LS24Runner(
            graph=graph,
            true_labels=true_labels,
            k=int(k),
            sampling_constant=int(sampling_constant),
            degree_trigger=float(degree_trigger),
            gamma=float(gamma),
            query_policy=str(ls24_query_policy),
        )
    if method == "semi":
        return SemiContractedRunner(
            graph=graph,
            true_labels=true_labels,
            k=int(k),
            beta=float(beta),
            sampling_constant=int(sampling_constant),
            degree_trigger=float(degree_trigger),
            verbose=bool(verbose),
        )
    if method == "hierarchy":
        return HierarchyRunner(
            graph=graph,
            true_labels=true_labels,
            k=int(k),
            num_levels=int(num_levels),
            sampling_constant=int(sampling_constant),
            degree_trigger=float(degree_trigger),
            verbose=bool(verbose),
        )
    raise ValueError(f"Unknown method: {method}")


def run_update_query_stream(
    *,
    runner,
    updates: list[tuple[int, int]],
    query_interval: int,
) -> dict:
    cumulative_update_s = 0.0
    cumulative_query_s = 0.0
    total_applied = 0
    num_queries = 0
    final_ari = float("nan")
    sources = Counter()

    for batch in edge_chunks(updates, int(query_interval)):
        t0 = time.perf_counter()
        update_info = runner.add_edge_batch(batch)
        cumulative_update_s += time.perf_counter() - t0
        total_applied += int(update_info.get("num_applied", len(batch)))

        q0 = time.perf_counter()
        query_info = runner.query()
        cumulative_query_s += time.perf_counter() - q0
        num_queries += 1
        final_ari = float(query_info.get("ari", float("nan")))
        sources[str(query_info.get("source", "hierarchy"))] += 1

    dynamic_total_s = cumulative_update_s + cumulative_query_s
    return {
        "num_applied": int(total_applied),
        "num_queries": int(num_queries),
        "cumulative_update_runtime_s": float(cumulative_update_s),
        "cumulative_query_runtime_s": float(cumulative_query_s),
        "cumulative_dynamic_runtime_s": float(dynamic_total_s),
        "avg_update_time_per_edge_s": float(
            cumulative_update_s / max(1, total_applied)
        ),
        "avg_query_time_s": float(cumulative_query_s / max(1, num_queries)),
        "final_ari": float(final_ari),
        "query_sources": ";".join(f"{k}:{v}" for k, v in sorted(sources.items())),
    }


def write_rows_csv(rows: list[dict], out_dir: str, prefix: str) -> str:
    if not rows:
        raise RuntimeError("No rows to write.")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{timestamp()}.csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def aggregate_mean_std(
    rows: list[dict], key_fields: tuple[str, ...], value_fields: tuple[str, ...]
):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        result = {field: value for field, value in zip(key_fields, key)}
        for field in value_fields:
            values = np.array([float(row[field]) for row in group], dtype=float)
            result[f"{field}_mean"] = float(np.mean(values))
            result[f"{field}_std"] = float(np.std(values))
        out.append(result)
    return out


def cleanup_runner(runner, graph) -> None:
    gc.collect()
