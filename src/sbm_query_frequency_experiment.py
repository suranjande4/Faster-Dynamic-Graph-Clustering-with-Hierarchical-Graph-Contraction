from __future__ import annotations

import os
import time
from collections import Counter

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

from workload_common import (
    METHOD_COLORS,
    METHOD_LABELS,
    aggregate_mean_std,
    cleanup_runner,
    generate_cluster_biased_nonedge_stream,
    make_runner,
    maybe_enable_fast_edge_lookup,
    write_rows_csv,
)

from sbm import DynamicSBMGraphMerge

import graph_tool as gt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _split_integer_total(total: int, parts: int) -> list[int]:
    total = int(total)
    parts = int(parts)
    if parts <= 0:
        raise ValueError("parts must be > 0.")
    if total < parts:
        raise ValueError(
            "total must be >= parts so each cluster gets at least one node."
        )

    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def build_default_cluster_sizes(
    *,
    init_nodes: int,
    init_clusters: int,
    clusters_to_merge: int,
) -> tuple[int, ...]:
    """
    Build a near-balanced cluster-size vector while keeping the last few clusters
    smaller so the existing pairwise merge helper remains practical.
    """
    init_nodes = int(init_nodes)
    init_clusters = int(init_clusters)
    clusters_to_merge = int(clusters_to_merge)

    if init_clusters <= 0:
        raise ValueError("init_clusters must be > 0.")
    if init_nodes < init_clusters:
        raise ValueError("init_nodes must be >= init_clusters.")
    if clusters_to_merge < 2:
        raise ValueError("clusters_to_merge must be >= 2.")
    if clusters_to_merge > init_clusters:
        raise ValueError("clusters_to_merge cannot exceed init_clusters.")

    if init_clusters == clusters_to_merge:
        return tuple(_split_integer_total(init_nodes, init_clusters))

    num_small = int(clusters_to_merge)
    num_large = int(init_clusters - num_small)

    target_small = max(10, init_nodes // max(1, 5 * init_clusters))
    target_small = min(target_small, 1000)

    max_small_feasible = max(1, (init_nodes - num_large) // num_small)
    target_small = min(target_small, max_small_feasible)

    remaining_nodes = init_nodes - num_small * target_small
    if remaining_nodes < num_large:
        return tuple(_split_integer_total(init_nodes, init_clusters))

    large_sizes = _split_integer_total(remaining_nodes, num_large)
    small_sizes = [int(target_small)] * num_small
    return tuple(large_sizes + small_sizes)


def _normalize_cluster_sizes(
    *,
    cluster_sizes: tuple[int, ...] | list[int] | None,
    init_nodes: int,
    init_clusters: int,
    clusters_to_merge: int,
) -> tuple[int, ...]:
    if cluster_sizes is not None:
        sizes = tuple(int(size) for size in cluster_sizes)
        if len(sizes) != int(init_clusters):
            raise ValueError("len(cluster_sizes) must match init_clusters.")
        if any(size <= 0 for size in sizes):
            raise ValueError("cluster_sizes must all be positive.")
        if sum(sizes) != int(init_nodes):
            raise ValueError("sum(cluster_sizes) must match init_nodes.")
        return sizes

    return build_default_cluster_sizes(
        init_nodes=int(init_nodes),
        init_clusters=int(init_clusters),
        clusters_to_merge=int(clusters_to_merge),
    )


def _normalize_merge_cluster_labels(
    *,
    init_clusters: int,
    clusters_to_merge: int,
    merge_cluster_labels: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    init_clusters = int(init_clusters)
    clusters_to_merge = int(clusters_to_merge)

    if clusters_to_merge < 2:
        raise ValueError("clusters_to_merge must be >= 2.")
    if clusters_to_merge > init_clusters:
        raise ValueError("clusters_to_merge cannot exceed init_clusters.")

    if merge_cluster_labels is None:
        return tuple(range(init_clusters - clusters_to_merge, init_clusters))

    labels = tuple(int(label) for label in merge_cluster_labels)
    if len(labels) != clusters_to_merge:
        raise ValueError("merge_cluster_labels length must match clusters_to_merge.")
    if len(set(labels)) != len(labels):
        raise ValueError("merge_cluster_labels must be unique.")
    if any(label < 0 or label >= init_clusters for label in labels):
        raise ValueError("merge_cluster_labels must refer to initial cluster labels.")
    return labels


def _norm_edge(u: int, v: int) -> tuple[int, int]:
    u = int(u)
    v = int(v)
    return (u, v) if u <= v else (v, u)


def _sample_missing_edge_within(
    graph: gt.Graph,
    nodes: np.ndarray,
    rng: np.random.Generator,
    *,
    max_attempts: int = 3000,
) -> tuple[int, int] | None:
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size < 2:
        return None

    for _ in range(int(max_attempts)):
        u, v = rng.choice(nodes, size=2, replace=False)
        edge = _norm_edge(int(u), int(v))
        if edge[0] == edge[1]:
            continue
        if graph.edge(edge[0], edge[1]) is None:
            return edge
    return None


def _sample_missing_edge_between(
    graph: gt.Graph,
    nodes_a: np.ndarray,
    nodes_b: np.ndarray,
    rng: np.random.Generator,
    *,
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
        edge = _norm_edge(u, v)
        if graph.edge(edge[0], edge[1]) is None:
            return edge
    return None


def _sample_random_missing_edge(
    graph: gt.Graph,
    nodes: np.ndarray,
    rng: np.random.Generator,
    *,
    max_attempts: int = 5000,
) -> tuple[int, int] | None:
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size < 2:
        return None

    for _ in range(int(max_attempts)):
        u, v = rng.choice(nodes, size=2, replace=False)
        edge = _norm_edge(int(u), int(v))
        if graph.edge(edge[0], edge[1]) is None:
            return edge
    return None


def _set_runner_state(runner, true_labels: np.ndarray, num_clusters: int) -> None:
    runner.k = int(num_clusters)
    runner.true_labels = np.asarray(true_labels)


def _ensure_runner_vertices(runner, num_vertices: int) -> None:
    graph = runner.graph
    current_n = int(graph.num_vertices())
    target_n = int(num_vertices)
    if target_n > current_n:
        graph.add_vertex(target_n - current_n)


def _ordered_methods_from_rows(rows: list[dict]) -> list[str]:
    present = {str(row["method"]) for row in rows}
    ordered = [name for name in ("ls24", "semi", "hierarchy") if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _build_wide_query_trace_rows(
    *,
    trace_rows: list[dict],
    run: int,
    query_interval: int,
    ari_query_interval: int,
    methods,
    init_nodes: int,
    init_clusters: int,
    total_steps: int,
    hierarchy_depth: int,
) -> list[dict]:
    method_names = [str(method) for method in methods]
    grouped: dict[tuple[int, str, int, int], dict] = {}

    for trace in trace_rows:
        time_step = int(trace["time_step"])
        phase = str(trace["phase"])
        num_clusters = int(trace["num_clusters"])
        counted_in_runtime = int(bool(trace.get("counted_in_runtime", False)))
        key = (time_step, phase, num_clusters, counted_in_runtime)

        if key not in grouped:
            row = {
                "experiment": "turbulent_query_trace",
                "run": int(run),
                "query_interval": int(query_interval),
                "ari_query_interval": int(ari_query_interval),
                "time_step": int(time_step),
                "phase": str(phase),
                "num_clusters": int(num_clusters),
                "counted_in_runtime": int(counted_in_runtime),
                "init_nodes": int(init_nodes),
                "init_clusters": int(init_clusters),
                "total_steps": int(total_steps),
                "hierarchy_depth": int(hierarchy_depth),
            }
            for method_name in method_names:
                row[f"{method_name}_ari"] = float("nan")
                row[f"{method_name}_query_runtime_s"] = float("nan")
                row[f"{method_name}_cumulative_update_runtime_s"] = float("nan")
                row[f"{method_name}_cumulative_query_runtime_s"] = float("nan")
                row[f"{method_name}_cumulative_dynamic_runtime_s"] = float("nan")
            grouped[key] = row

        row = grouped[key]
        method_name = str(trace["name"])
        row[f"{method_name}_ari"] = float(trace["ari"])
        row[f"{method_name}_query_runtime_s"] = float(trace["query_runtime_s"])
        row[f"{method_name}_cumulative_update_runtime_s"] = float(
            trace["cumulative_update_runtime_s"]
        )
        row[f"{method_name}_cumulative_query_runtime_s"] = float(
            trace["cumulative_query_runtime_s"]
        )
        row[f"{method_name}_cumulative_dynamic_runtime_s"] = float(
            trace["cumulative_dynamic_runtime_s"]
        )

    return [grouped[key] for key in sorted(grouped, key=lambda item: item[0])]


def _write_query_trace_csvs(
    *,
    rows: list[dict],
    out_dir: str,
    prefix: str,
) -> dict[tuple[int, int], str]:
    if not rows:
        return {}

    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = (int(row["query_interval"]), int(row["ari_query_interval"]))
        grouped.setdefault(key, []).append(row)

    out_paths: dict[tuple[int, int], str] = {}
    for key in sorted(grouped):
        query_interval, ari_query_interval = key
        group_rows = sorted(
            grouped[key],
            key=lambda row: (
                int(row["run"]),
                int(row["time_step"]),
                int(row["counted_in_runtime"]),
            ),
        )
        group_prefix = (
            f"{prefix}_qint_{query_interval}_ariq_{ari_query_interval}_query_trace"
        )
        out_paths[key] = write_rows_csv(group_rows, out_dir, group_prefix)

    return out_paths


def build_turbulent_stream_spec(
    *,
    init_nodes=50000,
    init_clusters=10,
    cluster_sizes=None,
    p=0.01,
    q=0.001,
    phase_1_updates=2000,
    phase_2_updates=3000,
    phase_3_updates=2000,
    num_new_clusters=1,
    nodes_per_new_cluster=500,
    clusters_to_merge=2,
    merge_cluster_labels=None,
    steady_intra_cluster_prob=0.85,
    birth_arrival_intra_prob=0.90,
    birth_reinforcement_intra_prob=0.90,
    merge_sampling_prob=0.003,
    seed_graph=7,
    seed_stream=19,
    verbose=True,
):
    init_nodes = int(init_nodes)
    init_clusters = int(init_clusters)
    phase_1_updates = int(phase_1_updates)
    phase_2_updates = int(phase_2_updates)
    phase_3_updates = int(phase_3_updates)
    num_new_clusters = int(num_new_clusters)
    nodes_per_new_cluster = int(nodes_per_new_cluster)
    clusters_to_merge = int(clusters_to_merge)

    if init_clusters < 2:
        raise ValueError("init_clusters must be >= 2.")
    if init_nodes < init_clusters:
        raise ValueError("init_nodes must be >= init_clusters.")
    if num_new_clusters < 1:
        raise ValueError("num_new_clusters must be >= 1.")
    if nodes_per_new_cluster < 1:
        raise ValueError("nodes_per_new_cluster must be >= 1.")
    if phase_2_updates < num_new_clusters * nodes_per_new_cluster:
        raise ValueError(
            "phase_2_updates must be at least num_new_clusters * nodes_per_new_cluster."
        )

    cluster_sizes = _normalize_cluster_sizes(
        cluster_sizes=cluster_sizes,
        init_nodes=init_nodes,
        init_clusters=init_clusters,
        clusters_to_merge=clusters_to_merge,
    )
    initial_k = len(cluster_sizes)
    merge_cluster_labels = _normalize_merge_cluster_labels(
        init_clusters=initial_k,
        clusters_to_merge=clusters_to_merge,
        merge_cluster_labels=merge_cluster_labels,
    )
    merge_target_label = int(merge_cluster_labels[0])
    merge_absorb_labels = tuple(int(label) for label in merge_cluster_labels[1:])

    total_initial_vertices = int(sum(cluster_sizes))
    if verbose:
        print(
            f"[setup] generating turbulent SBM: n={total_initial_vertices}, "
            f"k={initial_k}, sizes={cluster_sizes}, p={p}, q={q}",
            flush=True,
        )

    t0 = time.perf_counter()
    np.random.seed(int(seed_graph))
    gt.seed_rng(int(seed_graph))
    dynamic_sbm = DynamicSBMGraphMerge(
        initial_k, list(cluster_sizes), float(p), float(q)
    )
    base_graph = gt.Graph(dynamic_sbm.get_graph())
    maybe_enable_fast_edge_lookup(base_graph)
    initial_labels = np.array(dynamic_sbm.get_labels(), dtype=int)

    if verbose:
        print(
            f"[setup] turbulent base graph ready: n={int(base_graph.num_vertices())}, "
            f"m={int(base_graph.num_edges())}, time={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    current_graph = gt.Graph(base_graph)
    maybe_enable_fast_edge_lookup(current_graph)
    current_labels = np.array(initial_labels, copy=True)
    events: list[dict] = []
    step = 0

    if verbose:
        print(
            f"[setup] phase 1/3 steady growth: {int(phase_1_updates)} updates",
            flush=True,
        )
    phase1_edges = generate_cluster_biased_nonedge_stream(
        graph=current_graph,
        labels=current_labels,
        num_updates=int(phase_1_updates),
        seed=int(seed_stream),
        intra_cluster_prob=float(steady_intra_cluster_prob),
        verbose=False,
    )
    for edge in phase1_edges:
        u, v = _norm_edge(int(edge[0]), int(edge[1]))
        current_graph.add_edge(u, v)
        step += 1
        events.append(
            {
                "step": int(step),
                "phase": "steady_growth",
                "edge": (u, v),
                "num_clusters": int(initial_k),
                "new_vertex_label": None,
                "label_rewrites": (),
            }
        )

    birth_cluster_labels = tuple(
        int(np.max(initial_labels)) + 1 + idx for idx in range(int(num_new_clusters))
    )
    birth_rng = np.random.default_rng(int(seed_stream) + 1)
    base_nodes = np.arange(int(base_graph.num_vertices()), dtype=int)
    birth_nodes_by_label = {int(label): [] for label in birth_cluster_labels}
    total_birth_vertices = int(num_new_clusters) * int(nodes_per_new_cluster)

    if verbose:
        print(
            f"[setup] phase 2/3 cluster birth: {int(phase_2_updates)} updates, "
            f"{int(total_birth_vertices)} new vertices across {int(num_new_clusters)} cluster(s)",
            flush=True,
        )

    for local_step in range(int(phase_2_updates)):
        new_vertex_label = None
        label_rewrites = ()

        if local_step < int(total_birth_vertices):
            cluster_idx = int(local_step // int(nodes_per_new_cluster))
            assigned_label = int(birth_cluster_labels[cluster_idx])
            new_vertex = int(current_graph.num_vertices())
            current_graph.add_vertex(1)
            current_labels = np.append(current_labels, assigned_label)
            birth_nodes_by_label[assigned_label].append(new_vertex)
            new_vertex_label = int(assigned_label)

            same_cluster_nodes = np.asarray(
                birth_nodes_by_label[assigned_label][:-1], dtype=int
            )
            if same_cluster_nodes.size > 0 and float(birth_rng.random()) < float(
                birth_arrival_intra_prob
            ):
                target = int(birth_rng.choice(same_cluster_nodes))
            else:
                target = int(birth_rng.choice(base_nodes))
            edge = _norm_edge(new_vertex, target)
            current_num_clusters = int(initial_k + cluster_idx + 1)
        else:
            birth_label = int(
                birth_rng.choice(np.asarray(birth_cluster_labels, dtype=int))
            )
            birth_nodes_arr = np.asarray(birth_nodes_by_label[birth_label], dtype=int)
            other_birth_labels = [
                label for label in birth_cluster_labels if int(label) != birth_label
            ]
            draw = float(birth_rng.random())
            if draw < float(birth_reinforcement_intra_prob):
                edge = _sample_missing_edge_within(
                    current_graph,
                    birth_nodes_arr,
                    birth_rng,
                    max_attempts=1024,
                )
            elif other_birth_labels and draw < 0.95:
                other_label = int(
                    birth_rng.choice(np.asarray(other_birth_labels, dtype=int))
                )
                edge = _sample_missing_edge_between(
                    current_graph,
                    birth_nodes_arr,
                    np.asarray(birth_nodes_by_label[other_label], dtype=int),
                    birth_rng,
                    max_attempts=1024,
                )
            elif draw < 0.97:
                edge = _sample_missing_edge_between(
                    current_graph,
                    birth_nodes_arr,
                    base_nodes,
                    birth_rng,
                    max_attempts=1024,
                )
            else:
                edge = _sample_random_missing_edge(
                    current_graph,
                    np.arange(int(current_graph.num_vertices()), dtype=int),
                    birth_rng,
                    max_attempts=2048,
                )

            if edge is None:
                edge = _sample_missing_edge_between(
                    current_graph,
                    birth_nodes_arr,
                    base_nodes,
                    birth_rng,
                    max_attempts=8192,
                )
            if edge is None:
                edge = _sample_random_missing_edge(
                    current_graph,
                    np.arange(int(current_graph.num_vertices()), dtype=int),
                    birth_rng,
                    max_attempts=8192,
                )
            if edge is None:
                raise RuntimeError("Unable to sample a valid birth-phase insertion.")
            current_num_clusters = int(initial_k + num_new_clusters)

        current_graph.add_edge(edge[0], edge[1])
        step += 1
        events.append(
            {
                "step": int(step),
                "phase": "cluster_birth",
                "edge": edge,
                "num_clusters": int(current_num_clusters),
                "new_vertex_label": new_vertex_label,
                "label_rewrites": label_rewrites,
            }
        )

    if verbose:
        print(
            f"[setup] phase 3/3 cluster merge: {int(phase_3_updates)} updates, "
            f"merge labels {merge_cluster_labels} -> {merge_target_label}",
            flush=True,
        )

    merge_candidates = []
    np.random.seed(int(seed_stream) + 2)
    for offset, absorb_label in enumerate(merge_absorb_labels):
        np.random.seed(int(seed_stream) + 2 + offset)
        merge_candidate_edges, _ = dynamic_sbm.merge_clusters(
            int(merge_target_label),
            int(absorb_label),
            float(merge_sampling_prob),
        )
        for edge in merge_candidate_edges:
            if len(edge) < 2:
                continue
            u, v = _norm_edge(int(edge[0]), int(edge[1]))
            if current_graph.edge(u, v) is None:
                merge_candidates.append((u, v))

    if merge_candidates:
        merge_candidates = list(dict.fromkeys(merge_candidates))

    merge_rng = np.random.default_rng(int(seed_stream) + 2)
    if len(merge_candidates) > 0:
        merge_rng.shuffle(merge_candidates)
    merge_cursor = 0

    merge_nodes_keep = np.where(initial_labels == int(merge_target_label))[0].astype(
        int
    )
    merge_nodes_absorb = np.concatenate(
        [
            np.where(initial_labels == int(absorb_label))[0].astype(int)
            for absorb_label in merge_absorb_labels
        ],
        axis=0,
    )
    label_rewrites = tuple(
        (int(label), int(merge_target_label)) for label in merge_absorb_labels
    )
    for absorb_label in merge_absorb_labels:
        current_labels[current_labels == int(absorb_label)] = int(merge_target_label)
    all_nodes = np.arange(int(current_graph.num_vertices()), dtype=int)
    phase_3_cluster_count = int(initial_k + num_new_clusters - len(merge_absorb_labels))

    for local_step in range(int(phase_3_updates)):
        edge = None
        while merge_cursor < len(merge_candidates):
            candidate = merge_candidates[merge_cursor]
            merge_cursor += 1
            if current_graph.edge(candidate[0], candidate[1]) is None:
                edge = candidate
                break

        if edge is None:
            edge = _sample_missing_edge_between(
                current_graph,
                merge_nodes_keep,
                merge_nodes_absorb,
                merge_rng,
                max_attempts=8192,
            )
        if edge is None:
            edge = _sample_random_missing_edge(
                current_graph,
                all_nodes,
                merge_rng,
                max_attempts=8192,
            )
        if edge is None:
            raise RuntimeError("Unable to sample a valid merge-phase insertion.")

        current_graph.add_edge(edge[0], edge[1])
        step += 1
        events.append(
            {
                "step": int(step),
                "phase": "cluster_merge",
                "edge": edge,
                "num_clusters": int(phase_3_cluster_count),
                "new_vertex_label": None,
                "label_rewrites": label_rewrites if local_step == 0 else (),
            }
        )

    stream_spec = {
        "base_graph": base_graph,
        "initial_labels": initial_labels,
        "initial_num_clusters": int(initial_k),
        "initial_num_vertices": int(base_graph.num_vertices()),
        "initial_edges": int(base_graph.num_edges()),
        "cluster_sizes": cluster_sizes,
        "birth_cluster_label": int(birth_cluster_labels[0]),
        "birth_cluster_labels": birth_cluster_labels,
        "num_new_clusters": int(num_new_clusters),
        "nodes_per_new_cluster": int(nodes_per_new_cluster),
        "birth_vertices": int(total_birth_vertices),
        "clusters_to_merge": int(clusters_to_merge),
        "merge_target_label": int(merge_target_label),
        "merge_cluster_labels": merge_cluster_labels,
        "phase1_end": int(phase_1_updates),
        "phase2_end": int(phase_1_updates + phase_2_updates),
        "total_steps": int(step),
        "events": events,
    }

    del current_graph
    del dynamic_sbm
    cleanup_runner(None, None)
    return stream_spec


def iter_turbulent_stream(stream_spec: dict):
    """
    Yield one insertion at a time together with the current mutable ground-truth labels.
    """
    labels = np.array(stream_spec["initial_labels"], copy=True)

    for event in stream_spec["events"]:
        new_vertex_label = event.get("new_vertex_label")
        if new_vertex_label is not None:
            labels = np.append(labels, int(new_vertex_label))

        for from_label, to_label in event.get("label_rewrites", ()):
            labels[labels == int(from_label)] = int(to_label)

        yield {
            "step": int(event["step"]),
            "phase": str(event["phase"]),
            "edge": (int(event["edge"][0]), int(event["edge"][1])),
            "true_labels": labels,
            "num_clusters": int(event["num_clusters"]),
        }


def run_turbulent_stream_group(
    *,
    runners: dict,
    stream_spec: dict,
    query_interval: int,
    ari_query_interval: int | None = None,
    include_initial_ari_query: bool = False,
):
    query_interval = int(query_interval)
    if query_interval <= 0:
        raise ValueError("query_interval must be > 0.")
    if ari_query_interval is not None:
        ari_query_interval = int(ari_query_interval)
        if ari_query_interval <= 0:
            raise ValueError("ari_query_interval must be > 0 when provided.")

    metrics_by_name = {}
    trace_rows = []
    initial_labels = np.asarray(stream_spec["initial_labels"])
    initial_k = int(stream_spec["initial_num_clusters"])

    for name in runners:
        metrics_by_name[name] = {
            "num_applied": 0,
            "num_queries": 0,
            "cumulative_update_runtime_s": 0.0,
            "cumulative_query_runtime_s": 0.0,
            "final_ari": float("nan"),
            "sources": Counter(),
        }

    if include_initial_ari_query:
        for name, runner in runners.items():
            _ensure_runner_vertices(runner, len(initial_labels))
            _set_runner_state(runner, initial_labels, initial_k)
            t0 = time.perf_counter()
            query_info = runner.query()
            query_runtime_s = time.perf_counter() - t0
            trace_rows.append(
                {
                    "name": name,
                    "time_step": 0,
                    "phase": "initial",
                    "num_clusters": int(initial_k),
                    "ari": float(query_info.get("ari", float("nan"))),
                    "query_runtime_s": float(query_runtime_s),
                    "counted_in_runtime": False,
                    "cumulative_update_runtime_s": 0.0,
                    "cumulative_query_runtime_s": 0.0,
                    "cumulative_dynamic_runtime_s": 0.0,
                }
            )

    last_event = None
    for event in iter_turbulent_stream(stream_spec):
        last_event = event
        current_labels = event["true_labels"]
        current_k = int(event["num_clusters"])

        for name, runner in runners.items():
            _ensure_runner_vertices(runner, len(current_labels))
            _set_runner_state(runner, current_labels, current_k)

            t0 = time.perf_counter()
            update_info = runner.add_edge_batch([event["edge"]])
            update_runtime_s = time.perf_counter() - t0

            metrics = metrics_by_name[name]
            metrics["num_applied"] += int(update_info.get("num_applied", 0))
            metrics["cumulative_update_runtime_s"] += float(update_runtime_s)

        runtime_query_due = int(event["step"]) % query_interval == 0
        ari_query_due = (
            ari_query_interval is not None
            and int(event["step"]) % int(ari_query_interval) == 0
        )
        if not runtime_query_due and not ari_query_due:
            continue

        for name, runner in runners.items():
            _set_runner_state(runner, current_labels, current_k)
            q0 = time.perf_counter()
            query_info = runner.query()
            query_runtime_s = time.perf_counter() - q0

            metrics = metrics_by_name[name]
            metrics["final_ari"] = float(query_info.get("ari", float("nan")))

            if runtime_query_due:
                metrics["num_queries"] += 1
                metrics["cumulative_query_runtime_s"] += float(query_runtime_s)

                source = query_info.get("source", None)
                if source is None:
                    source = "hierarchy" if hasattr(runner, "hier") else str(name)
                metrics["sources"][str(source)] += 1

            if ari_query_due:
                trace_rows.append(
                    {
                        "name": name,
                        "time_step": int(event["step"]),
                        "phase": str(event["phase"]),
                        "num_clusters": int(current_k),
                        "ari": float(query_info.get("ari", float("nan"))),
                        "query_runtime_s": float(query_runtime_s),
                        "counted_in_runtime": bool(runtime_query_due),
                        "cumulative_update_runtime_s": float(
                            metrics["cumulative_update_runtime_s"]
                        ),
                        "cumulative_query_runtime_s": float(
                            metrics["cumulative_query_runtime_s"]
                        ),
                        "cumulative_dynamic_runtime_s": float(
                            metrics["cumulative_update_runtime_s"]
                            + metrics["cumulative_query_runtime_s"]
                        ),
                    }
                )

    runtime_final_due = (
        last_event is not None and int(last_event["step"]) % query_interval != 0
    )
    ari_final_due = (
        last_event is not None
        and ari_query_interval is not None
        and int(last_event["step"]) % int(ari_query_interval) != 0
    )
    if runtime_final_due or ari_final_due:
        current_labels = last_event["true_labels"]
        current_k = int(last_event["num_clusters"])
        for name, runner in runners.items():
            _set_runner_state(runner, current_labels, current_k)
            q0 = time.perf_counter()
            query_info = runner.query()
            query_runtime_s = time.perf_counter() - q0

            metrics = metrics_by_name[name]
            metrics["final_ari"] = float(query_info.get("ari", float("nan")))

            if runtime_final_due:
                metrics["num_queries"] += 1
                metrics["cumulative_query_runtime_s"] += float(query_runtime_s)

                source = query_info.get("source", None)
                if source is None:
                    source = "hierarchy" if hasattr(runner, "hier") else str(name)
                metrics["sources"][str(source)] += 1

            if ari_final_due:
                trace_rows.append(
                    {
                        "name": name,
                        "time_step": int(last_event["step"]),
                        "phase": str(last_event["phase"]),
                        "num_clusters": int(current_k),
                        "ari": float(query_info.get("ari", float("nan"))),
                        "query_runtime_s": float(query_runtime_s),
                        "counted_in_runtime": bool(runtime_final_due),
                        "cumulative_update_runtime_s": float(
                            metrics["cumulative_update_runtime_s"]
                        ),
                        "cumulative_query_runtime_s": float(
                            metrics["cumulative_query_runtime_s"]
                        ),
                        "cumulative_dynamic_runtime_s": float(
                            metrics["cumulative_update_runtime_s"]
                            + metrics["cumulative_query_runtime_s"]
                        ),
                    }
                )

    summary = {}
    for name, metrics in metrics_by_name.items():
        update_s = float(metrics["cumulative_update_runtime_s"])
        query_s = float(metrics["cumulative_query_runtime_s"])
        num_applied = int(metrics["num_applied"])
        num_queries = int(metrics["num_queries"])
        summary[name] = {
            "num_applied": num_applied,
            "num_queries": num_queries,
            "cumulative_update_runtime_s": update_s,
            "cumulative_query_runtime_s": query_s,
            "cumulative_dynamic_runtime_s": float(update_s + query_s),
            "avg_update_time_per_edge_s": float(update_s / max(1, num_applied)),
            "avg_query_time_s": float(query_s / max(1, num_queries)),
            "final_ari": float(metrics["final_ari"]),
            "query_sources": ";".join(
                f"{key}:{value}" for key, value in sorted(metrics["sources"].items())
            ),
        }

    return summary, trace_rows


def plot_query_frequency(rows: list[dict], out_dir: str, prefix: str) -> str:
    summary = aggregate_mean_std(
        rows,
        key_fields=("method", "query_interval"),
        value_fields=("cumulative_dynamic_runtime_s",),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_runtime_vs_query_frequency.png")
    plt.figure(figsize=(9, 6))

    methods = _ordered_methods_from_rows(rows)
    for method in methods:
        method_rows = [row for row in summary if row["method"] == method]
        method_rows.sort(key=lambda row: int(row["query_interval"]))
        x = np.array([int(row["query_interval"]) for row in method_rows], dtype=float)
        y = np.array(
            [float(row["cumulative_dynamic_runtime_s_mean"]) for row in method_rows]
        )
        y_std = np.array(
            [float(row["cumulative_dynamic_runtime_s_std"]) for row in method_rows]
        )

        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)
        plt.plot(x, y, marker="o", linewidth=2.2, label=label, color=color)
        if np.any(y_std > 0):
            plt.fill_between(
                x,
                np.maximum(y - y_std, 1e-12),
                y + y_std,
                alpha=0.15,
                color=color,
            )

    plt.yscale("log")
    plt.xlabel("Query Interval q")
    plt.ylabel("Total Cumulative Runtime (s, log scale)")
    plt.title("Turbulent Stream Runtime vs Query Frequency")
    plt.xticks(sorted({int(row["query_interval"]) for row in rows}))
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_ari_over_time(
    rows: list[dict],
    out_dir: str,
    prefix: str,
    *,
    phase1_end: int,
    phase2_end: int,
) -> str:
    if not rows:
        raise ValueError("rows must be non-empty for ARI plotting.")

    summary = aggregate_mean_std(
        rows,
        key_fields=("method", "time_step"),
        value_fields=("ari",),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_ari_vs_time.png")
    plt.figure(figsize=(10, 6))

    methods = _ordered_methods_from_rows(rows)
    for method in methods:
        method_rows = [row for row in summary if row["method"] == method]
        method_rows.sort(key=lambda row: int(row["time_step"]))
        x = np.array([int(row["time_step"]) for row in method_rows], dtype=float)
        y = np.array([float(row["ari_mean"]) for row in method_rows])
        y_std = np.array([float(row["ari_std"]) for row in method_rows])

        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)
        plt.plot(x, y, linewidth=2.2, label=label, color=color)
        if np.any(y_std > 0):
            plt.fill_between(
                x,
                np.clip(y - y_std, 0.0, 1.0),
                np.clip(y + y_std, 0.0, 1.0),
                alpha=0.15,
                color=color,
            )

    plt.axvline(int(phase1_end), color="#666666", linestyle="--", linewidth=1.3)
    plt.axvline(int(phase2_end), color="#666666", linestyle="--", linewidth=1.3)
    plt.xlim(0, max(int(phase2_end), int(max(row["time_step"] for row in rows))))
    plt.ylim(0.0, 1.05)
    plt.xlabel("Time Step")
    plt.ylabel("ARI")
    if len({int(row["query_interval"]) for row in rows}) == 1:
        query_interval = int(rows[0]["query_interval"])
        ari_query_interval = int(rows[0]["ari_query_interval"])
        plt.title(
            f"Turbulent Stream ARI vs Time Step (q={query_interval}, ari_q={ari_query_interval})"
        )
    else:
        plt.title("Turbulent Stream ARI vs Time Step")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def run_query_frequency_experiment(
    *,
    init_nodes=10000,
    init_clusters=5,
    cluster_sizes=None,
    p=0.01,
    q=0.001,
    phase_1_updates=2000,
    phase_2_updates=3000,
    phase_3_updates=2000,
    num_new_clusters=1,
    nodes_per_new_cluster=500,
    clusters_to_merge=2,
    query_intervals=(10, 50, 500),
    methods=("ls24", "semi", "hierarchy"),
    hierarchy_depth=4,
    ari_query_intervals=None,
    num_runs=2,
    seed_graph=7,
    seed_stream=19,
    seed_algo=123,
    sampling_constant=80,
    degree_trigger=0.5,
    gamma=2.0,
    beta=0.5,
    ls24_query_policy="algorithm",
    merge_sampling_prob=0.003,
    merge_cluster_labels=None,
    runner_verbose=False,
    results_dir="results/sbm_turbulent_query_frequency",
    verbose=True,
):
    query_intervals = [int(qi) for qi in query_intervals]
    if len(query_intervals) == 0:
        raise ValueError("query_intervals must contain at least one value.")
    if any(int(qi) <= 0 for qi in query_intervals):
        raise ValueError("query_intervals values must all be > 0.")

    if ari_query_intervals is None:
        ari_query_intervals = list(query_intervals)
    else:
        ari_query_intervals = [int(qi) for qi in ari_query_intervals]
        if len(ari_query_intervals) != len(query_intervals):
            raise ValueError(
                "ari_query_intervals must have the same length as query_intervals."
            )
        if any(int(qi) <= 0 for qi in ari_query_intervals):
            raise ValueError("ari_query_intervals values must all be > 0.")

    stream_spec = build_turbulent_stream_spec(
        init_nodes=init_nodes,
        init_clusters=init_clusters,
        cluster_sizes=cluster_sizes,
        p=p,
        q=q,
        phase_1_updates=phase_1_updates,
        phase_2_updates=phase_2_updates,
        phase_3_updates=phase_3_updates,
        num_new_clusters=num_new_clusters,
        nodes_per_new_cluster=nodes_per_new_cluster,
        clusters_to_merge=clusters_to_merge,
        merge_sampling_prob=merge_sampling_prob,
        merge_cluster_labels=merge_cluster_labels,
        seed_graph=seed_graph,
        seed_stream=seed_stream,
        verbose=verbose,
    )

    rows = []
    ari_rows = []
    query_trace_rows = []
    method_offsets = {"ls24": 11, "semi": 23, "hierarchy": 37}

    for run_idx in range(int(num_runs)):
        for query_interval, ari_query_interval in zip(
            query_intervals, ari_query_intervals
        ):
            if verbose:
                print(
                    f"[run {run_idx + 1}/{num_runs}] turbulent q={int(query_interval)}, "
                    f"ari_q={int(ari_query_interval)}",
                    flush=True,
                )

            runners = {}
            init_runtimes = {}
            for method in methods:
                np.random.seed(
                    int(seed_algo)
                    + 10000 * int(run_idx)
                    + 101 * int(query_interval)
                    + method_offsets[str(method)]
                )
                graph = gt.Graph(stream_spec["base_graph"])
                maybe_enable_fast_edge_lookup(graph)

                init_start = time.perf_counter()
                runners[method] = make_runner(
                    method=str(method),
                    graph=graph,
                    true_labels=stream_spec["initial_labels"],
                    k=stream_spec["initial_num_clusters"],
                    sampling_constant=sampling_constant,
                    degree_trigger=degree_trigger,
                    gamma=gamma,
                    beta=beta,
                    num_levels=int(hierarchy_depth),
                    ls24_query_policy=ls24_query_policy,
                    verbose=bool(runner_verbose),
                )
                init_runtimes[method] = float(time.perf_counter() - init_start)

            metrics_by_method, trace_rows = run_turbulent_stream_group(
                runners=runners,
                stream_spec=stream_spec,
                query_interval=int(query_interval),
                ari_query_interval=int(ari_query_interval),
                include_initial_ari_query=True,
            )
            query_trace_rows.extend(
                _build_wide_query_trace_rows(
                    trace_rows=trace_rows,
                    run=int(run_idx + 1),
                    query_interval=int(query_interval),
                    ari_query_interval=int(ari_query_interval),
                    methods=methods,
                    init_nodes=int(init_nodes),
                    init_clusters=int(init_clusters),
                    total_steps=int(stream_spec["total_steps"]),
                    hierarchy_depth=int(hierarchy_depth),
                )
            )

            for method, metrics in metrics_by_method.items():
                row = {
                    "experiment": "turbulent_query_frequency",
                    "run": int(run_idx + 1),
                    "method": str(method),
                    "cluster_sizes": ";".join(
                        str(x) for x in stream_spec["cluster_sizes"]
                    ),
                    "init_nodes": int(init_nodes),
                    "init_clusters": int(init_clusters),
                    "k_init": int(stream_spec["initial_num_clusters"]),
                    "num_vertices": int(stream_spec["initial_num_vertices"]),
                    "initial_edges": int(stream_spec["initial_edges"]),
                    "p": float(p),
                    "q": float(q),
                    "phase_1_updates": int(phase_1_updates),
                    "phase_2_updates": int(phase_2_updates),
                    "phase_3_updates": int(phase_3_updates),
                    "total_steps": int(stream_spec["total_steps"]),
                    "num_new_clusters": int(num_new_clusters),
                    "nodes_per_new_cluster": int(nodes_per_new_cluster),
                    "birth_vertices": int(stream_spec["birth_vertices"]),
                    "birth_cluster_labels": ";".join(
                        str(label) for label in stream_spec["birth_cluster_labels"]
                    ),
                    "clusters_to_merge": int(clusters_to_merge),
                    "merge_target_label": int(stream_spec["merge_target_label"]),
                    "merge_cluster_labels": ";".join(
                        str(label) for label in stream_spec["merge_cluster_labels"]
                    ),
                    "query_interval": int(query_interval),
                    "ari_query_interval": int(ari_query_interval),
                    "hierarchy_depth": int(hierarchy_depth),
                    "sampling_constant": int(sampling_constant),
                    "degree_trigger": float(degree_trigger),
                    "gamma": float(gamma),
                    "beta": float(beta),
                    "ls24_query_policy": str(ls24_query_policy),
                    "init_runtime_s": float(init_runtimes[method]),
                    **metrics,
                    "end_to_end_runtime_s": float(
                        float(init_runtimes[method])
                        + float(metrics["cumulative_dynamic_runtime_s"])
                    ),
                }
                rows.append(row)

                if verbose:
                    print(
                        f"  {METHOD_LABELS.get(str(method), str(method))}: "
                        f"init={row['init_runtime_s']:.3f}s, "
                        f"update={row['cumulative_update_runtime_s']:.3f}s, "
                        f"query={row['cumulative_query_runtime_s']:.3f}s, "
                        f"total={row['cumulative_dynamic_runtime_s']:.3f}s, "
                        f"ari={row['final_ari']:.4f}",
                        flush=True,
                    )

            for trace in trace_rows:
                ari_rows.append(
                    {
                        "experiment": "turbulent_ari_trace",
                        "run": int(run_idx + 1),
                        "method": str(trace["name"]),
                        "query_interval": int(query_interval),
                        "ari_query_interval": int(ari_query_interval),
                        "time_step": int(trace["time_step"]),
                        "phase": str(trace["phase"]),
                        "num_clusters": int(trace["num_clusters"]),
                        "ari": float(trace["ari"]),
                        "query_runtime_s": float(trace["query_runtime_s"]),
                        "counted_in_runtime": int(
                            bool(trace.get("counted_in_runtime", False))
                        ),
                    }
                )

            del runners
            cleanup_runner(None, None)

    prefix = (
        f"turbulent_query_frequency_n_{int(stream_spec['initial_num_vertices'])}_"
        f"updates_{int(stream_spec['total_steps'])}_d_{int(hierarchy_depth)}"
    )
    runtime_csv = write_rows_csv(rows, results_dir, f"{prefix}_runtime")
    query_trace_csvs = _write_query_trace_csvs(
        rows=query_trace_rows,
        out_dir=results_dir,
        prefix=prefix,
    )
    runtime_plot = plot_query_frequency(rows, results_dir, prefix)
    ari_plot_paths = []
    for query_interval in sorted({int(row["query_interval"]) for row in ari_rows}):
        query_rows = [
            row for row in ari_rows if int(row["query_interval"]) == int(query_interval)
        ]
        ari_query_interval = int(query_rows[0]["ari_query_interval"])
        plot_prefix = f"{prefix}_qint_{query_interval}_ariq_{ari_query_interval}"
        ari_plot_paths.append(
            plot_ari_over_time(
                query_rows,
                results_dir,
                plot_prefix,
                phase1_end=int(stream_spec["phase1_end"]),
                phase2_end=int(stream_spec["phase2_end"]),
            )
        )

    print(f"runtime_csv={runtime_csv}")
    for (query_interval, ari_query_interval), path in query_trace_csvs.items():
        print(
            f"query_trace_csv[q={int(query_interval)},ari_q={int(ari_query_interval)}]={path}"
        )
    print(f"runtime_plot={runtime_plot}")
    if len(ari_plot_paths) == 1:
        print(f"ari_plot={ari_plot_paths[0]}")
        ari_plot_output = ari_plot_paths[0]
    else:
        for path in ari_plot_paths:
            print(f"ari_plot={path}")
        ari_plot_output = ari_plot_paths
    return runtime_csv, query_trace_csvs, runtime_plot, ari_plot_output


def main():
    INIT_NODES = 15000
    INIT_CLUSTERS = 5
    p = 0.07
    q = 0.01
    PHASE_1_UPDATES = 5000
    PHASE_2_UPDATES = 5000
    PHASE_3_UPDATES = 5000
    NUM_NEW_CLUSTERS = 2
    NODES_PER_NEW_CLUSTER = 200
    CLUSTERS_TO_MERGE = 3
    QUERY_INTERVALS = [20]
    methods = ("ls24", "semi", "hierarchy")
    HIERARCHY_DEPTH = 4
    ARI_QUERY_INTERVALS = QUERY_INTERVALS
    num_runs = 3
    seed_graph = 7
    seed_stream = 19
    seed_algo = 123
    sampling_constant = 3
    degree_trigger = 0.5
    gamma = 2.0
    beta = 0.5
    ls24_query_policy = "algorithm"
    merge_sampling_prob = 0.003
    MERGE_CLUSTER_LABELS = None
    results_dir = "results/sbm_turbulent_query_frequency"
    runner_verbose = False
    verbose = True

    run_query_frequency_experiment(
        init_nodes=INIT_NODES,
        init_clusters=INIT_CLUSTERS,
        p=p,
        q=q,
        phase_1_updates=PHASE_1_UPDATES,
        phase_2_updates=PHASE_2_UPDATES,
        phase_3_updates=PHASE_3_UPDATES,
        num_new_clusters=NUM_NEW_CLUSTERS,
        nodes_per_new_cluster=NODES_PER_NEW_CLUSTER,
        clusters_to_merge=CLUSTERS_TO_MERGE,
        query_intervals=QUERY_INTERVALS,
        methods=methods,
        hierarchy_depth=HIERARCHY_DEPTH,
        ari_query_intervals=ARI_QUERY_INTERVALS,
        num_runs=num_runs,
        seed_graph=seed_graph,
        seed_stream=seed_stream,
        seed_algo=seed_algo,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        gamma=gamma,
        beta=beta,
        ls24_query_policy=ls24_query_policy,
        merge_sampling_prob=merge_sampling_prob,
        merge_cluster_labels=MERGE_CLUSTER_LABELS,
        runner_verbose=runner_verbose,
        results_dir=results_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
