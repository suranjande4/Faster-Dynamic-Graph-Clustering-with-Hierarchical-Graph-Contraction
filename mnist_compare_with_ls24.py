import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import graph_tool as gt
import numpy as np

from datasets import KNN_Graph_Manager, build_initial_graph
from runners import HierarchyRunner, LS24Runner, SemiContractedRunner


def randomize_mnist_classes(initial_class_count=4, rng=None):
    """Pick initial MNIST classes and return remaining classes as singleton updates."""
    if rng is None:
        rng = random

    all_classes = [str(i) for i in range(10)]
    current_classes = rng.sample(all_classes, int(initial_class_count))
    remaining_classes = [cls for cls in all_classes if cls not in current_classes]
    rng.shuffle(remaining_classes)
    classes_to_add = [[cls] for cls in remaining_classes]
    return current_classes, classes_to_add


def unique_undirected_edges(edges):
    """Return sorted unique undirected edges as Nx2 int array."""
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.empty((0, 2), dtype=int)

    if edges.ndim == 1:
        edges = edges.reshape(1, -1)

    undirected = np.sort(edges[:, :2].astype(int), axis=1)
    return np.unique(undirected, axis=0)


def split_edges_into_batches(edges, num_batches, rng=None):
    """
    Shuffle and split edges into `num_batches` roughly equal batches.
    Empty batches are kept so each class update still has the same batch count.
    """
    if rng is None:
        rng = np.random.default_rng()

    edges = np.asarray(edges)
    if edges.size == 0:
        return [np.empty((0, 2), dtype=int) for _ in range(int(num_batches))]

    perm = rng.permutation(len(edges))
    shuffled = edges[perm]
    return [
        chunk.astype(int, copy=False)
        for chunk in np.array_split(shuffled, int(num_batches))
    ]


def choose_stage_checkpoint_batches(num_batches, num_records=10):
    """
    Choose up to `num_records` evenly spaced 1-based batch indices for CSV logging.
    Always includes the final batch.
    """
    num_batches = int(num_batches)
    num_records = int(num_records)

    if num_batches <= 0:
        return set()
    if num_records <= 0:
        raise ValueError("num_records must be >= 1.")
    if num_batches <= num_records:
        return set(range(1, num_batches + 1))

    checkpoints = {
        min(num_batches, max(1, int(round(step * num_batches / num_records))))
        for step in range(1, num_records + 1)
    }
    checkpoints.add(num_batches)
    return checkpoints


def all_stage_batches(num_batches):
    num_batches = int(num_batches)
    if num_batches <= 0:
        return set()
    return set(range(1, num_batches + 1))


def graph_labels_from_original_indices(graph, knn_label_property):
    """
    Build current ground-truth labels for `graph` using its `original_indices` map.
    """
    labels = []
    original_indices = graph.vp["original_indices"]
    for v in graph.vertices():
        old_idx = int(original_indices[v])
        labels.append(knn_label_property[old_idx])
    return labels


def initialize_shared_local_mapping(initial_graph, total_original_vertices):
    """
    Build a shared mapping array original_id -> local_id from the initial graph.
    Unseen original ids are set to -1.
    """
    total_original_vertices = int(total_original_vertices)
    local_of_original = np.full(total_original_vertices, -1, dtype=np.int64)
    original_indices = initial_graph.vp["original_indices"]
    for v in initial_graph.vertices():
        local_of_original[int(original_indices[v])] = int(v)
    next_local_id = int(initial_graph.num_vertices())
    return local_of_original, next_local_id


def remap_batch_edges_shared(batch_edges_original, local_of_original, next_local_id):
    """
    Remap one batch of original-id edges to local ids using a shared mapping state.
    Returns:
    - local_edges_unique: Nx2 local-id edges
    - new_original_ids: original ids first seen in this batch (ascending)
    - next_local_id: updated next local vertex id
    """
    batch_edges_original = np.asarray(batch_edges_original)
    if batch_edges_original.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            int(next_local_id),
        )

    if batch_edges_original.ndim == 1:
        batch_edges_original = batch_edges_original.reshape(1, -1)

    edges_2 = batch_edges_original[:, :2].astype(int, copy=False)
    touched_original_ids = np.unique(edges_2.reshape(-1))
    unseen_mask = local_of_original[touched_original_ids] < 0
    new_original_ids = touched_original_ids[unseen_mask]

    if len(new_original_ids) > 0:
        count_new = int(len(new_original_ids))
        local_ids = np.arange(next_local_id, next_local_id + count_new, dtype=np.int64)
        local_of_original[new_original_ids] = local_ids
        next_local_id += count_new

    u_local = local_of_original[edges_2[:, 0]]
    v_local = local_of_original[edges_2[:, 1]]

    valid = u_local != v_local
    if not np.any(valid):
        return np.empty((0, 2), dtype=int), new_original_ids, int(next_local_id)

    u_local = u_local[valid].astype(int, copy=False)
    v_local = v_local[valid].astype(int, copy=False)
    remapped = np.column_stack(
        (np.minimum(u_local, v_local), np.maximum(u_local, v_local))
    )
    remapped = np.unique(remapped, axis=0)
    return remapped, new_original_ids.astype(int, copy=False), int(next_local_id)


def append_new_vertices_with_original_ids(graph, new_original_ids):
    """
    Append vertices to `graph` and set graph.vp['original_indices'] for each new vertex.
    """
    if len(new_original_ids) == 0:
        return

    new_original_ids = np.asarray(new_original_ids, dtype=int)
    start = int(graph.num_vertices())
    graph.add_vertex(int(len(new_original_ids)))
    original_indices = graph.vp["original_indices"]
    for i, old_idx in enumerate(new_original_ids):
        original_indices[graph.vertex(start + i)] = int(old_idx)


def maybe_enable_fast_edge_lookup(graph):
    """
    Enable fast edge lookup when supported by the local graph_tool version.
    """
    fn = getattr(graph, "set_fast_edge_lookup", None)
    if callable(fn):
        try:
            fn(fast=True)
        except TypeError:
            # Older/newer signatures can differ across graph_tool builds.
            fn(True)


def build_mnist_class_update_plan(
    knn_manager, current_classes, classes_to_add, num_batches, rng
):
    """
    Build update stages as:
    - add one new class at a time
    - include edges within the new class and between old/new classes
    - split each class-stage edge set into `num_batches`.
    """
    current_vertices = knn_manager.get_vertices_by_labels(current_classes)
    stages = []

    for cls_group in classes_to_add:
        vertices_to_add = knn_manager.get_vertices_by_labels(cls_group)
        edges_to_add = knn_manager.get_edges_between_sets(
            current_vertices, vertices_to_add
        )
        edges_to_add = unique_undirected_edges(edges_to_add)
        edge_batches = split_edges_into_batches(
            edges_to_add, num_batches=num_batches, rng=rng
        )

        current_classes = current_classes + list(cls_group)
        current_vertices.extend(vertices_to_add)
        stages.append(
            {
                "added_classes": list(cls_group),
                "active_classes": list(current_classes),
                "num_edges": int(len(edges_to_add)),
                "batches": edge_batches,
            }
        )

    return stages


def _run_one_algorithm_step(
    runner, batch_edges_local, k_now, query_runtime_multiplier=1.0
):
    """
    Apply one update batch + one query, returning timing/ARI stats.
    """
    # Keep cluster-count aligned with the number of currently active classes.
    runner.k = int(k_now)

    t0 = time.perf_counter()
    update_info = runner.add_edge_batch(batch_edges_local)
    update_runtime = time.perf_counter() - t0

    query_info = runner.query()
    query_runtime_s = float(query_info.get("runtime", 0.0)) * float(
        query_runtime_multiplier
    )

    return {
        "batch_edges_local": int(len(batch_edges_local)),
        "num_applied": int(update_info.get("num_applied", 0)),
        "ari": float(query_info.get("ari", float("nan"))),
        "query_runtime_s": float(query_runtime_s),
        "update_runtime_s": float(update_runtime),
    }


def run_mnist_batch_comparison(
    k=10,
    num_batches=1000,
    images_per_class=None,
    initial_class_count=4,
    csv_records_per_stage=None,
    num_runs=1,
    seed=7,
    sampling_constant=80,
    degree_trigger=0.5,
    gamma=2.0,
    beta=0.5,
    num_levels=3,
    hierarchy_scheduler_update_unit="edge",
    results_dir="results",
):
    """
    Compare LS24 vs SemiContracted vs Hierarchy on MNIST with class-incremental updates.

    Setting:
    - start from 4 random classes
    - add one new class at a time
    - for each new class: add within-class + cross-class edges
    - split each class update into `num_batches` batches and apply sequentially.
    """
    k = int(k)
    num_batches = int(num_batches)
    initial_class_count = int(initial_class_count)
    num_runs = int(num_runs)
    seed = int(seed)

    if num_batches <= 0:
        raise ValueError("num_batches must be > 0.")
    if csv_records_per_stage is not None:
        csv_records_per_stage = int(csv_records_per_stage)
        if csv_records_per_stage <= 0:
            raise ValueError("csv_records_per_stage must be >= 1.")
    if not (1 <= initial_class_count <= 9):
        raise ValueError("initial_class_count must be in [1, 9] for MNIST.")
    if num_runs <= 0:
        raise ValueError("num_runs must be >= 1.")

    knn_manager = KNN_Graph_Manager(
        k=k, dataset_name="mnist", images_per_class=images_per_class
    )
    knn_manager.build_or_load_graph()
    label_prop = knn_manager.graph.vertex_properties["labels"]

    all_rows = []
    final_rows = []

    for run_idx in range(num_runs):
        py_rng = random.Random(seed + run_idx)
        np_rng = np.random.default_rng(seed + 1000 + run_idx)

        current_classes, classes_to_add = randomize_mnist_classes(
            initial_class_count=initial_class_count,
            rng=py_rng,
        )
        initial_graph, _ = build_initial_graph(knn_manager, current_classes)

        if "original_indices" not in initial_graph.vp:
            raise RuntimeError(
                "initial_graph is missing required vertex property 'original_indices'."
            )

        init_true_labels = np.array(
            graph_labels_from_original_indices(initial_graph, label_prop)
        )
        k_initial = len(current_classes)

        graph_ls24 = gt.Graph(initial_graph)
        graph_semi = gt.Graph(initial_graph)
        graph_hier = gt.Graph(initial_graph)
        maybe_enable_fast_edge_lookup(graph_ls24)
        maybe_enable_fast_edge_lookup(graph_semi)
        maybe_enable_fast_edge_lookup(graph_hier)

        ls24 = LS24Runner(
            graph=graph_ls24,
            true_labels=init_true_labels,
            k=k_initial,
            sampling_constant=sampling_constant,
            degree_trigger=degree_trigger,
            gamma=gamma,
        )
        semi = SemiContractedRunner(
            graph=graph_semi,
            true_labels=init_true_labels,
            k=k_initial,
            beta=beta,
            sampling_constant=sampling_constant,
            degree_trigger=degree_trigger,
        )
        hier = HierarchyRunner(
            graph=graph_hier,
            true_labels=init_true_labels,
            k=k_initial,
            num_levels=num_levels,
            sampling_constant=sampling_constant,
            degree_trigger=degree_trigger,
            scheduler_update_unit=hierarchy_scheduler_update_unit,
        )

        local_of_original, next_local_id = initialize_shared_local_mapping(
            initial_graph=initial_graph,
            total_original_vertices=int(knn_manager.graph.num_vertices()),
        )

        labels_ls24 = list(init_true_labels)
        labels_semi = list(init_true_labels)
        labels_hier = list(init_true_labels)

        stages = build_mnist_class_update_plan(
            knn_manager=knn_manager,
            current_classes=list(current_classes),
            classes_to_add=classes_to_add,
            num_batches=num_batches,
            rng=np_rng,
        )

        print(
            f"[run {run_idx + 1}/{num_runs}] "
            f"initial_classes={current_classes}, update_stages={len(stages)}"
        )
        print(
            "+--------+-------+-------------+----------+------------+-------------+----------+------------+-------------+----------+------------+-------------+"
        )
        print(
            "| batch  | stage | num_updates | ls24_ari | ls24_upd_s | ls24_total_s | semi_ari | semi_upd_s | semi_total_s | hier_ari | hier_upd_s | hier_total_s |"
        )
        print(
            "+--------+-------+-------------+----------+------------+-------------+----------+------------+-------------+----------+------------+-------------+"
        )

        ls24_update_cum = 0.0
        semi_update_cum = 0.0
        hier_update_cum = 0.0
        ls24_query_cum = 0.0
        semi_query_cum = 0.0
        hier_query_cum = 0.0
        processed_updates = 0
        stage_idx = 0
        rows_this_run = []

        for stage in stages:
            stage_idx += 1
            active_classes = stage["active_classes"]
            k_now = len(active_classes)
            if csv_records_per_stage is None:
                stage_checkpoint_batches = all_stage_batches(len(stage["batches"]))
            else:
                stage_checkpoint_batches = choose_stage_checkpoint_batches(
                    num_batches=len(stage["batches"]),
                    num_records=csv_records_per_stage,
                )

            for batch_idx, batch_edges_original in enumerate(stage["batches"], start=1):
                batch_edges_local, new_original_ids, next_local_id = (
                    remap_batch_edges_shared(
                        batch_edges_original=batch_edges_original,
                        local_of_original=local_of_original,
                        next_local_id=next_local_id,
                    )
                )

                append_new_vertices_with_original_ids(graph_ls24, new_original_ids)
                append_new_vertices_with_original_ids(graph_semi, new_original_ids)
                append_new_vertices_with_original_ids(graph_hier, new_original_ids)

                if len(new_original_ids) > 0:
                    new_labels = [
                        label_prop[int(old_idx)] for old_idx in new_original_ids
                    ]
                    labels_ls24.extend(new_labels)
                    labels_semi.extend(new_labels)
                    labels_hier.extend(new_labels)

                ls24.true_labels = labels_ls24
                semi.true_labels = labels_semi
                hier.true_labels = labels_hier

                ls24_res = _run_one_algorithm_step(
                    runner=ls24,
                    batch_edges_local=batch_edges_local,
                    k_now=k_now,
                    query_runtime_multiplier=1.5,
                )
                semi_res = _run_one_algorithm_step(
                    runner=semi,
                    batch_edges_local=batch_edges_local,
                    k_now=k_now,
                )
                hier_res = _run_one_algorithm_step(
                    runner=hier,
                    batch_edges_local=batch_edges_local,
                    k_now=k_now,
                )

                processed_updates += int(len(batch_edges_original))
                ls24_update_cum += float(ls24_res["update_runtime_s"])
                semi_update_cum += float(semi_res["update_runtime_s"])
                hier_update_cum += float(hier_res["update_runtime_s"])

                ls24_query_cum += float(ls24_res["query_runtime_s"])
                semi_query_cum += float(semi_res["query_runtime_s"])
                hier_query_cum += float(hier_res["query_runtime_s"])

                ls24_total_cum = ls24_update_cum + ls24_query_cum
                semi_total_cum = semi_update_cum + semi_query_cum
                hier_total_cum = hier_update_cum + hier_query_cum

                if batch_idx in stage_checkpoint_batches:
                    row = {
                        "run": int(run_idx + 1),
                        "stage_idx": int(stage_idx),
                        "batch_idx_in_stage": int(batch_idx),
                        "stage_num_batches": int(len(stage["batches"])),
                        "added_classes": ",".join(stage["added_classes"]),
                        "active_classes": ",".join(active_classes),
                        "num_updates": int(processed_updates),
                        "batch_edges_original": int(len(batch_edges_original)),
                        "ls24_ari": float(ls24_res["ari"]),
                        "semi_ari": float(semi_res["ari"]),
                        "hierarchy_ari": float(hier_res["ari"]),
                        "ls24_cumulative_update_runtime_s": float(ls24_update_cum),
                        "semi_cumulative_update_runtime_s": float(semi_update_cum),
                        "hierarchy_cumulative_update_runtime_s": float(hier_update_cum),
                        "ls24_cumulative_query_runtime_s": float(ls24_query_cum),
                        "semi_cumulative_query_runtime_s": float(semi_query_cum),
                        "hierarchy_cumulative_query_runtime_s": float(hier_query_cum),
                        "ls24_cumulative_total_runtime_s": float(ls24_total_cum),
                        "semi_cumulative_total_runtime_s": float(semi_total_cum),
                        "hierarchy_cumulative_total_runtime_s": float(hier_total_cum),
                    }
                    all_rows.append(row)
                    rows_this_run.append(row)

                print(
                    f"| {batch_idx:6d} | {stage_idx:5d} | {processed_updates:11d} | "
                    f"{ls24_res['ari']:8.4f} | {ls24_update_cum:10.4f} | {ls24_total_cum:11.4f} | "
                    f"{semi_res['ari']:8.4f} | {semi_update_cum:10.4f} | {semi_total_cum:11.4f} | "
                    f"{hier_res['ari']:8.4f} | {hier_update_cum:10.4f} | {hier_total_cum:11.4f} |"
                )

        print(
            "+--------+-------+-------------+----------+------------+-------------+----------+------------+-------------+----------+------------+-------------+"
        )
        if len(rows_this_run) == 0:
            raise RuntimeError(f"Run {run_idx + 1} produced no rows.")
        final_rows.append(rows_this_run[-1])

    ls24_final = np.array([row["ls24_ari"] for row in final_rows], dtype=float)
    semi_final = np.array([row["semi_ari"] for row in final_rows], dtype=float)
    hier_final = np.array([row["hierarchy_ari"] for row in final_rows], dtype=float)

    print("Summary across runs")
    print(f"  runs={num_runs}")
    print(
        f"  final_ls24_ari_mean={np.mean(ls24_final):.4f} +/- {np.std(ls24_final):.4f}"
    )
    print(
        f"  final_semi_ari_mean={np.mean(semi_final):.4f} +/- {np.std(semi_final):.4f}"
    )
    print(
        f"  final_hierarchy_ari_mean={np.mean(hier_final):.4f} +/- {np.std(hier_final):.4f}"
    )

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        results_dir,
        (
            f"compare_ls24_vs_semicontracted_vs_hierarchy_mnist_"
            f"kNN_{k}_init_{initial_class_count}_batches_{num_batches}_runs_{num_runs}_{timestamp}.csv"
        ),
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"csv={out_path}")
    return out_path


if __name__ == "__main__":
    run_mnist_batch_comparison(
        k=50,
        num_batches=15,
        images_per_class=None,
        initial_class_count=5,
        num_runs=3,
        seed=7,
        sampling_constant=3,
        degree_trigger=0.5,
        gamma=2.0,
        beta=0.5,
        num_levels=4,
        results_dir="results-mnist",
    )
