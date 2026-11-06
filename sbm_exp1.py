import csv
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import graph_tool as gt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from runners import HierarchyRunner, LS24Runner, SemiContractedRunner
from sbm import DynamicSBMGraph, DynamicSBMGraphMerge


def _edge_array(*chunks):
    arrays = []
    for chunk in chunks:
        arr = np.asarray(chunk, dtype=int)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arrays.append(arr[:, :2])

    if not arrays:
        return np.empty((0, 2), dtype=int)
    return np.vstack(arrays).astype(int, copy=False)


def _set_runner_state(runner, true_labels, num_clusters):
    runner.k = int(num_clusters)
    runner.true_labels = np.array(true_labels)


def _clustering_accuracy(true_labels, predicted_labels):
    truth = np.asarray(true_labels)
    prediction = np.asarray(predicted_labels)
    if len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("ACC requires two nonempty equal-length label vectors.")

    true_values, true_inverse = np.unique(truth, return_inverse=True)
    pred_values, pred_inverse = np.unique(prediction, return_inverse=True)
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    rows, cols = linear_sum_assignment(-contingency)
    return float(contingency[rows, cols].sum() / len(truth))


def _clustering_metrics(true_labels, predicted_labels):
    truth = np.asarray(true_labels)
    prediction = np.asarray(predicted_labels)
    return {
        "ari": float(adjusted_rand_score(truth, prediction)),
        "nmi": float(
            normalized_mutual_info_score(truth, prediction, average_method="arithmetic")
        ),
        "acc": _clustering_accuracy(truth, prediction),
    }


def _make_runners(
    *,
    base_graph,
    true_labels,
    num_clusters,
    sampling_constant,
    degree_trigger,
    gamma,
    beta,
    num_levels,
    ls24_query_policy,
):
    graph_ls24 = gt.Graph(base_graph)
    graph_semi = gt.Graph(base_graph)
    graph_hier = gt.Graph(base_graph)

    ls24 = LS24Runner(
        graph=graph_ls24,
        true_labels=true_labels,
        k=num_clusters,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        gamma=gamma,
        query_policy=ls24_query_policy,
    )
    semi = SemiContractedRunner(
        graph=graph_semi,
        true_labels=true_labels,
        k=num_clusters,
        beta=beta,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        verbose=False,
    )
    hier = HierarchyRunner(
        graph=graph_hier,
        true_labels=true_labels,
        k=num_clusters,
        num_levels=num_levels,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        verbose=False,
    )

    return {"ls24": ls24, "semi": semi, "hier": hier}


def _run_runner_step(
    runner, edges, true_labels, num_clusters, query_runtime_multiplier=1.0
):
    _set_runner_state(runner, true_labels, num_clusters)

    t0 = time.perf_counter()
    update_info = runner.add_edge_batch(edges)
    update_runtime_s = time.perf_counter() - t0

    _set_runner_state(runner, true_labels, num_clusters)
    query_info = runner.query()
    query_runtime_s = float(query_info.get("runtime", 0.0)) * float(
        query_runtime_multiplier
    )
    metrics = _clustering_metrics(true_labels, query_info["labels"])

    return {
        "num_applied": int(update_info.get("num_applied", 0)),
        **metrics,
        "update_runtime_s": float(update_runtime_s),
        "query_runtime_s": float(query_runtime_s),
    }


def _print_table_header(scenario, run_idx, iterations):
    print(f"[{scenario} run {run_idx + 1}] iterations={int(iterations)}", flush=True)
    print(
        "+------+-----+-------+----------+----------+----------+------------+-------------+----------+----------+----------+------------+-------------+----------+----------+----------+------------+-------------+",
        flush=True,
    )
    print(
        "| iter | k   | edges | ls24_ari | ls24_nmi | ls24_acc | ls24_upd_s | ls24_total_s | semi_ari | semi_nmi | semi_acc | semi_upd_s | semi_total_s | hier_ari | hier_nmi | hier_acc | hier_upd_s | hier_total_s |",
        flush=True,
    )
    print(
        "+------+-----+-------+----------+----------+----------+------------+-------------+----------+----------+----------+------------+-------------+----------+----------+----------+------------+-------------+",
        flush=True,
    )


def _print_table_row(row):
    print(
        f"| {row['iteration']:4d} | {row['num_clusters']:3d} | {row['batch_edges']:5d} | "
        f"{row['ls24_ari']:8.4f} | {row['ls24_nmi']:8.4f} | {row['ls24_acc']:8.4f} | "
        f"{row['ls24_cumulative_update_runtime_s']:10.4f} | {row['ls24_cumulative_total_runtime_s']:11.4f} | "
        f"{row['semi_ari']:8.4f} | {row['semi_nmi']:8.4f} | {row['semi_acc']:8.4f} | "
        f"{row['semi_cumulative_update_runtime_s']:10.4f} | {row['semi_cumulative_total_runtime_s']:11.4f} | "
        f"{row['hierarchy_ari']:8.4f} | {row['hierarchy_nmi']:8.4f} | {row['hierarchy_acc']:8.4f} | "
        f"{row['hierarchy_cumulative_update_runtime_s']:10.4f} | {row['hierarchy_cumulative_total_runtime_s']:11.4f} |",
        flush=True,
    )


def _print_table_footer():
    print(
        "+------+-----+-------+----------+----------+----------+------------+-------------+----------+----------+----------+------------+-------------+----------+----------+----------+------------+-------------+",
        flush=True,
    )


def _append_result_row(
    *,
    rows,
    scenario,
    run_idx,
    iteration,
    num_clusters,
    num_edges,
    results,
    cumulative,
):
    row = {
        "scenario": str(scenario),
        "run": int(run_idx + 1),
        "iteration": int(iteration + 1),
        "num_clusters": int(num_clusters),
        "batch_edges": int(num_edges),
    }

    for name, result in results.items():
        cumulative[name]["update"] += float(result["update_runtime_s"])
        cumulative[name]["query"] += float(result["query_runtime_s"])
        name_out = "hierarchy" if name == "hier" else name
        row[f"{name_out}_ari"] = float(result["ari"])
        row[f"{name_out}_nmi"] = float(result["nmi"])
        row[f"{name_out}_acc"] = float(result["acc"])
        row[f"{name_out}_num_applied"] = int(result["num_applied"])
        row[f"{name_out}_update_runtime_s"] = float(result["update_runtime_s"])
        row[f"{name_out}_query_runtime_s"] = float(result["query_runtime_s"])
        row[f"{name_out}_cumulative_update_runtime_s"] = float(
            cumulative[name]["update"]
        )
        row[f"{name_out}_cumulative_query_runtime_s"] = float(cumulative[name]["query"])
        row[f"{name_out}_cumulative_total_runtime_s"] = float(
            cumulative[name]["update"] + cumulative[name]["query"]
        )

    rows.append(row)


def _run_scenario_loop(
    *,
    scenario,
    dynamic_sbm,
    initial_num_clusters,
    iterations,
    sampling_constant,
    degree_trigger,
    gamma,
    beta,
    num_levels,
    ls24_query_policy,
    run_idx,
    edge_generator,
    verbose,
):
    base_graph = gt.Graph(dynamic_sbm.graph)
    true_labels = np.array(dynamic_sbm.labels)
    num_clusters = int(initial_num_clusters)

    if verbose:
        print(
            f"[{scenario} run {run_idx + 1}] initializing runners "
            f"(n={int(base_graph.num_vertices())}, m={int(base_graph.num_edges())}, "
            f"k={num_clusters})",
            flush=True,
        )

    runners = _make_runners(
        base_graph=base_graph,
        true_labels=true_labels,
        num_clusters=num_clusters,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        gamma=gamma,
        beta=beta,
        num_levels=num_levels,
        ls24_query_policy=ls24_query_policy,
    )
    if verbose:
        print(f"[{scenario} run {run_idx + 1}] runners ready", flush=True)

    cumulative = {name: {"update": 0.0, "query": 0.0} for name in runners}
    rows = []

    if verbose:
        _print_table_header(scenario, run_idx, iterations)

    for iteration in range(int(iterations)):
        edges_to_add, true_labels, num_clusters = edge_generator(
            iteration, num_clusters
        )
        if len(edges_to_add) > 0:
            dynamic_sbm.graph.add_edge_list(edges_to_add)

        results = {
            name: _run_runner_step(
                runner,
                edges_to_add,
                true_labels,
                num_clusters,
                query_runtime_multiplier=(1.5 if name == "ls24" else 1.0),
            )
            for name, runner in runners.items()
        }
        _append_result_row(
            rows=rows,
            scenario=scenario,
            run_idx=run_idx,
            iteration=iteration,
            num_clusters=num_clusters,
            num_edges=len(edges_to_add),
            results=results,
            cumulative=cumulative,
        )

        if verbose:
            _print_table_row(rows[-1])

    if verbose:
        _print_table_footer()
        print(flush=True)

    del runners
    gc.collect()
    return rows


def run_experiment_new_clusters(
    *,
    num_runs=1,
    iterations=10,
    n=3000,
    k=10,
    n_new=400,
    p=0.1,
    q=0.001,
    r=0.95,
    s=0.000001,
    sampling_constant=5,
    degree_trigger=0.8,
    gamma=2.0,
    beta=0.5,
    num_levels=4,
    ls24_query_policy="algorithm",
    seed=7,
    verbose=True,
):
    all_rows = []

    for run_idx in range(int(num_runs)):
        np.random.seed(int(seed) + run_idx)
        dynamic_sbm = DynamicSBMGraph(int(k), int(n), float(p), float(q))

        def edge_generator(_iteration, num_clusters):
            cluster_edges, labels = dynamic_sbm.sample_internal_edges(
                int(n_new), float(r)
            )
            random_edges = dynamic_sbm.overlay_erdos_renyi_optimized(float(s))
            edges = _edge_array(cluster_edges, random_edges)
            return edges, np.array(labels), int(num_clusters + 1)

        all_rows.extend(
            _run_scenario_loop(
                scenario="new_clusters",
                dynamic_sbm=dynamic_sbm,
                initial_num_clusters=int(k),
                iterations=int(iterations),
                sampling_constant=sampling_constant,
                degree_trigger=degree_trigger,
                gamma=gamma,
                beta=beta,
                num_levels=num_levels,
                ls24_query_policy=ls24_query_policy,
                run_idx=run_idx,
                edge_generator=edge_generator,
                verbose=verbose,
            )
        )

        del dynamic_sbm
        gc.collect()

    return all_rows


def run_experiment_merge_clusters(
    *,
    num_runs=1,
    iterations=10,
    n_large=20000,
    k_large=5,
    n_small=500,
    k_small=20,
    p=0.1,
    q=0.001,
    r=0.95,
    s=0.000001,
    sampling_constant=5,
    degree_trigger=0.5,
    gamma=2.0,
    beta=0.5,
    num_levels=4,
    ls24_query_policy="algorithm",
    seed=19,
    merge_pair_schedule="ls24_code",
    verbose=True,
):
    all_rows = []
    sizes = [int(n_large)] * int(k_large) + [int(n_small)] * int(k_small)
    initial_k = int(k_large) + int(k_small)

    for run_idx in range(int(num_runs)):
        np.random.seed(int(seed) + run_idx)
        if verbose:
            print(
                f"[merge_clusters run {run_idx + 1}] generating initial SBM "
                f"(large={int(k_large)}x{int(n_large)}, small={int(k_small)}x{int(n_small)})",
                flush=True,
            )
        dynamic_sbm = DynamicSBMGraphMerge(initial_k, sizes, float(p), float(q))
        if verbose:
            print(
                f"[merge_clusters run {run_idx + 1}] initial SBM ready "
                f"(n={int(dynamic_sbm.graph.num_vertices())}, "
                f"m={int(dynamic_sbm.graph.num_edges())})",
                flush=True,
            )
        clusters_to_merge = [initial_k - 2, initial_k - 1]
        small_cluster_labels = list(range(int(k_large), initial_k))
        np.random.shuffle(small_cluster_labels)
        merge_cursor = 0

        def edge_generator(_iteration, num_clusters):
            nonlocal merge_cursor, clusters_to_merge

            if merge_pair_schedule == "pdf_random":
                label_a = small_cluster_labels[merge_cursor]
                label_b = small_cluster_labels[merge_cursor + 1]
                merge_cursor += 2
            else:
                label_a, label_b = clusters_to_merge

            cluster_edges, labels = dynamic_sbm.merge_clusters(
                int(label_a),
                int(label_b),
                float(r),
            )
            random_edges = dynamic_sbm.overlay_erdos_renyi_optimized(float(s))
            edges = _edge_array(cluster_edges, random_edges)

            if merge_pair_schedule == "ls24_code":
                clusters_to_merge = [clusters_to_merge[0] - 2, clusters_to_merge[0] - 2]
            else:
                clusters_to_merge = [clusters_to_merge[0] - 2, clusters_to_merge[1] - 2]

            return edges, np.array(labels), int(num_clusters - 1)

        all_rows.extend(
            _run_scenario_loop(
                scenario="merge_clusters",
                dynamic_sbm=dynamic_sbm,
                initial_num_clusters=initial_k,
                iterations=int(iterations),
                sampling_constant=sampling_constant,
                degree_trigger=degree_trigger,
                gamma=gamma,
                beta=beta,
                num_levels=num_levels,
                ls24_query_policy=ls24_query_policy,
                run_idx=run_idx,
                edge_generator=edge_generator,
                verbose=verbose,
            )
        )

        del dynamic_sbm
        gc.collect()

    return all_rows


def write_rows_csv(rows, results_dir, prefix):
    if not rows:
        raise RuntimeError("No rows were produced.")

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"{prefix}_{timestamp}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def sbm_experiment_2(
    *,
    scenario="both",
    num_runs=1,
    results_dir="res-exp2",
    verbose=True,
    new_config=None,
    merge_config=None,
):
    rows = []
    new_config = dict(new_config or {})
    merge_config = dict(merge_config or {})

    if scenario in {"new", "new_clusters", "both"}:
        rows.extend(
            run_experiment_new_clusters(
                num_runs=num_runs,
                verbose=verbose,
                **new_config,
            )
        )
    if scenario in {"merge", "merge_clusters", "both"}:
        rows.extend(
            run_experiment_merge_clusters(
                num_runs=num_runs,
                verbose=verbose,
                **merge_config,
            )
        )

    out_path = write_rows_csv(
        rows=rows,
        results_dir=results_dir,
        prefix=f"sbm_experiment_2_{scenario}_runs_{int(num_runs)}",
    )
    print(f"csv={out_path}")
    return out_path


def main():
    SCENARIO = "both"  # "new", "merge", or "both"
    NUM_RUNS = 3
    RESULTS_DIR = "results/sbm_experiment_2_large_query_dominant"
    VERBOSE = True

    # New-cluster workload: frequent queries with moderate edge batches.
    NEW_CONFIG = {
        "iterations": 10,
        "n": 5000,
        "k": 10,
        "n_new": 200,
        "p": 0.1,
        "q": 0.01,
        "r": 0.75,
        "s": 0.0000002,
        "sampling_constant": 3,
        "degree_trigger": 0.5,
        "gamma": 2.0,
        "beta": 0.5,
        "num_levels": 4,
        "ls24_query_policy": "algorithm",
        "seed": 7,
    }

    # Merge workload: many small clusters are gradually merged into the graph.
    MERGE_CONFIG = {
        "iterations": 10,
        "n_large": 10000,
        "k_large": 3,
        "n_small": 200,
        "k_small": 20,
        "p": 0.1,
        "q": 0.01,
        "r": 0.65,
        "s": 0.0000001,
        "sampling_constant": 3,
        "degree_trigger": 0.5,
        "gamma": 2.0,
        "beta": 0.5,
        "num_levels": 4,
        "ls24_query_policy": "algorithm",
        "seed": 19,
        "merge_pair_schedule": "pdf_random",  # or "sequential", "ls24_code"
    }

    sbm_experiment_2(
        scenario=SCENARIO,
        num_runs=NUM_RUNS,
        results_dir=RESULTS_DIR,
        verbose=VERBOSE,
        new_config=NEW_CONFIG,
        merge_config=MERGE_CONFIG,
    )


if __name__ == "__main__":
    main()
