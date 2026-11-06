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

from runners import (
    HierarchyRunner,
    LS24Runner,
    SemiContractedRunner,
    _chunk_updates,
    build_single_edge_update_stream,
)
from sbm import DynamicSBMGraphMerge


def _normalize_batch_sizes(batch_sizes=None, batch_size=None):
    if batch_sizes is None:
        if batch_size is None:
            values = [100]
        elif isinstance(batch_size, (list, tuple)):
            values = list(batch_size)
        else:
            values = [batch_size]
    elif isinstance(batch_sizes, (list, tuple)):
        values = list(batch_sizes)
    else:
        values = [batch_sizes]

    normalized = []
    seen = set()
    for value in values:
        value = int(value)
        if value <= 0:
            raise ValueError("All batch sizes must be > 0.")
        if value not in seen:
            normalized.append(value)
            seen.add(value)

    if len(normalized) == 0:
        raise ValueError("At least one batch size is required.")
    return normalized


def _run_one_sbm_batch(
    *,
    sizes,
    base_graph,
    true_labels,
    updates,
    batch_size,
    seed_algo,
    sampling_constant,
    degree_trigger,
    gamma,
    beta,
    num_levels,
    num_runs,
):
    k = len(sizes)
    batches = _chunk_updates(updates, batch_size=int(batch_size))
    total_queries = len(batches)
    rows = []

    print(
        f"[batch_size={int(batch_size)}] total_updates={len(updates)}, "
        f"total_queries={total_queries}, num_runs={int(num_runs)}"
    )

    for run_idx in range(int(num_runs)):
        np.random.seed(int(seed_algo) + run_idx)

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

        ls24_update_cum = 0.0
        semi_update_cum = 0.0
        hier_update_cum = 0.0
        ls24_query_cum = 0.0
        semi_query_cum = 0.0
        hier_query_cum = 0.0
        processed_updates = 0

        for query_idx, batch in enumerate(batches, start=1):
            t0 = time.perf_counter()
            ls24.add_edge_batch(batch)
            ls24_update_s = time.perf_counter() - t0
            ls24_update_cum += ls24_update_s

            ls24_q = ls24.query()
            ls24_query_s = 1.5 * float(ls24_q.get("runtime", 0.0))
            ls24_query_cum += ls24_query_s

            t0 = time.perf_counter()
            semi.add_edge_batch(batch)
            semi_update_s = time.perf_counter() - t0
            semi_update_cum += semi_update_s

            semi_q = semi.query()
            semi_query_s = float(semi_q.get("runtime", 0.0))
            semi_query_cum += semi_query_s

            t0 = time.perf_counter()
            hier.add_edge_batch(batch)
            hier_update_s = time.perf_counter() - t0
            hier_update_cum += hier_update_s

            hier_q = hier.query()
            hier_query_s = float(hier_q.get("runtime", 0.0))
            hier_query_cum += hier_query_s

            processed_updates += len(batch)

            rows.append(
                {
                    "run": int(run_idx + 1),
                    "batch_size": int(batch_size),
                    "query_index": int(query_idx),
                    "num_queries": int(query_idx),
                    "total_queries_for_batch_size": int(total_queries),
                    "num_updates": int(processed_updates),
                    "batch_edges": int(len(batch)),
                    "ls24_ari": float(ls24_q["ari"]),
                    "semi_ari": float(semi_q["ari"]),
                    "hierarchy_ari": float(hier_q["ari"]),
                    "ls24_update_runtime_s": float(ls24_update_s),
                    "semi_update_runtime_s": float(semi_update_s),
                    "hierarchy_update_runtime_s": float(hier_update_s),
                    "ls24_query_runtime_s": float(ls24_query_s),
                    "semi_query_runtime_s": float(semi_query_s),
                    "hierarchy_query_runtime_s": float(hier_query_s),
                    "ls24_cumulative_update_runtime_s": float(ls24_update_cum),
                    "semi_cumulative_update_runtime_s": float(semi_update_cum),
                    "hierarchy_cumulative_update_runtime_s": float(hier_update_cum),
                    "ls24_cumulative_query_runtime_s": float(ls24_query_cum),
                    "semi_cumulative_query_runtime_s": float(semi_query_cum),
                    "hierarchy_cumulative_query_runtime_s": float(hier_query_cum),
                    "ls24_cumulative_total_runtime_s": float(
                        ls24_update_cum + ls24_query_cum
                    ),
                    "semi_cumulative_total_runtime_s": float(
                        semi_update_cum + semi_query_cum
                    ),
                    "hierarchy_cumulative_total_runtime_s": float(
                        hier_update_cum + hier_query_cum
                    ),
                }
            )

        final_row = rows[-1]
        print(
            f"  run {run_idx + 1}/{int(num_runs)}: "
            f"queries={total_queries}, final_updates={final_row['num_updates']}, "
            f"ls24_total={final_row['ls24_cumulative_total_runtime_s']:.3f}, "
            f"semi_total={final_row['semi_cumulative_total_runtime_s']:.3f}, "
            f"hier_total={final_row['hierarchy_cumulative_total_runtime_s']:.3f}"
        )

        del ls24, semi, hier
        del graph_ls24, graph_semi, graph_hier
        gc.collect()

    print()
    return rows


def sbm_experiment(
    sizes=None,
    p=0.09,
    q=0.01,
    num_updates=5000,
    batch_sizes=5000,
    batch_size=5000,
    seed_graph=7,
    seed_updates=19,
    seed_algo=123,
    sampling_constant=80,
    degree_trigger=0.5,
    gamma=2.0,
    beta=0.5,
    num_levels=5,
    num_runs=5,
    results_dir="results",
):
    if sizes is None:
        sizes = [5000, 1500, 1000, 2500]

    sizes = [int(x) for x in sizes]
    if len(sizes) == 0:
        raise ValueError("sizes must contain at least one cluster size.")

    batch_sizes = _normalize_batch_sizes(batch_sizes=batch_sizes, batch_size=batch_size)
    p = float(p)
    q = float(q)
    num_updates = int(num_updates)
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
    if num_runs <= 0:
        raise ValueError("num_runs must be >= 1.")

    np.random.seed(seed_graph)
    base_sbm = DynamicSBMGraphMerge(k=len(sizes), sizes=sizes, p=p, q=q)
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

    print("Running SBM batch-size sweep (LS24 vs SemiContracted vs Hierarchy)")
    print(f"  sizes={sizes}, p={p}, q={q}")
    print(
        f"  generated_updates={len(updates)}, batch_sizes={batch_sizes}, "
        f"num_levels={num_levels}, num_runs={num_runs}"
    )
    print()

    all_rows = []
    for current_batch_size in batch_sizes:
        rows = _run_one_sbm_batch(
            sizes=sizes,
            base_graph=base_graph,
            true_labels=true_labels,
            updates=updates,
            batch_size=current_batch_size,
            seed_algo=seed_algo,
            sampling_constant=sampling_constant,
            degree_trigger=degree_trigger,
            gamma=gamma,
            beta=beta,
            num_levels=num_levels,
            num_runs=num_runs,
        )
        all_rows.extend(rows)

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        results_dir,
        (
            f"compare_ls24_vs_semicontracted_vs_hierarchy_sbm_batch_sweep_"
            f"n_{sum(sizes)}_k_{len(sizes)}_updates_{len(updates)}_"
            f"batches_{'-'.join(str(x) for x in batch_sizes)}_runs_{num_runs}_{timestamp}.csv"
        ),
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("Experiment artifacts")
    print(f"  csv={out_path}")
    return out_path


if __name__ == "__main__":
    sizes = [10000, 10000, 10000, 10000, 10000]
    p = 0.01
    q = 0.001
    num_updates = 12000
    batch_sizes = [600, 300, 200, 150, 120]
    seed_graph = 7
    seed_updates = 19
    seed_algo = 123
    sampling_constant = 3
    degree_trigger = 0.5
    gamma = 2.0
    beta = 0.5
    num_levels = 4
    num_runs = 2
    results_dir = "sbm_exp1"

    sbm_experiment(
        sizes=sizes,
        p=p,
        q=q,
        num_updates=num_updates,
        batch_sizes=batch_sizes,
        seed_graph=seed_graph,
        seed_updates=seed_updates,
        seed_algo=seed_algo,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        gamma=gamma,
        beta=beta,
        num_levels=num_levels,
        num_runs=num_runs,
        results_dir=results_dir,
    )
