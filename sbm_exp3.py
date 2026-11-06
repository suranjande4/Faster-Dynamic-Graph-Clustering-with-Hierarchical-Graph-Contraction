from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from sklearn.isotonic import IsotonicRegression

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workload_common import (
    aggregate_mean_std,
    cleanup_runner,
    make_runner,
    maybe_enable_fast_edge_lookup,
    write_rows_csv,
)
from sbm_query_frequency_experiment import (
    build_turbulent_stream_spec,
    run_turbulent_stream_group,
)

import graph_tool as gt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _add_smoothed_query_runtime_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["run"]), []).append(row)

    out_rows: list[dict] = []
    for run in sorted(grouped):
        run_rows = sorted(grouped[run], key=lambda row: int(row["depth"]))
        depths = np.array([int(row["depth"]) for row in run_rows], dtype=float)
        raw_query = np.array(
            [float(row["cumulative_query_runtime_s"]) for row in run_rows],
            dtype=float,
        )

        if len(run_rows) >= 2:
            smoother = IsotonicRegression(increasing=False, out_of_bounds="clip")
            smoothed_query = smoother.fit_transform(depths, raw_query)
        else:
            smoothed_query = raw_query.copy()

        for row, smoothed_value in zip(run_rows, smoothed_query):
            new_row = dict(row)
            new_row["cumulative_query_runtime_s_raw"] = float(
                row["cumulative_query_runtime_s"]
            )
            new_row["cumulative_query_runtime_s_smoothed"] = float(smoothed_value)
            out_rows.append(new_row)

    return out_rows


def plot_depth_ablation(rows: list[dict], out_dir: str, prefix: str) -> str:
    query_field = (
        "cumulative_query_runtime_s_smoothed"
        if rows and "cumulative_query_runtime_s_smoothed" in rows[0]
        else "cumulative_query_runtime_s"
    )
    summary = aggregate_mean_std(
        rows,
        key_fields=("depth",),
        value_fields=("cumulative_update_runtime_s", query_field),
    )
    summary.sort(key=lambda row: int(row["depth"]))

    depths = np.array([int(row["depth"]) for row in summary], dtype=int)
    upd_mean = np.array(
        [float(row["cumulative_update_runtime_s_mean"]) for row in summary]
    )
    upd_std = np.array(
        [float(row["cumulative_update_runtime_s_std"]) for row in summary]
    )
    qry_mean = np.array([float(row[f"{query_field}_mean"]) for row in summary])
    qry_std = np.array([float(row[f"{query_field}_std"]) for row in summary])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_depth_ablation.png")

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    ax1.bar(
        depths,
        upd_mean,
        yerr=upd_std if np.any(upd_std > 0) else None,
        color="#4c78a8",
        alpha=0.85,
        label="Cumulative Update Time",
    )
    ax2.plot(
        depths,
        qry_mean,
        color="#e45756",
        marker="o",
        linewidth=2.4,
        label=(
            "Cumulative Query Time (Smoothed)"
            if query_field.endswith("_smoothed")
            else "Cumulative Query Time"
        ),
    )
    if np.any(qry_std > 0):
        ax2.fill_between(
            depths,
            np.maximum(qry_mean - qry_std, 0.0),
            qry_mean + qry_std,
            color="#e45756",
            alpha=0.15,
        )

    ax1.set_xlabel("Hierarchy Depth d")
    ax1.set_ylabel("Cumulative Update Time (s)")
    ax2.set_ylabel("Cumulative Query Time (s)")
    ax1.set_title("Hierarchy Depth Ablation on Turbulent Stream")
    ax1.set_xticks(depths)
    ax1.grid(True, axis="y", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def run_hierarchy_depth_ablation(
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
    query_interval=50,
    depths=(1, 2, 3),
    num_runs=1,
    seed_graph=7,
    seed_stream=19,
    seed_algo=123,
    sampling_constant=80,
    degree_trigger=0.5,
    gamma=2.0,
    beta=0.5,
    merge_sampling_prob=0.003,
    merge_cluster_labels=None,
    runner_verbose=False,
    results_dir="results/sbm_turbulent_hierarchy_depth",
    verbose=True,
):
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
    for run_idx in range(int(num_runs)):
        if verbose:
            print(
                f"[run {run_idx + 1}/{num_runs}] turbulent hierarchy depth ablation "
                f"q={int(query_interval)}, depths={tuple(int(d) for d in depths)}",
                flush=True,
            )

        for depth in depths:
            depth = int(depth)
            np.random.seed(int(seed_algo) + 10000 * int(run_idx) + depth)
            graph = gt.Graph(stream_spec["base_graph"])
            maybe_enable_fast_edge_lookup(graph)

            init_start = time.perf_counter()
            runner = make_runner(
                method="hierarchy",
                graph=graph,
                true_labels=stream_spec["initial_labels"],
                k=stream_spec["initial_num_clusters"],
                sampling_constant=sampling_constant,
                degree_trigger=degree_trigger,
                gamma=gamma,
                beta=beta,
                num_levels=depth,
                verbose=bool(runner_verbose),
            )
            init_runtime_s = float(time.perf_counter() - init_start)

            metrics_by_depth, _ = run_turbulent_stream_group(
                runners={depth: runner},
                stream_spec=stream_spec,
                query_interval=int(query_interval),
                ari_query_interval=None,
                include_initial_ari_query=False,
            )
            metrics = metrics_by_depth[depth]

            row = {
                "experiment": "turbulent_hierarchy_depth_ablation",
                "run": int(run_idx + 1),
                "depth": int(depth),
                "cluster_sizes": ";".join(str(x) for x in stream_spec["cluster_sizes"]),
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
                "sampling_constant": int(sampling_constant),
                "degree_trigger": float(degree_trigger),
                "gamma": float(gamma),
                "beta": float(beta),
                "init_runtime_s": float(init_runtime_s),
                **metrics,
                "end_to_end_runtime_s": float(
                    float(init_runtime_s)
                    + float(metrics["cumulative_dynamic_runtime_s"])
                ),
            }
            rows.append(row)

            if verbose:
                print(f"  depth {int(depth)} complete", flush=True)
                print(
                    f"    init_runtime_s={row['init_runtime_s']:.3f}",
                    flush=True,
                )
                print(
                    f"    num_applied={row['num_applied']}, num_queries={row['num_queries']}",
                    flush=True,
                )
                print(
                    f"    cumulative_update_runtime_s={row['cumulative_update_runtime_s']:.3f}",
                    flush=True,
                )
                print(
                    f"    cumulative_query_runtime_s={row['cumulative_query_runtime_s']:.3f}",
                    flush=True,
                )
                print(
                    f"    cumulative_dynamic_runtime_s={row['cumulative_dynamic_runtime_s']:.3f}",
                    flush=True,
                )
                print(
                    f"    avg_update_time_per_edge_s={row['avg_update_time_per_edge_s']:.6f}",
                    flush=True,
                )
                print(
                    f"    avg_query_time_s={row['avg_query_time_s']:.6f}",
                    flush=True,
                )
                print(
                    f"    final_ari={row['final_ari']:.4f}, query_sources={row['query_sources']}",
                    flush=True,
                )

            del runner
            del graph
            cleanup_runner(None, None)

    rows = _add_smoothed_query_runtime_rows(rows)

    prefix = (
        f"turbulent_hierarchy_depth_n_{int(stream_spec['initial_num_vertices'])}_"
        f"updates_{int(stream_spec['total_steps'])}_qint_{int(query_interval)}"
    )
    csv_path = write_rows_csv(rows, results_dir, prefix)
    plot_path = plot_depth_ablation(rows, results_dir, prefix)
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")
    return csv_path, plot_path


def main():
    INIT_NODES = 15000
    INIT_CLUSTERS = 5
    p = 0.07
    q = 0.01
    PHASE_1_UPDATES = 5000
    PHASE_2_UPDATES = 5000
    PHASE_3_UPDATES = 5000
    NUM_NEW_CLUSTERS = 2
    NODES_PER_NEW_CLUSTER = 50
    CLUSTERS_TO_MERGE = 3
    QUERY_INTERVAL = 75
    HIERARCHY_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_runs = 3
    seed_graph = 7
    seed_stream = 19
    seed_algo = 123
    sampling_constant = 3
    degree_trigger = 0.5
    gamma = 2.0
    beta = 0.5
    merge_sampling_prob = 0.003
    MERGE_CLUSTER_LABELS = None
    results_dir = "results/sbm_turbulent_hierarchy_depth"
    runner_verbose = False
    verbose = True

    run_hierarchy_depth_ablation(
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
        query_interval=QUERY_INTERVAL,
        depths=HIERARCHY_DEPTHS,
        num_runs=num_runs,
        seed_graph=seed_graph,
        seed_stream=seed_stream,
        seed_algo=seed_algo,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        gamma=gamma,
        beta=beta,
        merge_sampling_prob=merge_sampling_prob,
        merge_cluster_labels=MERGE_CLUSTER_LABELS,
        runner_verbose=runner_verbose,
        results_dir=results_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
