import csv
import os
import time
from datetime import datetime

import graph_tool as gt
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sbm import DynamicSBMGraphMerge
from sbm_compare_with_ls24 import (
    HierarchyRunner,
    LS24Runner,
    SemiContractedRunner,
    _chunk_updates,
    build_single_edge_update_stream,
)


def run_plot_comparison(
    sizes,
    p,
    q,
    num_updates,
    batch_size,
    sampling_constant,
    degree_trigger,
    gamma,
    beta,
    num_levels,
    seed_graph,
    seed_updates,
    run_id,
):
    k = len(sizes)

    np.random.seed(int(seed_graph))
    sbm = DynamicSBMGraphMerge(k=int(k), sizes=list(sizes), p=float(p), q=float(q))
    base_graph = sbm.get_graph()
    true_labels = np.array(sbm.get_labels())

    updates = build_single_edge_update_stream(
        base_graph=base_graph,
        true_labels=true_labels,
        num_updates=int(num_updates),
        seed=int(seed_updates),
    )
    if len(updates) == 0:
        raise RuntimeError("No updates were generated. Try increasing p/q or changing seeds.")

    batches = _chunk_updates(updates, batch_size=int(batch_size))

    graph_ls24 = gt.Graph(base_graph)
    graph_semi = gt.Graph(base_graph)
    graph_hier = gt.Graph(base_graph)

    ls24 = LS24Runner(
        graph=graph_ls24,
        true_labels=true_labels,
        k=int(k),
        sampling_constant=int(sampling_constant),
        degree_trigger=float(degree_trigger),
        gamma=float(gamma),
    )
    semi = SemiContractedRunner(
        graph=graph_semi,
        true_labels=true_labels,
        k=int(k),
        beta=float(beta),
        sampling_constant=int(sampling_constant),
        degree_trigger=float(degree_trigger),
    )
    hier = HierarchyRunner(
        graph=graph_hier,
        true_labels=true_labels,
        k=int(k),
        num_levels=int(num_levels),
        sampling_constant=int(sampling_constant),
        degree_trigger=float(degree_trigger),
    )

    rows = []
    processed_updates = 0

    ls24_update_cum = 0.0
    semi_update_cum = 0.0
    hier_update_cum = 0.0
    ls24_query_cum = 0.0
    semi_query_cum = 0.0
    hier_query_cum = 0.0

    print(
        f"[run {run_id}] sizes={sizes}, p={p}, q={q}, updates={len(updates)}, "
        f"batch_size={batch_size}, num_levels={num_levels}"
    )

    for batch in batches:
        t0 = time.perf_counter()
        ls24.add_edge_batch(batch)
        ls24_update_cum += time.perf_counter() - t0
        ls24_q = ls24.query()
        ls24_query_cum += float(ls24_q.get("runtime", 0.0))

        t0 = time.perf_counter()
        semi.add_edge_batch(batch)
        semi_update_cum += time.perf_counter() - t0
        semi_q = semi.query()
        semi_query_cum += float(semi_q.get("runtime", 0.0))

        t0 = time.perf_counter()
        hier.add_edge_batch(batch)
        hier_update_cum += time.perf_counter() - t0
        hier_q = hier.query()
        hier_query_cum += float(hier_q.get("runtime", 0.0))

        processed_updates += len(batch)

        rows.append(
            {
                "num_updates": int(processed_updates),
                "ls24_ari": float(ls24_q.get("ari", np.nan)),
                "semi_ari": float(semi_q.get("ari", np.nan)),
                "hierarchy_ari": float(hier_q.get("ari", np.nan)),
                "ls24_cumulative_update_runtime_s": float(ls24_update_cum),
                "semi_cumulative_update_runtime_s": float(semi_update_cum),
                "hierarchy_cumulative_update_runtime_s": float(hier_update_cum),
                "ls24_cumulative_query_runtime_s": float(ls24_query_cum),
                "semi_cumulative_query_runtime_s": float(semi_query_cum),
                "hierarchy_cumulative_query_runtime_s": float(hier_query_cum),
            }
        )

    return rows


def run_plot_comparison_multi(
    sizes,
    p,
    q,
    num_updates,
    batch_size,
    sampling_constant,
    degree_trigger,
    gamma,
    beta,
    num_levels,
    seed_graph,
    seed_updates,
    num_runs,
):
    if int(num_runs) <= 0:
        raise ValueError("num_runs must be >= 1.")

    k = len(sizes)
    print("Running LS24 vs SemiContracted vs Hierarchy (multi-run)")
    print(
        f"sizes={sizes}, p={p}, q={q}, num_updates={num_updates}, batch_size={batch_size}, "
        f"num_levels={num_levels}, num_runs={num_runs}"
    )

    all_runs = []
    for run_idx in range(int(num_runs)):
        # deterministic but distinct seeds per run
        sg = int(seed_graph) + 1000 * run_idx
        su = int(seed_updates) + 1000 * run_idx
        rows = run_plot_comparison(
            sizes=sizes,
            p=p,
            q=q,
            num_updates=num_updates,
            batch_size=batch_size,
            sampling_constant=sampling_constant,
            degree_trigger=degree_trigger,
            gamma=gamma,
            beta=beta,
            num_levels=num_levels,
            seed_graph=sg,
            seed_updates=su,
            run_id=run_idx + 1,
        )
        all_runs.append(rows)

    min_len = min(len(rows) for rows in all_runs)
    if min_len <= 0:
        raise RuntimeError("No rows were produced across runs.")
    all_runs = [rows[:min_len] for rows in all_runs]

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_csv_path = os.path.join(
        results_dir,
        (
            "plot_compare_ls24_semicontracted_hierarchy_raw_"
            f"n_{sum(sizes)}_k_{k}_updates_{num_updates}_batch_{batch_size}_runs_{num_runs}_{timestamp}.csv"
        ),
    )
    raw_keys = ["run"] + list(all_runs[0][0].keys())
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_keys)
        writer.writeheader()
        for run_idx, run_rows in enumerate(all_runs, start=1):
            for row in run_rows:
                rec = {"run": run_idx}
                rec.update(row)
                writer.writerow(rec)

    x = np.array([r["num_updates"] for r in all_runs[0]], dtype=int)
    metric_keys = [
        "ls24_ari",
        "semi_ari",
        "hierarchy_ari",
        "ls24_cumulative_update_runtime_s",
        "semi_cumulative_update_runtime_s",
        "hierarchy_cumulative_update_runtime_s",
        "ls24_cumulative_query_runtime_s",
        "semi_cumulative_query_runtime_s",
        "hierarchy_cumulative_query_runtime_s",
    ]
    metric_mean = {}
    metric_std = {}
    for key in metric_keys:
        stacked = np.stack(
            [np.array([row[key] for row in run_rows], dtype=float) for run_rows in all_runs],
            axis=0,
        )
        metric_mean[key] = np.nanmean(stacked, axis=0)
        metric_std[key] = np.nanstd(stacked, axis=0)

    agg_rows = []
    for i, updates_i in enumerate(x):
        agg_rows.append(
            {
                "num_updates": int(updates_i),
                "ls24_ari_mean": float(metric_mean["ls24_ari"][i]),
                "ls24_ari_std": float(metric_std["ls24_ari"][i]),
                "semi_ari_mean": float(metric_mean["semi_ari"][i]),
                "semi_ari_std": float(metric_std["semi_ari"][i]),
                "hierarchy_ari_mean": float(metric_mean["hierarchy_ari"][i]),
                "hierarchy_ari_std": float(metric_std["hierarchy_ari"][i]),
                "ls24_cumulative_update_runtime_s_mean": float(
                    metric_mean["ls24_cumulative_update_runtime_s"][i]
                ),
                "ls24_cumulative_update_runtime_s_std": float(
                    metric_std["ls24_cumulative_update_runtime_s"][i]
                ),
                "semi_cumulative_update_runtime_s_mean": float(
                    metric_mean["semi_cumulative_update_runtime_s"][i]
                ),
                "semi_cumulative_update_runtime_s_std": float(
                    metric_std["semi_cumulative_update_runtime_s"][i]
                ),
                "hierarchy_cumulative_update_runtime_s_mean": float(
                    metric_mean["hierarchy_cumulative_update_runtime_s"][i]
                ),
                "hierarchy_cumulative_update_runtime_s_std": float(
                    metric_std["hierarchy_cumulative_update_runtime_s"][i]
                ),
                "ls24_cumulative_query_runtime_s_mean": float(
                    metric_mean["ls24_cumulative_query_runtime_s"][i]
                ),
                "ls24_cumulative_query_runtime_s_std": float(
                    metric_std["ls24_cumulative_query_runtime_s"][i]
                ),
                "semi_cumulative_query_runtime_s_mean": float(
                    metric_mean["semi_cumulative_query_runtime_s"][i]
                ),
                "semi_cumulative_query_runtime_s_std": float(
                    metric_std["semi_cumulative_query_runtime_s"][i]
                ),
                "hierarchy_cumulative_query_runtime_s_mean": float(
                    metric_mean["hierarchy_cumulative_query_runtime_s"][i]
                ),
                "hierarchy_cumulative_query_runtime_s_std": float(
                    metric_std["hierarchy_cumulative_query_runtime_s"][i]
                ),
            }
        )

    agg_csv_path = os.path.join(
        results_dir,
        (
            "plot_compare_ls24_semicontracted_hierarchy_agg_"
            f"n_{sum(sizes)}_k_{k}_updates_{num_updates}_batch_{batch_size}_runs_{num_runs}_{timestamp}.csv"
        ),
    )
    with open(agg_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def plot_mean_std(ax, xvals, mean_vals, std_vals, label):
        ax.plot(xvals, mean_vals, label=label, linewidth=2)
        ax.fill_between(xvals, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

    plot_mean_std(
        axes[0], x, metric_mean["ls24_ari"], metric_std["ls24_ari"], "LS24"
    )
    plot_mean_std(
        axes[0], x, metric_mean["semi_ari"], metric_std["semi_ari"], "SemiContracted"
    )
    plot_mean_std(
        axes[0], x, metric_mean["hierarchy_ari"], metric_std["hierarchy_ari"], "Hierarchy"
    )
    axes[0].set_title("ARI vs Updates")
    axes[0].set_xlabel("Number of updates")
    axes[0].set_ylabel("ARI")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    plot_mean_std(
        axes[1],
        x,
        metric_mean["ls24_cumulative_update_runtime_s"],
        metric_std["ls24_cumulative_update_runtime_s"],
        "LS24",
    )
    plot_mean_std(
        axes[1],
        x,
        metric_mean["semi_cumulative_update_runtime_s"],
        metric_std["semi_cumulative_update_runtime_s"],
        "SemiContracted",
    )
    plot_mean_std(
        axes[1],
        x,
        metric_mean["hierarchy_cumulative_update_runtime_s"],
        metric_std["hierarchy_cumulative_update_runtime_s"],
        "Hierarchy",
    )
    axes[1].set_title("Cumulative Update Time")
    axes[1].set_xlabel("Number of updates")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(alpha=0.3)

    plot_mean_std(
        axes[2],
        x,
        metric_mean["ls24_cumulative_query_runtime_s"],
        metric_std["ls24_cumulative_query_runtime_s"],
        "LS24",
    )
    plot_mean_std(
        axes[2],
        x,
        metric_mean["semi_cumulative_query_runtime_s"],
        metric_std["semi_cumulative_query_runtime_s"],
        "SemiContracted",
    )
    plot_mean_std(
        axes[2],
        x,
        metric_mean["hierarchy_cumulative_query_runtime_s"],
        metric_std["hierarchy_cumulative_query_runtime_s"],
        "Hierarchy",
    )
    axes[2].set_title("Cumulative Query Time")
    axes[2].set_xlabel("Number of updates")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"LS24-style SBM Comparison (mean ± std over {num_runs} runs)",
        fontsize=12,
    )
    fig.tight_layout()

    png_path = os.path.join(
        results_dir,
        (
            "plot_compare_ls24_semicontracted_hierarchy_"
            f"n_{sum(sizes)}_k_{k}_updates_{num_updates}_batch_{batch_size}_runs_{num_runs}_{timestamp}.png"
        ),
    )
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"raw_csv={raw_csv_path}")
    print(f"agg_csv={agg_csv_path}")
    print(f"plot={png_path}")


def run_sbm_batch_comparison():
    # 4 clusters, total n=10000
    sizes = [2000, 1500, 1000, 2500]
    k = len(sizes)
    p = 0.09
    q = 0.01
    num_updates = 1000
    batch_size = 50
    seed_graph = 7
    seed_updates = 19

    sampling_constant = 80
    degree_trigger = 0.5
    gamma = 2.0
    beta = 0.5
    num_levels = 2
    num_runs = 3

    run_plot_comparison_multi(
        sizes=sizes,
        p=p,
        q=q,
        num_updates=num_updates,
        batch_size=batch_size,
        sampling_constant=sampling_constant,
        degree_trigger=degree_trigger,
        gamma=gamma,
        beta=beta,
        num_levels=num_levels,
        seed_graph=seed_graph,
        seed_updates=seed_updates,
        num_runs=num_runs,
    )


if __name__ == "__main__":
    run_sbm_batch_comparison()
