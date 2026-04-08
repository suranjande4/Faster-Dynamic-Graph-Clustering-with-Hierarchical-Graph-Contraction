

import numpy as np
import time
from graph_tool import Graph
import graph_tool.generation as gt_gen

# Import your modules
from spectral_clustering import run_spectral_clustering
from sbm import DynamicSBMGraphMerge

def run_sbm_experiment():
    """Run SBM experiment with incremental edge addition"""

    # Parameters
    n = 500  # total vertices
    k = 3     # number of clusters
    p = 0.5   # intra-cluster edge probability
    q = 0.1  # inter-cluster edge probability

    # Divide vertices equally among clusters
    base_size = n // k
    remainder = n % k
    sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]

    print(f"Creating SBM graph with n={n}, k={k}, sizes={sizes}, p={p}, q={q}")

    # Create initial SBM graph
    sbm_graph = DynamicSBMGraphMerge(k, sizes, p, q)
    graph = sbm_graph.graph
    true_labels = sbm_graph.labels

    print(f"Initial graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")

    # Run initial spectral clustering
    print("\nInitial clustering:")
    clusters, labels, ari, runtime = run_spectral_clustering(graph, true_labels, k)
    print(".4f")
    print(f"Cluster sizes: {[len(c) for c in clusters]}")

    # Now add random edges incrementally
    num_steps = 2  # Number of steps to add edges
    edges_per_step = 10  # Number of edges to add per step

    for step in range(1, num_steps + 1):
        print(f"\nStep {step}: Adding {edges_per_step} random edges...")

        # Add random edges using Erdős–Rényi overlay
        # Use small s to add approximately edges_per_step edges
        s = edges_per_step / (n * (n-1) / 2)  # probability for ER model
        new_edges = sbm_graph.overlay_erdos_renyi_optimized(s)

        # Add the edges to the graph
        for u, v in new_edges:
            graph.add_edge(u, v)

        print(f"Graph now has {graph.num_edges()} edges")

        # Run spectral clustering again
        clusters, labels, ari, runtime = run_spectral_clustering(graph, true_labels, k)
        print(".4f")
        print(f"Cluster sizes: {[len(c) for c in clusters]}")
        print(".2f")

if __name__ == "__main__":
    run_sbm_experiment()