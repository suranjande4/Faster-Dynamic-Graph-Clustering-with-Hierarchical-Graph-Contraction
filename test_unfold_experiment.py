import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import graph_tool as gt

from sbm import DynamicSBMGraphMerge
from sparsifier import DynamicGraphSparsifier
from contractedgraph import Contracted_Graph
from spectral_clustering import spectral_clustering, clusters_to_labels
from unfold_clusters import unfold_clusters


def rebuild_contracted_from_mapping(contracted_obj, H, updated_clusters):
    """Rebuild contracted_graph from updated_clusters using sparsifier H.

    This duplicates the aggregation logic from unfold_clusters but acts on an
    externally provided `updated_clusters` list of sets.
    """
    num_new = len(updated_clusters)
    n_H = H.num_vertices()
    vertex_to_cluster = np.full(n_H, -1, dtype=int)
    for cid, s in enumerate(updated_clusters):
        for v in s:
            if 0 <= int(v) < n_H:
                vertex_to_cluster[int(v)] = cid

    # Extract edges from H
    if "edge_weight" in H.edge_properties:
        hep = H.ep["edge_weight"]
        try:
            edges_H = H.get_edges([hep])
        except Exception:
            edges_H = np.array([[int(e.source()), int(e.target()), float(hep[e])] for e in H.edges()])
    else:
        edges_tmp = np.array(list(H.get_edges()))
        if edges_tmp.size == 0:
            edges_H = np.zeros((0, 3))
        else:
            edges_H = np.column_stack((edges_tmp[:, 0].astype(int), edges_tmp[:, 1].astype(int), np.ones(len(edges_tmp))))

    if edges_H.size == 0:
        new_g = gt.Graph(directed=False)
        new_g.add_vertex(num_new)
        ep = new_g.new_edge_property("double")
        new_g.edge_properties["edge_weight"] = ep
        contracted_obj.contracted_graph = new_g
        contracted_obj.contracted_node_to_full_node_map = updated_clusters
        return contracted_obj

    src = vertex_to_cluster[edges_H[:, 0].astype(int)]
    tgt = vertex_to_cluster[edges_H[:, 1].astype(int)]
    valid = (src >= 0) & (tgt >= 0)
    src = src[valid]
    tgt = tgt[valid]
    weights = edges_H[valid, 2].astype(float)

    pairs = np.column_stack((src, tgt, weights))
    unique_pairs, inv = np.unique(pairs[:, :2].astype(int), axis=0, return_inverse=True)
    total_weights = np.zeros(unique_pairs.shape[0], dtype=float)
    np.add.at(total_weights, inv, pairs[:, 2])

    new_g = gt.Graph(directed=False)
    new_g.add_vertex(num_new)
    new_ep = new_g.new_edge_property("double")
    if unique_pairs.size > 0:
        edges_and_weights = np.hstack((unique_pairs, total_weights[:, None]))
        new_g.add_edge_list(edges_and_weights, eprops=[new_ep])
    new_g.edge_properties["edge_weight"] = new_ep

    contracted_obj.contracted_graph = new_g
    contracted_obj.contracted_node_to_full_node_map = updated_clusters
    return contracted_obj


def run_test():
    # Create SBM with two clusters: sizes 20 and 40 (one twice the other)
    sizes = [200,400]
    k = len(sizes)
    p = 0.6
    q = 0.105

    sbm = DynamicSBMGraphMerge(k, sizes, p, q)
    G = sbm.graph
    true_labels = sbm.labels

    print(f"Original graph: {G.num_vertices()} vertices, {G.num_edges()} edges")

    # Build sparsifier
    spars = DynamicGraphSparsifier(G)
    spars.create_sparsifier()
    H = spars.sparsified_graph
    print(f"Sparsifier: {H.num_vertices()} vertices, {H.num_edges()} edges")

    # Run spectral clustering on the sparsified graph to obtain initial clusters
    clusters = spectral_clustering(H, 2)
    labels = clusters_to_labels(clusters)

    print(f"Spectral clusters sizes: {[len(c) for c in clusters]}")

    # Contract clusters
    cg = Contracted_Graph(G, H)
    cg.initialize(clusters, labels)
    print(f"Contracted graph has {len(cg.contracted_node_to_full_node_map)} supernodes and {cg.contracted_graph.num_edges()} edges")

    # compute volumes per contracted node (sum incident edge weights)
    ep = cg.contracted_graph.ep["edge_weight"] if "edge_weight" in cg.contracted_graph.edge_properties else None
    volumes = []
    if ep is not None:
        edges = cg.contracted_graph.get_edges([ep])
        for i in range(len(cg.contracted_node_to_full_node_map)):
            mask = (edges[:, 0].astype(int) == i) | (edges[:, 1].astype(int) == i)
            volumes.append(float(np.sum(edges[mask, 2])))
    else:
        for i in range(len(cg.contracted_node_to_full_node_map)):
            volumes.append(float(cg.contracted_graph.vertex(i).out_degree()))

    print(f"Volumes per supernode: {volumes}")

    # pick the smaller cluster (by cardinality of mapped vertices) to unfold
    sizes_map = [len(s) for s in cg.contracted_node_to_full_node_map]
    small_idx = int(np.argmin(sizes_map))
    print(f"Small cluster index: {small_idx}, size: {sizes_map[small_idx]}")

    # build updated mapping: remove the small supernode and add its vertices as singletons
    updated_map = []
    for idx, s in enumerate(cg.contracted_node_to_full_node_map):
        if idx == small_idx:
            continue
        updated_map.append(set(s))
    # add singletons for small cluster members
    for v in sorted(list(cg.contracted_node_to_full_node_map[small_idx])):
        updated_map.append(set([v]))

    # rebuild contracted graph from updated_map using edges from H
    cg = rebuild_contracted_from_mapping(cg, H, updated_map)

    print(f"After unfolding small cluster: {len(cg.contracted_node_to_full_node_map)} supernodes, {cg.contracted_graph.num_edges()} edges")


if __name__ == "__main__":
    start = time.time()
    run_test()
    print(f"Done in {time.time()-start:.2f}s")
