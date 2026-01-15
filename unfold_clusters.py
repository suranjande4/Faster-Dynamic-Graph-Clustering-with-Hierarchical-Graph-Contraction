import numpy as np
import graph_tool as gt
import typing

"""Unfold clusters procedure.

This module provides `unfold_clusters` which takes a contracted-graph object
(expected to expose `contracted_graph` (graph_tool.Graph),
`contracted_node_to_full_node_map` (list of sets of original vertex indices)),
and a sparsifier graph `H` (graph_tool.Graph with edge property `edge_weight`).

The function unfolds any super-vertex whose volume in the contracted graph
is >= `alpha`. After unfolding it rebuilds the contracted graph by
aggregating edges from `H` between the updated clusters.
"""


def unfold_clusters(contracted_graph_obj, H: gt.Graph, alpha: float):
    """Unfold clusters in `contracted_graph_obj` using sparsifier `H`.

    Parameters
    - contracted_graph_obj: an object providing at least these attributes:
        - `contracted_graph`: a graph_tool.Graph representing the contracted graph
            and having an edge property named "edge_weight".
        - `contracted_node_to_full_node_map`: list of sets. Each set contains
            original-graph vertex indices that form that super-vertex.
    - H: sparsifier graph (graph_tool.Graph). Should have edge property
         named "edge_weight" (if missing, weights default to 1.0).
    - alpha: volume threshold; super-vertices with volume >= alpha are unfolded.

    Returns: the modified `contracted_graph_obj` (mutated in-place and also returned).
    """

    # Read current clusters and the contracted graph
    clusters = list(contracted_graph_obj.contracted_node_to_full_node_map)
    cg = contracted_graph_obj.contracted_graph

    # Edge weights on contracted graph
    if "edge_weight" in cg.edge_properties:
        ep = cg.ep["edge_weight"]
        # get numeric edges array from contracted graph: (u, v, weight)
        try:
            edges_tilde = cg.get_edges([ep])
        except Exception:
            # fallback: iterate edges
            edges_tilde = np.array([[int(e.source()), int(e.target()), float(ep[e])] for e in cg.edges()])
    else:
        # no edge weights -> treat every edge weight as 1
        try:
            edges_tmp = np.array(list(cg.get_edges()))
            if edges_tmp.size == 0:
                edges_tilde = np.zeros((0, 3))
            else:
                edges_tilde = np.column_stack((edges_tmp[:, 0].astype(int), edges_tmp[:, 1].astype(int), np.ones(len(edges_tmp))))
        except Exception:
            edges_tilde = np.zeros((0, 3))

    # Identify clusters to unfold (volume >= alpha)
    unfolded_indices = []
    num_clusters_before = len(clusters)
    for i in range(num_clusters_before):
        if edges_tilde.size == 0:
            vol = 0.0
        else:
            mask = (edges_tilde[:, 0].astype(int) == i) | (edges_tilde[:, 1].astype(int) == i)
            vol = float(np.sum(edges_tilde[mask, 2]))

        if vol >= alpha:
            unfolded_indices.append(i)

    if len(unfolded_indices) == 0:
        # nothing to do
        return contracted_graph_obj

    # Build updated cluster list: keep non-unfolded clusters in original order
    updated_clusters = []
    unfolded_vertex_lists = []
    for idx, cluster in enumerate(clusters):
        if idx in unfolded_indices:
            unfolded_vertex_lists.append(sorted(list(cluster)))
        else:
            updated_clusters.append(set(cluster))

    # Append singletons for each vertex from unfolded clusters (as non-contracted)
    for vertex_list in unfolded_vertex_lists:
        for v in vertex_list:
            updated_clusters.append(set([v]))

    # Rebuild the contracted graph by aggregating edges from H according to updated_clusters
    num_new = len(updated_clusters)
    # Prepare mapping from original vertex id -> new cluster id
    # We assume original vertex ids are in range [0, H.num_vertices()-1]
    n_H = H.num_vertices()
    vertex_to_cluster = np.full(n_H, -1, dtype=int)
    for cid, s in enumerate(updated_clusters):
        for v in s:
            if 0 <= int(v) < n_H:
                vertex_to_cluster[int(v)] = cid

    # Extract edges and weights from H
    if "edge_weight" in H.edge_properties:
        hep = H.ep["edge_weight"]
        try:
            edges_H = H.get_edges([hep])
        except Exception:
            edges_H = np.array([[int(e.source()), int(e.target()), float(hep[e])] for e in H.edges()])
    else:
        try:
            edges_tmp = np.array(list(H.get_edges()))
            if edges_tmp.size == 0:
                edges_H = np.zeros((0, 3))
            else:
                edges_H = np.column_stack((edges_tmp[:, 0].astype(int), edges_tmp[:, 1].astype(int), np.ones(len(edges_tmp))))
        except Exception:
            edges_H = np.zeros((0, 3))

    if edges_H.size == 0:
        # Create empty contracted graph with the required number of vertices
        new_g = gt.Graph(directed=False)
        new_g.add_vertex(num_new)
        new_ep = new_g.new_edge_property("double")
        new_g.edge_properties["edge_weight"] = new_ep
        contracted_graph_obj.contracted_graph = new_g
        contracted_graph_obj.contracted_node_to_full_node_map = updated_clusters
        return contracted_graph_obj

    # Map each H-edge to cluster pair
    src_clusters = vertex_to_cluster[edges_H[:, 0].astype(int)]
    tgt_clusters = vertex_to_cluster[edges_H[:, 1].astype(int)]

    # Drop edges with unknown mapping
    valid_mask = (src_clusters >= 0) & (tgt_clusters >= 0)
    if not np.all(valid_mask):
        src_clusters = src_clusters[valid_mask]
        tgt_clusters = tgt_clusters[valid_mask]
        weights = edges_H[valid_mask, 2].astype(float)
    else:
        weights = edges_H[:, 2].astype(float)

    if src_clusters.size == 0:
        # no valid edges
        new_g = gt.Graph(directed=False)
        new_g.add_vertex(num_new)
        new_ep = new_g.new_edge_property("double")
        new_g.edge_properties["edge_weight"] = new_ep
        contracted_graph_obj.contracted_graph = new_g
        contracted_graph_obj.contracted_node_to_full_node_map = updated_clusters
        return contracted_graph_obj

    cluster_pairs_and_weights = np.column_stack((src_clusters[:, None], tgt_clusters[:, None], weights[:, None]))

    # Aggregate weights for identical cluster pairs
    unique_pairs, inv = np.unique(cluster_pairs_and_weights[:, :2].astype(int), axis=0, return_inverse=True)
    total_weights = np.zeros(unique_pairs.shape[0], dtype=float)
    np.add.at(total_weights, inv, cluster_pairs_and_weights[:, 2])

    # Build new contracted graph
    new_g = gt.Graph(directed=False)
    new_g.add_vertex(num_new)
    new_ep = new_g.new_edge_property("double")

    # Add aggregated edges
    # Ensure unique_pairs and total_weights align
    if unique_pairs.size > 0:
        edges_and_weights = np.hstack((unique_pairs, total_weights[:, None]))
        # add_edge_list will create edges (u, v, weight)
        new_g.add_edge_list(edges_and_weights, eprops=[new_ep])

    new_g.edge_properties["edge_weight"] = new_ep

    # Update object
    contracted_graph_obj.contracted_graph = new_g
    contracted_graph_obj.contracted_node_to_full_node_map = updated_clusters

    return contracted_graph_obj


__all__ = ["unfold_clusters"]
