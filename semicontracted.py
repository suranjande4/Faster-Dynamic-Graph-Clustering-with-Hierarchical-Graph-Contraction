from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import graph_tool as gt
import time
from contractedgraph import Contracted_Graph
from spectral_clustering import clusters_to_labels, spectral_clustering


class SemiContractedGraph:
    """
    Semi-contracted dynamic graph.

    Core logic:
    - Maintain graph + sparsifier + contracted representation.
    - Maintain alpha = number of successful updates since last full reconstruction.
    - Unfold a supernode when volume(supernode) <= alpha.
    - Keep unfolded graph until next full reconstruction.
    """

    def __init__(
        self,
        original_graph: gt.Graph,
        sparsifier,
        degree_trigger: float = 0.5,
        contracted: Optional[Contracted_Graph] = None,
        allow_self_loops: bool = False,
        count_existing_edge_updates: bool = False,
    ) -> None:
        # Keep reference to the main dynamic graph.
        self.original_graph = original_graph

        # Keep reference to the dynamic sparsifier object.
        self.sparsifier = sparsifier

        # Threshold used by underlying Contracted_Graph.update logic.
        self.degree_trigger = float(degree_trigger)
        # Higher hierarchy levels may intentionally propagate same-supernode updates
        # as self-loop mass.
        self.allow_self_loops = bool(allow_self_loops)
        # When True, an incoming edge on an existing weighted edge is treated as
        # one more unit of edge mass (weight +1), rather than being ignored.
        self.count_existing_edge_updates = bool(count_existing_edge_updates)

        # Read current sparsified graph H_t from sparsifier.
        self.sparsified_graph = self.sparsifier.get_sparsified_graph()

        # Build or reuse contracted graph state.
        if contracted is None:
            # Fresh contracted wrapper if caller did not provide one.
            self.contracted = Contracted_Graph(
                self.original_graph,
                self.sparsified_graph,
                degree_trigger=self.degree_trigger,
            )
        else:
            # Reuse caller-provided contracted object.
            self.contracted = contracted
            # Ensure it points to the same current sparsifier graph.
            self.contracted.sparsified_graph = self.sparsified_graph

       
        self.alpha = 0  #unfolding constant
        self.updates_since_rebuild = 0

        # Mark initialized if contracted has reconstruction metadata already.
        self.is_initialized = hasattr(self.contracted, "degree_array_at_construction")

        # Remember last cluster count used by reconstruction.
        self.last_num_clusters: Optional[int] = None

    @staticmethod
    def _degree_array(graph: gt.Graph) -> np.ndarray:
        if "edge_weight" in graph.edge_properties:
            return graph.degree_property_map("total", weight=graph.ep["edge_weight"]).a.copy()
        return graph.degree_property_map("total").a.copy()

    @staticmethod
    def _safe_weighted_edges(graph: gt.Graph) -> np.ndarray:
        """
        Return edges as Nx3 array [u, v, w].

        If graph has no edge_weight property, assign w=1.
        If graph is empty, return shape (0, 3).
        """
        # Branch: weighted graph path.
        if "edge_weight" in graph.edge_properties:
            ep = graph.ep["edge_weight"]
            try:
                # Fast path: ask graph-tool for numeric edge table with weights.
                edges = graph.get_edges([ep])
            except Exception:
                # Fallback path: iterate edges manually.
                edges = np.array(
                    [[int(e.source()), int(e.target()), float(ep[e])] for e in graph.edges()]
                )
        else:
            # Unweighted graph path.
            base = np.array(list(graph.get_edges()))
            if base.size == 0:
                # Keep exact expected shape.
                return np.zeros((0, 3))
            # Add synthetic weight column of ones.
            edges = np.column_stack(
                (base[:, 0].astype(int), base[:, 1].astype(int), np.ones(base.shape[0]))
            )

        # Handle empty result uniformly.
        if edges.size == 0:
            return np.zeros((0, 3))

        # Normalize dtypes and layout.
        return np.column_stack(
            (
                edges[:, 0].astype(int),
                edges[:, 1].astype(int),
                edges[:, 2].astype(float),
            )
        )

    @staticmethod
    def _norm_edge(u: int, v: int) -> Tuple[int, int]:
        """
        Canonicalize undirected edge endpoints to (min, max).
        """
        u = int(u)
        v = int(v)
        return (u, v) if u <= v else (v, u)

    def _sync_graph_refs(self) -> None:
        """
        Refresh internal references to current sparsifier output.
        """
        # Read latest H_t from sparsifier object.
        self.sparsified_graph = self.sparsifier.get_sparsified_graph()
        # Keep contracted wrapper synced with same H_t.
        self.contracted.sparsified_graph = self.sparsified_graph

    def _ensure_vertices_exist(self, max_vertex_idx: int) -> List[int]:
        """
        Ensure original graph has vertices [0..max_vertex_idx].

        Returns newly created vertex indices.
        """
        if max_vertex_idx < 0:
            return []

        created = []
        n = int(self.original_graph.num_vertices())
        for idx in range(n, max_vertex_idx + 1):
            self.original_graph.add_vertex()
            created.append(idx)
        return created

    def initialize(
        self,
        num_clusters: Optional[int] = None,
        clusters: Optional[Sequence[Sequence[int]]] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        
        self.full_reconstruct(num_clusters=num_clusters, clusters=clusters, labels=labels)

    def full_reconstruct(
        self,
        num_clusters: Optional[int] = None,
        clusters: Optional[Sequence[Sequence[int]]] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Rebuild contracted state from current sparsifier and reset alpha counters.
        """
        # Ensure references point to latest sparsified graph.
        self._sync_graph_refs()

        # If clusters are not provided, compute them from spectral clustering on H_t.
        if clusters is None:
            if num_clusters is None:
                if self.last_num_clusters is None:
                    raise ValueError("num_clusters must be provided for first reconstruction.")
                num_clusters = self.last_num_clusters
            clusters = spectral_clustering(self.sparsified_graph, int(num_clusters))
            labels = clusters_to_labels(clusters, num_data_points=self.sparsified_graph.num_vertices())
        else:
            # If clusters provided but labels missing, derive labels.
            if labels is None:
                labels = clusters_to_labels(
                    clusters, num_data_points=self.sparsified_graph.num_vertices()
                )

        # Build a fresh contracted wrapper from (G_t, H_t).
        self.contracted = Contracted_Graph(
            self.original_graph,
            self.sparsified_graph,
            degree_trigger=self.degree_trigger,
        )
        self.contracted.initialize(list(clusters), np.array(labels))

        # Reset  counters after full reconstruction.
        self.alpha = 0
        self.updates_since_rebuild = 0
        self.is_initialized = True

        # Remember cluster count if provided by caller.
        if num_clusters is not None:
            self.last_num_clusters = int(num_clusters)

    def cluster_volumes(self) -> np.ndarray:
        """
        Compute contracted-node volume as weighted degree in the contracted graph.
        """
        cg = self.contracted.contracted_graph
        if int(cg.num_vertices()) == 0:
            return np.zeros(0, dtype=float)
        return self._degree_array(cg).astype(float)

    def _rebuild_semicontracted_from_mapping(
        self, updated_clusters: Sequence[Sequence[int]]
    ) -> None:
        """
        Rebuild contracted graph weights using a fixed cluster mapping.

        This is used to preserve current folded/unfolded state.
        """
        # Normalize mapping into list[set[int]] for safe operations.
        clusters = [set(int(v) for v in cluster) for cluster in updated_clusters]
        num_new = len(clusters)
        n_h = int(self.sparsified_graph.num_vertices())

        # Map original vertex id -> cluster id in semi-contracted graph.
        vertex_to_cluster = np.full(n_h, -1, dtype=int)
        for cid, cluster in enumerate(clusters):
            for v in cluster:
                if 0 <= int(v) < n_h:
                    vertex_to_cluster[int(v)] = cid

        # Read weighted edges from H_t.
        edges_h = self._safe_weighted_edges(self.sparsified_graph)

        # Create fresh contracted graph container.
        new_g = gt.Graph(directed=False)
        new_g.add_vertex(num_new)
        new_ep = new_g.new_edge_property("double")

        if edges_h.size > 0:
            # Reindex edge endpoints into new cluster ids.
            src = vertex_to_cluster[edges_h[:, 0].astype(int)]
            tgt = vertex_to_cluster[edges_h[:, 1].astype(int)]
            valid = (src >= 0) & (tgt >= 0)

            src = src[valid]
            tgt = tgt[valid]
            w = edges_h[valid, 2].astype(float)

            if src.size > 0:
                # Aggregate duplicate cluster pairs by summing weights.
                pairs = np.column_stack((src, tgt)).astype(int)
                unique_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
                total_w = np.zeros(unique_pairs.shape[0], dtype=float)
                np.add.at(total_w, inv, w)

                # Add weighted edges to new contracted graph.
                edges_and_weights = np.hstack((unique_pairs, total_w[:, None]))
                new_g.add_edge_list(edges_and_weights, eprops=[new_ep])

        # Attach weight property map.
        new_g.edge_properties["edge_weight"] = new_ep

        # Replace contracted state with rebuilt one.
        self.contracted.contracted_graph = new_g
        self.contracted.contracted_node_to_full_node_map = clusters
        self.contracted.og_num_contract_nodes = len(clusters)
        self.contracted.degree_array = self._degree_array(self.original_graph)
        self.contracted.degree_array_at_construction = self.contracted.degree_array.copy()

    def unfold_by_alpha(self) -> int:
        """
        Unfold every supernode with volume <= alpha.

        Returns number of supernodes unfolded in this call.
        """
        current_map = self.contracted.contracted_node_to_full_node_map
        if len(current_map) == 0:
            return 0

        volumes = self.cluster_volumes()

        # Only unfold true supernodes (size > 1).
        to_unfold = [
            i
            for i, cluster in enumerate(current_map)
            if len(cluster) > 1 and volumes[i] <= float(self.alpha)
        ]

        if len(to_unfold) == 0:
            return 0

        # Build updated mapping by splitting selected supernodes into singletons.
        to_unfold_set = set(to_unfold)
        updated = []
        for idx, cluster in enumerate(current_map):
            if idx in to_unfold_set:
                for v in sorted(cluster):
                    updated.append(set([int(v)]))
            else:
                updated.append(set(int(v) for v in cluster))

        # Rebuild contracted graph according to updated mapping.
        self._rebuild_semicontracted_from_mapping(updated)
        return len(to_unfold)

    def update(self, new_edges: Iterable[Tuple[int, int]]) -> dict:
        """
        Batch insertion update (add-only).

        Procedure:
        1) normalize + deduplicate incoming edges
        2) apply updates to G:
           - add truly new edges
           - optionally count repeated updates as weight increments on weighted graphs
        3) alpha += number of effective updates
        5) update sparsifier + contracted graph once
        6) unfold by alpha once
        """
        if not self.is_initialized:
            raise RuntimeError("Call initialize(...) before applying updates.")

        update_start = time.perf_counter()

        raw_edges = list(new_edges)
        if len(raw_edges) == 0:
            return {
                "changed": False,
                "attempted_edges": 0,
                "unique_edges": 0,
                "applied_edges": [],
                "num_applied": 0,
                "created_vertices": [],
                "alpha": self.alpha,
                "updates_since_rebuild": self.updates_since_rebuild,
                "unfolded_clusters": 0,
                "maintenance_runtime_s": 0.0,
                "unfold_runtime_s": 0.0,
                "update_runtime_s": 0.0,
            }

        # Ensure newly referenced vertices exist.
        max_vertex_idx = max(max(int(u), int(v)) for u, v in raw_edges)
        created_vertices = self._ensure_vertices_exist(max_vertex_idx)

        # Normalize + deduplicate.
        unique_edges = []
        seen = set()
        for edge in raw_edges:
            u, v = self._norm_edge(int(edge[0]), int(edge[1]))
            if u == v and not self.allow_self_loops:
                continue
            if (u, v) in seen:
                continue
            seen.add((u, v))
            unique_edges.append((u, v))

        has_edge_weights = "edge_weight" in self.original_graph.edge_properties
        weight_prop = self.original_graph.ep["edge_weight"] if has_edge_weights else None

        # Keep track of topological inserts and all effective updates.
        edges_to_add = []
        effective_edges = []
        for u, v in unique_edges:
            existing = self.original_graph.edge(u, v)
            if existing is None:
                edges_to_add.append((u, v))
                effective_edges.append((u, v))
            elif self.count_existing_edge_updates and has_edge_weights:
                weight_prop[existing] += 1.0
                effective_edges.append((u, v))

        # Add all truly new edges.
        if len(edges_to_add) > 0:
            if has_edge_weights:
                for u, v in edges_to_add:
                    e = self.original_graph.add_edge(u, v)
                    weight_prop[e] = 1.0
            else:
                self.original_graph.add_edge_list(np.array(edges_to_add, dtype=int))

        num_applied = len(effective_edges)
        if num_applied > 0:
            self.alpha += num_applied
            self.updates_since_rebuild += num_applied

        # Keep H and semi-contracted state synced.
        incoming = (
            np.array(effective_edges, dtype=int)
            if num_applied > 0
            else np.empty((0, 2), dtype=int)
        )
        self.sparsifier.update_sparsifier(incoming, verbose=False)
        self._sync_graph_refs()
        self.contracted.update(incoming)

        unfold_start = time.perf_counter()
        unfolded = self.unfold_by_alpha()
        unfold_runtime_s = time.perf_counter() - unfold_start
        update_runtime_s = time.perf_counter() - update_start
        maintenance_runtime_s = max(0.0, update_runtime_s - unfold_runtime_s)

        return {
            "changed": bool(num_applied > 0 or len(created_vertices) > 0),
            "attempted_edges": len(raw_edges),
            "unique_edges": len(unique_edges),
            "applied_edges": effective_edges,
            "num_applied": num_applied,
            "created_vertices": created_vertices,
            "alpha": self.alpha,
            "updates_since_rebuild": self.updates_since_rebuild,
            "unfolded_clusters": unfolded,
            "maintenance_runtime_s": maintenance_runtime_s,
            "unfold_runtime_s": unfold_runtime_s,
            "update_runtime_s": update_runtime_s,
        }

    def spectral_clustering_on_semicontracted(self, num_clusters: int, true_labels) -> Tuple:
        """
        Run spectral clustering query on the current semi-contracted graph state.
        """
        return self.contracted.spectral_clustering_on_contracted(num_clusters, true_labels)
