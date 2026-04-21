from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import graph_tool as gt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from semicontracted import SemiContractedGraph
from sparsifier import DynamicGraphSparsifier
from spectral_clustering import clusters_to_labels, spectral_clustering


class _LevelState:
    def __init__(
        self,
        level: int,
        semi: SemiContractedGraph,
        base_node_map: List[set[int]],
    ) -> None:
        """Store one level object and its map back to base-graph vertices."""
        self.level = int(level)
        self.semi = semi
        # contracted-node id (at this level) -> base-graph vertex ids
        self.base_node_map = base_node_map


class HierarchicalContractedStructure:
    """Hierarchical contracted structure for dynamic clustering with insertions.
    Step 1: initialize full hierarchy at  start.
    Step 2: apply one insertion update across all levels.
    Step 3: run phase-based rebuild scheduler and restart logic.
    Step 4: query the deepest (or requested) level and map back to base vertices.
    """

    def __init__(
        self,
        base_graph: gt.Graph,
        num_clusters: int,
        num_levels: int = 5,
        sampling_constant: int = 80,
        degree_trigger: float = 0.5,
    ) -> None:
        """Initialize hierarchy parameters, counters, and initial level states."""
        self.base_graph = base_graph
        self.k = int(num_clusters)
        self.L = int(num_levels)
        self.sampling_constant = int(sampling_constant)
        self.degree_trigger = float(degree_trigger)

        self.delta_t = 0
        self.n_t_prime = 0

        self.levels: List[Optional[_LevelState]] = [None] * self.L
        self.active_levels = 0

        # Step 1 on construction.
        self.initialize_structure()

    # ----------------------------
    # Helpers
    # ----------------------------
    def _cluster_count_for_graph(self, k: int, g: gt.Graph) -> int:
        """Clamp requested cluster count to a valid range for graph g."""
        return max(1, min(int(k), int(g.num_vertices())))

    def _copy_graph(self, g: gt.Graph) -> gt.Graph:
        """Return a graph-tool copy of g."""
        return gt.Graph(g)

    def _map_level1_to_base(self, semi: SemiContractedGraph) -> List[set[int]]:
        """Map level-1 contracted nodes directly to base-graph vertex ids."""
        return [
            set(int(v) for v in cluster)
            for cluster in semi.contracted.contracted_node_to_full_node_map
        ]

    def _compose_map(
        self,
        local_map: Sequence[Sequence[int]],
        parent_base_map: Sequence[set[int]],
    ) -> List[set[int]]:
        """Compose child-to-parent map with parent-to-base map."""
        composed: List[set[int]] = []
        for cluster in local_map:
            base_vertices = set()
            for parent_node in cluster:
                idx = int(parent_node)
                if 0 <= idx < len(parent_base_map):
                    base_vertices.update(parent_base_map[idx])
            composed.append(base_vertices)
        return composed

    def _vertex_to_contract_idx(self, semi: SemiContractedGraph) -> np.ndarray:
        """Build vertex->contracted-node lookup for the given semi-contracted state."""
        n = int(semi.original_graph.num_vertices())
        out = np.full(n, -1, dtype=int)
        for cid, cluster in enumerate(semi.contracted.contracted_node_to_full_node_map):
            for v in cluster:
                iv = int(v)
                if 0 <= iv < n:
                    out[iv] = cid
        return out

    def _build_level_from_graph(
        self,
        level_idx_1based: int,
        graph_for_level: gt.Graph,
        parent_base_map: Optional[Sequence[set[int]]],
        use_graph_in_place: bool = False,
    ) -> _LevelState:
        """Construct one hierarchy level from a graph and optional parent base mapping."""
        g = graph_for_level if use_graph_in_place else self._copy_graph(graph_for_level)

        sparsifier = DynamicGraphSparsifier(g, sampling_constant=self.sampling_constant)
        sparsifier.create_sparsifier()

        # Level-1 tracks simple-graph insertions; higher levels also track
        # repeated cluster-pair mass and same-cluster updates via self-loops.
        is_higher_level = level_idx_1based > 1
        semi = SemiContractedGraph(
            original_graph=g,
            sparsifier=sparsifier,
            degree_trigger=self.degree_trigger,
            allow_self_loops=is_higher_level,
            count_existing_edge_updates=is_higher_level,
        )
        level_k = self._cluster_count_for_graph(self.k, g)
        semi.full_reconstruct(num_clusters=level_k)

        if parent_base_map is None:
            base_map = self._map_level1_to_base(semi)
        else:
            local_map = semi.contracted.contracted_node_to_full_node_map
            base_map = self._compose_map(local_map, parent_base_map)

        return _LevelState(level=level_idx_1based, semi=semi, base_node_map=base_map)

    def _current_level_sizes(self) -> List[int]:
        """Return contracted-graph vertex counts per level."""
        out: List[int] = []
        for state in self.levels:
            if state is None:
                out.append(0)
            else:
                out.append(int(state.semi.contracted.contracted_graph.num_vertices()))
        return out

    def _current_level_alpha(self) -> List[int]:
        """Return alpha values per level (0 for inactive levels)."""
        out: List[int] = []
        for state in self.levels:
            out.append(0 if state is None else int(state.semi.alpha))
        return out

    # ----------------------------
    # Step 1: InitializeStructure
    # ----------------------------
    def initialize_structure(self) -> None:
        """Rebuild all hierarchy levels from scratch and reset epoch counters."""
        self.delta_t = 0
        self.n_t_prime = max(1, int(self.base_graph.num_vertices()))

        self.levels = [None] * self.L

        self.levels[0] = self._build_level_from_graph(
            level_idx_1based=1,
            graph_for_level=self.base_graph,
            parent_base_map=None,
            use_graph_in_place=True,
        )

        for lev in range(2, self.L + 1):
            parent = self.levels[lev - 2]
            if parent is None:
                break
            self.levels[lev - 1] = self._build_level_from_graph(
                level_idx_1based=lev,
                graph_for_level=parent.semi.contracted.contracted_graph,
                parent_base_map=parent.base_node_map,
                use_graph_in_place=False,
            )

        self.active_levels = sum(1 for s in self.levels if s is not None)

    # ----------------------------
    # Step 2: Incremental update
    # ----------------------------
    def _project_edges_to_next_level(
        self,
        parent_semi: SemiContractedGraph,
        parent_applied_edges: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Project applied parent edges to unique contracted edges at the next level."""
        if parent_applied_edges is None or len(parent_applied_edges) == 0:
            return []

        v_to_c = self._vertex_to_contract_idx(parent_semi)
        projected: List[Tuple[int, int]] = []
        seen = set()

        for (u, v) in parent_applied_edges:
            iu = int(u)
            iv = int(v)
            if iu < 0 or iv < 0 or iu >= len(v_to_c) or iv >= len(v_to_c):
                continue
            cu = int(v_to_c[iu])
            cv = int(v_to_c[iv])
            if cu < 0 or cv < 0:
                continue
            a, b = (cu, cv) if cu < cv else (cv, cu)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            projected.append((a, b))

        return projected

    def _update_all_levels_for_single_edge(self, edge: Tuple[int, int]) -> List[dict]:
        """Propagate one insertion through active levels until no projected edges remain."""
        infos: List[dict] = []
        projected: List[Tuple[int, int]] = [edge]

        for lev in range(1, self.active_levels + 1):
            if len(projected) == 0:
                break

            state = self.levels[lev - 1]
            if state is None:
                break

            info = state.semi.update(projected)
            infos.append(info)

            if lev == 1:
                state.base_node_map = self._map_level1_to_base(state.semi)
            else:
                parent = self.levels[lev - 2]
                if parent is not None:
                    local_map = state.semi.contracted.contracted_node_to_full_node_map
                    state.base_node_map = self._compose_map(local_map, parent.base_node_map)

            if lev < self.active_levels:
                projected = self._project_edges_to_next_level(
                    parent_semi=state.semi,
                    parent_applied_edges=info.get("applied_edges", []),
                )

        return infos

    # ----------------------------
    # Step 3: Scheduler + rebuilds
    # ----------------------------
    def _phase_threshold(self, k_1based: int) -> int:
        """Compute threshold ceil(n_t_prime^(k/L)) for phase k."""
        return max(1, int(np.ceil(self.n_t_prime ** (float(k_1based) / float(self.L)))))

    def _argmax_divisible_phase(self, u: int) -> Optional[int]:
        """Return largest phase k whose threshold divides update counter u."""
        if u <= 0:
            return None

        best: Optional[int] = None
        for k in range(1, self.L):
            thr = self._phase_threshold(k)
            if thr > 0 and u % thr == 0:
                best = k
        return best

    def _rebuild_suffix(self, start_level_1based: int) -> List[int]:
        """Rebuild all levels from start_level_1based up to active_levels."""
        if start_level_1based > self.active_levels:
            return []

        rebuilt: List[int] = []
        for lev in range(start_level_1based, self.active_levels + 1):
            if lev == 1:
                self.levels[0] = self._build_level_from_graph(
                    level_idx_1based=1,
                    graph_for_level=self.base_graph,
                    parent_base_map=None,
                    use_graph_in_place=True,
                )
            else:
                parent = self.levels[lev - 2]
                if parent is None:
                    break
                self.levels[lev - 1] = self._build_level_from_graph(
                    level_idx_1based=lev,
                    graph_for_level=parent.semi.contracted.contracted_graph,
                    parent_base_map=parent.base_node_map,
                    use_graph_in_place=False,
                )
            rebuilt.append(lev)
        return rebuilt

    def update_hierarchy(self, new_edges: Iterable[Tuple[int, int]]) -> dict:
        """Apply a batch of insertions, run scheduler, and return update statistics."""
        raw_edges = list(new_edges)
        if len(raw_edges) == 0:
            return {
                "num_applied": 0,
                "attempted_edges": 0,
                "unique_edges": 0,
                "updates_since_t_prime": int(self.delta_t),
                "n_at_t_prime": int(self.n_t_prime),
                "phase_k": None,
                "rebuilt_levels": [],
                "restarted": False,
                "active_levels": int(self.active_levels),
                "level_sizes": self._current_level_sizes(),
                "level_applied": [0] * self.L,
                "level_unfolded": [0] * self.L,
                "level_alpha": self._current_level_alpha(),
                "level_update_runtime_s": [0.0] * self.L,
            }

        attempted_total = 0
        unique_total = 0
        applied_total = 0
        phase_k_last = None
        rebuilt_levels_all: List[int] = []
        restarted = False

        level_applied = [0] * self.L
        level_unfolded = [0] * self.L
        level_runtime = [0.0] * self.L

        for edge in raw_edges:
            infos = self._update_all_levels_for_single_edge(edge)
            if len(infos) == 0:
                continue

            for idx, info in enumerate(infos):
                level_applied[idx] += int(info.get("num_applied", 0))
                level_unfolded[idx] += int(info.get("unfolded_clusters", 0))
                level_runtime[idx] += float(info.get("update_runtime_s", 0.0))

            base_info = infos[0]
            attempted_total += int(base_info.get("attempted_edges", 0))
            unique_total += int(base_info.get("unique_edges", 0))
            applied_now = int(base_info.get("num_applied", 0))
            applied_total += applied_now

            if applied_now <= 0:
                continue

            self.delta_t += applied_now

            if self.delta_t >= self._phase_threshold(1):
                phase_k = self._argmax_divisible_phase(self.delta_t)
                if phase_k is not None:
                    start = self.L - phase_k + 1
                    rebuilt = self._rebuild_suffix(start)
                    if len(rebuilt) > 0:
                        rebuilt_levels_all.extend(int(x) for x in rebuilt)
                    phase_k_last = int(phase_k)

            if self.delta_t == self.n_t_prime:
                self.initialize_structure()
                restarted = True

        return {
            "num_applied": int(applied_total),
            "attempted_edges": int(attempted_total),
            "unique_edges": int(unique_total),
            "updates_since_t_prime": int(self.delta_t),
            "n_at_t_prime": int(self.n_t_prime),
            "phase_k": phase_k_last,
            "rebuilt_levels": sorted(set(rebuilt_levels_all)),
            "restarted": bool(restarted),
            "active_levels": int(self.active_levels),
            "level_sizes": self._current_level_sizes(),
            "level_applied": [int(x) for x in level_applied],
            "level_unfolded": [int(x) for x in level_unfolded],
            "level_alpha": self._current_level_alpha(),
            "level_update_runtime_s": [float(x) for x in level_runtime],
        }

    # ----------------------------
    # Step 4: Query
    # ----------------------------
    def query(
        self,
        num_clusters: Optional[int] = None,
        true_labels: Optional[Sequence[int]] = None,
        level: Optional[int] = None,
        map_to_base: bool = True,
    ) -> dict:
        """Run spectral clustering on a chosen level and optionally map results to base graph."""
        if self.active_levels <= 0:
            raise RuntimeError("Hierarchy has no active levels.")

        query_level = int(level) if level is not None else int(self.active_levels)
        if not (1 <= query_level <= self.active_levels):
            raise ValueError("level must be between 1 and active_levels.")

        state = self.levels[query_level - 1]
        if state is None:
            raise RuntimeError("Requested hierarchy level is not active.")

        cg = state.semi.contracted.contracted_graph
        if int(cg.num_vertices()) == 0:
            labels = np.array([], dtype=int)
            return {
                "level": int(query_level),
                "active_levels": int(self.active_levels),
                "level_graph_vertices": 0,
                "level_graph_edges": 0,
                "clusters_level": [],
                "base_clusters": [],
                "labels": labels,
                "ari": None,
                "level_sizes": self._current_level_sizes(),
            }

        query_k = self.k if num_clusters is None else int(num_clusters)
        level_k = self._cluster_count_for_graph(query_k, cg)
        clusters_level = spectral_clustering(cg, level_k)

        if not bool(map_to_base):
            return {
                "level": int(query_level),
                "active_levels": int(self.active_levels),
                "level_graph_vertices": int(cg.num_vertices()),
                "level_graph_edges": int(cg.num_edges()),
                "clusters_level": clusters_level,
                "base_clusters": None,
                "labels": None,
                "ari": None,
                "level_sizes": self._current_level_sizes(),
            }

        base_clusters: List[List[int]] = []
        for cluster in clusters_level:
            base_nodes = set()
            for node in cluster:
                idx = int(node)
                if 0 <= idx < len(state.base_node_map):
                    base_nodes.update(state.base_node_map[idx])
            base_clusters.append(sorted(base_nodes))

        base_n = int(self.base_graph.num_vertices())
        labels = clusters_to_labels(base_clusters, num_data_points=base_n)

        ari = None
        if true_labels is not None:
            y_true = np.array(true_labels)
            m = min(len(y_true), len(labels))
            if m > 0:
                ari = float(adjusted_rand_score(y_true[:m], labels[:m]))

        return {
            "level": int(query_level),
            "active_levels": int(self.active_levels),
            "level_graph_vertices": int(cg.num_vertices()),
            "level_graph_edges": int(cg.num_edges()),
            "clusters_level": clusters_level,
            "base_clusters": base_clusters,
            "labels": labels,
            "ari": ari,
            "level_sizes": self._current_level_sizes(),
        }
