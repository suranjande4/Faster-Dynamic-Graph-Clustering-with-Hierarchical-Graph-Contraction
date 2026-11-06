from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import graph_tool as gt
import numpy as np
import time
from sklearn.metrics import adjusted_rand_score

from contractedgraph import Contracted_Graph
from semicontracted import SemiContractedGraph
from sparsifier import DynamicGraphSparsifier
from spectral_clustering import clusters_to_labels, spectral_clustering


class _LevelState:
    def __init__(
        self,
        level: int,
        semi: SemiContractedGraph,
        input_base_node_map: List[set[int]],
        base_node_map: List[set[int]],
        base_map_dirty: bool = False,
    ) -> None:
        """Store one level object and its map back to base-graph vertices."""
        self.level = int(level)
        self.semi = semi
        # input vertex id at this level -> base-graph vertex ids
        self.input_base_node_map = input_base_node_map
        # contracted-node id (at this level) -> base-graph vertex ids
        self.base_node_map = base_node_map
        # True when base_node_map must be recomputed before use.
        self.base_map_dirty = bool(base_map_dirty)
        # Dense cache: base vertex id -> input vertex id at this level.
        self.base_vertex_to_input: Optional[np.ndarray] = None
        self.input_map_dirty = True


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
        semicontracted_beta: float = 1.0,
        batch_update_mode: bool = True,
        scheduler_update_unit: str = "edge",
        verbose: bool = False,
    ) -> None:
        """Initialize hierarchy parameters, counters, and initial level states."""
        self.base_graph = base_graph
        self.k = int(num_clusters)
        self.L = int(num_levels)
        self.sampling_constant = int(sampling_constant)
        self.degree_trigger = float(degree_trigger)
        self.semicontracted_beta = float(semicontracted_beta)
        self.verbose = bool(verbose)
        # Fast path: process each incoming batch level-by-level instead of
        # edge-by-edge propagation.
        self.batch_update_mode = bool(batch_update_mode)
        mode = str(scheduler_update_unit).strip().lower()
        if mode not in {"edge", "batch"}:
            raise ValueError("scheduler_update_unit must be one of {'edge', 'batch'}.")
        # Controls how delta_t is advanced for scheduler/rebuild logic.
        # - edge: advance by number of applied edges (original behavior)
        # - batch: advance by one for any non-empty effective batch
        self.scheduler_update_unit = mode

        self.delta_t = 0
        self.n_t_prime = 0
        # Cache phase thresholds for the current epoch (n_t_prime).
        self._phase_threshold_cache_n: Optional[int] = None
        self._phase_threshold_cache: dict[int, int] = {}

        self.levels: List[Optional[_LevelState]] = [None] * self.L
        self.active_levels = 0
        # Optional read-only diagnostics hook. The scheduler invokes it
        # synchronously after propagation and immediately before an existing
        # rebuild call; leaving it as None preserves normal behavior.
        self.pre_rebuild_callback: Optional[Callable[[dict], None]] = None

        # Step 1 on construction.
        self.initialize_structure()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _cluster_count_for_graph(self, k: int, g: gt.Graph) -> int:
        """Clamp requested cluster count to a valid range for graph g."""
        return max(1, min(int(k), int(g.num_vertices())))

    def _copy_graph(self, g: gt.Graph) -> gt.Graph:
        """Return a graph-tool copy of g."""
        return gt.Graph(g)

    @staticmethod
    def _maybe_enable_fast_edge_lookup(g: gt.Graph) -> None:
        """
        Enable fast edge lookup when supported by this graph-tool build.
        """
        fn = getattr(g, "set_fast_edge_lookup", None)
        if callable(fn):
            try:
                fn(fast=True)
            except TypeError:
                # graph-tool API signature can differ across versions.
                fn(True)

    @staticmethod
    def _copy_base_map(base_map: Sequence[set[int]]) -> List[set[int]]:
        """Return a deep copy of a vertex/node -> base vertices map."""
        return [set(int(v) for v in group) for group in base_map]

    @staticmethod
    def _singleton_base_map(num_vertices: int) -> List[set[int]]:
        """Map each vertex id to the same base-graph vertex id."""
        return [{int(i)} for i in range(int(num_vertices))]

    def _compose_map(
        self,
        local_map: Sequence[Sequence[int]],
        input_base_map: Sequence[set[int]],
    ) -> List[set[int]]:
        """Compose contracted-node -> input-node map with input-node -> base map."""
        composed: List[set[int]] = []
        for cluster in local_map:
            base_vertices = set()
            for input_node in cluster:
                idx = int(input_node)
                if 0 <= idx < len(input_base_map):
                    base_vertices.update(input_base_map[idx])
            composed.append(base_vertices)
        return composed

    def _ensure_base_vertex_to_input(self, state: _LevelState) -> np.ndarray:
        """Ensure dense base-vertex -> input-vertex cache for a level."""
        if state.base_vertex_to_input is not None and not state.input_map_dirty:
            return state.base_vertex_to_input

        max_base = int(self.base_graph.num_vertices()) - 1
        for group in state.input_base_node_map:
            if len(group) > 0:
                max_base = max(max_base, max(int(v) for v in group))

        size = max(0, max_base + 1)
        mapping = np.full(size, -1, dtype=int)
        for input_idx, group in enumerate(state.input_base_node_map):
            for base_v in group:
                iv = int(base_v)
                if iv >= len(mapping):
                    mapping = np.pad(
                        mapping, (0, iv + 1 - len(mapping)), constant_values=-1
                    )
                mapping[iv] = int(input_idx)

        state.base_vertex_to_input = mapping
        state.input_map_dirty = False
        return mapping

    def _ensure_base_vertices_in_level_input(
        self, state: _LevelState, base_vertices: Sequence[int]
    ) -> None:
        """
        Add unseen base vertices as singleton input vertices for this level.
        """
        if len(base_vertices) == 0:
            return

        mapping = self._ensure_base_vertex_to_input(state)
        unique_vertices = np.unique(np.asarray(base_vertices, dtype=int))
        if unique_vertices.size == 0:
            return

        max_vertex = int(np.max(unique_vertices))
        if max_vertex >= len(mapping):
            mapping = np.pad(
                mapping, (0, max_vertex + 1 - len(mapping)), constant_values=-1
            )

        for base_v in unique_vertices:
            iv = int(base_v)
            if iv < 0:
                continue
            if mapping[iv] >= 0:
                continue
            mapping[iv] = int(len(state.input_base_node_map))
            state.input_base_node_map.append({iv})

        state.base_vertex_to_input = mapping
        state.input_map_dirty = False

    def _project_base_edges_to_level(
        self, state: _LevelState, base_endpoints: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """Project base-graph edges into one level's current input graph."""
        if base_endpoints is None or len(base_endpoints) == 0:
            return []

        endpoints = np.asarray(base_endpoints)
        if endpoints.size == 0:
            return []
        if endpoints.ndim == 1:
            endpoints = endpoints.reshape(1, -1)
        weights = (
            endpoints[:, 2].astype(float)
            if endpoints.shape[1] >= 3
            else np.ones(endpoints.shape[0], dtype=float)
        )
        endpoints = endpoints[:, :2].astype(int, copy=False)

        self._ensure_base_vertices_in_level_input(state, endpoints.reshape(-1))
        mapping = self._ensure_base_vertex_to_input(state)

        in_range = (
            (endpoints[:, 0] >= 0)
            & (endpoints[:, 1] >= 0)
            & (endpoints[:, 0] < len(mapping))
            & (endpoints[:, 1] < len(mapping))
        )
        if not np.any(in_range):
            return []

        projected = mapping[endpoints[in_range]]
        projected_weights = weights[in_range]
        valid = (projected[:, 0] >= 0) & (projected[:, 1] >= 0)
        if not np.any(valid):
            return []

        projected = projected[valid].astype(int, copy=False)
        projected_weights = projected_weights[valid]
        return [
            (int(row[0]), int(row[1]), float(weight))
            for row, weight in zip(projected, projected_weights)
        ]

    def _build_level_from_graph(
        self,
        level_idx_1based: int,
        graph_for_level: gt.Graph,
        input_base_map: Sequence[set[int]],
        use_graph_in_place: bool = False,
    ) -> _LevelState:
        """Construct one hierarchy level from a graph and its input-to-base map."""
        t_level = time.perf_counter()
        g = graph_for_level if use_graph_in_place else self._copy_graph(graph_for_level)
        self._maybe_enable_fast_edge_lookup(g)
        self._log(
            f"    hierarchy: level {level_idx_1based} input graph "
            f"n={int(g.num_vertices())}, m={int(g.num_edges())}"
        )

        t0 = time.perf_counter()
        sparsifier = DynamicGraphSparsifier(g, sampling_constant=self.sampling_constant)
        sparsifier.create_sparsifier()
        self._log(
            f"    hierarchy: level {level_idx_1based} sparsifier ready in "
            f"{time.perf_counter() - t0:.2f}s"
        )

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
        t0 = time.perf_counter()
        semi.full_reconstruct(num_clusters=level_k)
        self._log(
            f"    hierarchy: level {level_idx_1based} reconstruction ready in "
            f"{time.perf_counter() - t0:.2f}s"
        )

        input_map = self._copy_base_map(input_base_map)
        local_map = semi.contracted.contracted_node_to_full_node_map
        base_map = self._compose_map(local_map, input_map)

        state = _LevelState(
            level=level_idx_1based,
            semi=semi,
            input_base_node_map=input_map,
            base_node_map=base_map,
            base_map_dirty=False,
        )
        self._log(
            f"    hierarchy: level {level_idx_1based} ready in "
            f"{time.perf_counter() - t_level:.2f}s"
        )
        return state

    def _clone_level_state(
        self, source: _LevelState, level_idx_1based: int
    ) -> _LevelState:
        """Create an independent copy of one maintained contracted level."""
        t_clone = time.perf_counter()
        self._log(f"    hierarchy: cloning level {level_idx_1based}...")
        g = self._copy_graph(source.semi.original_graph)
        self._maybe_enable_fast_edge_lookup(g)

        sparsifier = DynamicGraphSparsifier(
            g, sampling_constant=source.semi.sparsifier.sampling_constant
        )
        sparsifier.sparsified_graph = self._copy_graph(source.semi.sparsified_graph)
        sparsifier.degree_array = np.array(
            source.semi.sparsifier.degree_array, copy=True
        )
        if source.semi.sparsifier.used_sampling_parameters is None:
            sparsifier.used_sampling_parameters = None
        else:
            sparsifier.used_sampling_parameters = np.array(
                source.semi.sparsifier.used_sampling_parameters, copy=True
            )
        sparsifier._edge_sampling_state = {
            tuple(key): dict(value)
            for key, value in source.semi.sparsifier._edge_sampling_state.items()
        }

        contracted = Contracted_Graph(
            g,
            sparsifier.sparsified_graph,
            degree_trigger=source.semi.degree_trigger,
        )
        src_contracted = source.semi.contracted
        contracted.contracted_graph = self._copy_graph(src_contracted.contracted_graph)
        contracted.cluster_map = dict(getattr(src_contracted, "cluster_map", {}))
        contracted.cluster_vertices = dict(
            getattr(src_contracted, "cluster_vertices", {})
        )
        contracted.contracted_node_to_full_node_map = [
            set(int(v) for v in cluster)
            for cluster in src_contracted.contracted_node_to_full_node_map
        ]
        contracted.contract_idx_to_vertex_label = list(
            getattr(src_contracted, "contract_idx_to_vertex_label", [])
        )
        contracted.og_num_contract_nodes = int(
            getattr(
                src_contracted,
                "og_num_contract_nodes",
                len(contracted.contracted_node_to_full_node_map),
            )
        )
        contracted.degree_array = np.array(src_contracted.degree_array, copy=True)
        contracted.degree_array_at_construction = np.array(
            src_contracted.degree_array_at_construction, copy=True
        )
        if src_contracted.og_vertex_to_contract_idx is None:
            contracted.og_vertex_to_contract_idx = None
        else:
            contracted.og_vertex_to_contract_idx = np.array(
                src_contracted.og_vertex_to_contract_idx, copy=True
            )

        semi = SemiContractedGraph(
            original_graph=g,
            sparsifier=sparsifier,
            degree_trigger=source.semi.degree_trigger,
            contracted=contracted,
            allow_self_loops=source.semi.allow_self_loops,
            count_existing_edge_updates=source.semi.count_existing_edge_updates,
        )
        semi.sparsified_graph = sparsifier.sparsified_graph
        semi.alpha = int(source.semi.alpha)
        semi.updates_since_rebuild = int(source.semi.updates_since_rebuild)
        semi.n_at_last_rebuild = int(
            getattr(
                source.semi,
                "n_at_last_rebuild",
                source.semi.original_graph.num_vertices(),
            )
        )
        semi.is_initialized = bool(source.semi.is_initialized)
        semi.last_num_clusters = source.semi.last_num_clusters

        state = _LevelState(
            level=level_idx_1based,
            semi=semi,
            input_base_node_map=self._copy_base_map(source.input_base_node_map),
            base_node_map=self._copy_base_map(source.base_node_map),
            base_map_dirty=bool(source.base_map_dirty),
        )
        self._log(
            f"    hierarchy: cloned level {level_idx_1based} in "
            f"{time.perf_counter() - t_clone:.2f}s"
        )
        return state

    def _ensure_level_base_map(self, level_1based: int) -> None:
        """
        Ensure base-node map for one active level is up to date.
        """
        lev = int(level_1based)
        if not (1 <= lev <= self.active_levels):
            return

        state = self.levels[lev - 1]
        if state is None or not bool(state.base_map_dirty):
            return

        local_map = state.semi.contracted.contracted_node_to_full_node_map
        state.base_node_map = self._compose_map(local_map, state.input_base_node_map)
        state.base_map_dirty = False

    def _current_level_sizes(self) -> List[int]:
        """Return contracted-graph vertex counts per level."""
        return [
            (
                0
                if state is None
                else int(state.semi.contracted.contracted_graph.num_vertices())
            )
            for state in self.levels
        ]

    def _current_level_alpha(self) -> List[int]:
        """Return alpha values per level (0 for inactive levels)."""
        return [0 if state is None else int(state.semi.alpha) for state in self.levels]

    def _level_alpha_limit(self, state: _LevelState) -> int:
        """Return the Semi-style maintenance limit for one hierarchy level."""
        n = int(
            getattr(
                state.semi,
                "n_at_last_rebuild",
                state.semi.original_graph.num_vertices(),
            )
        )
        return max(1, int(np.ceil(max(1, n) ** self.semicontracted_beta)))

    def _ensure_level_contracted_current(self, level_1based: int) -> float:
        """
        Ensure a level's contracted graph represents its current sparsifier.

        Levels may skip contracted-graph maintenance after their Semi threshold is
        exceeded. Before such a level is used as the input to rebuild lower
        hierarchy levels, reconstruct it from its current sparsifier so newly
        introduced input vertices are not dropped from the base-node map.

        Returns the reconstruction time, if a reconstruction was needed.
        """
        lev = int(level_1based)
        if not (1 <= lev <= self.active_levels):
            return 0.0

        state = self.levels[lev - 1]
        if state is None:
            return 0.0

        if int(state.semi.alpha) <= self._level_alpha_limit(state):
            self._ensure_level_base_map(lev)
            return 0.0

        t0 = time.perf_counter()
        state.semi.full_reconstruct(
            num_clusters=self._cluster_count_for_graph(
                self.k, state.semi.sparsified_graph
            )
        )
        state.base_node_map = self._compose_map(
            state.semi.contracted.contracted_node_to_full_node_map,
            state.input_base_node_map,
        )
        state.base_map_dirty = False
        state.input_map_dirty = True
        state.base_vertex_to_input = None
        return time.perf_counter() - t0

    # ----------------------------
    # Step 1: InitializeStructure
    # ----------------------------
    def initialize_structure(self) -> None:
        """Rebuild all hierarchy levels from scratch and reset epoch counters."""
        t_init = time.perf_counter()
        self._log(
            f"    hierarchy: initializing {self.L} levels from base graph "
            f"n={int(self.base_graph.num_vertices())}, m={int(self.base_graph.num_edges())}"
        )
        self.delta_t = 0
        self.n_t_prime = max(1, int(self.base_graph.num_vertices()))

        self.levels = [None] * self.L

        base_input_map = self._singleton_base_map(int(self.base_graph.num_vertices()))
        self.levels[0] = self._build_level_from_graph(
            level_idx_1based=1,
            graph_for_level=self.base_graph,
            input_base_map=base_input_map,
            use_graph_in_place=True,
        )

        # Section 4 initializes C2, ..., CL as identical copies of C1.
        # We share the same state object for identical suffix levels and update
        # each distinct state only once; this avoids O(L) deep cloning during
        # initialization while preserving the same observable hierarchy.
        for lev in range(2, self.L + 1):
            first = self.levels[0]
            if first is None:
                break
            self.levels[lev - 1] = first

        self.active_levels = sum(1 for s in self.levels if s is not None)
        self._log(
            f"    hierarchy: initialized {self.active_levels} levels in "
            f"{time.perf_counter() - t_init:.2f}s"
        )

    # ----------------------------
    # Step 2: Incremental update
    # ----------------------------
    def _update_all_levels_from_base_edges(
        self, base_edges: Iterable[Tuple[int, int]]
    ) -> List[dict]:
        """
        Section-4 update path: apply the same base-edge batch to every level,
        projected into that level's current input vertex space.
        """
        edges = list(base_edges)
        if len(edges) == 0:
            return []

        weighted_edges = np.asarray(edges)
        if weighted_edges.ndim == 1:
            weighted_edges = weighted_edges.reshape(1, -1)
        infos: List[dict] = []
        info_by_state_id: dict[int, dict] = {}

        for lev in range(1, self.active_levels + 1):
            state = self.levels[lev - 1]
            if state is None:
                break

            state_id = id(state)
            if state_id in info_by_state_id:
                info = info_by_state_id[state_id]
            else:
                projected = self._project_base_edges_to_level(state, weighted_edges)
                next_alpha = int(state.semi.alpha) + len(projected)
                maintain_contracted = next_alpha <= self._level_alpha_limit(state)
                info = state.semi.update(
                    projected,
                    maintain_contracted=maintain_contracted,
                )
                info_by_state_id[state_id] = info
            infos.append(info)
            state.base_map_dirty = True

        return infos

    # ----------------------------
    # Step 3: Scheduler + rebuilds
    # ----------------------------
    def _phase_threshold(self, k_1based: int) -> int:
        """Compute threshold ceil(n_t_prime^(k/L)) for phase k."""
        k = int(k_1based)
        n_now = int(self.n_t_prime)
        if self._phase_threshold_cache_n != n_now:
            self._phase_threshold_cache_n = n_now
            self._phase_threshold_cache = {}
        cached = self._phase_threshold_cache.get(k)
        if cached is not None:
            return cached
        value = max(1, int(np.ceil(n_now ** (float(k) / float(self.L)))))
        self._phase_threshold_cache[k] = value
        return value

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

    def _argmax_phase_triggered_in_interval(self, lo: int, hi: int) -> Optional[int]:
        """
        Return largest phase k with at least one trigger in [lo, hi].

        This avoids scanning every intermediate update index in large batches.
        """
        lo = int(lo)
        hi = int(hi)
        if hi < lo or hi <= 0:
            return None

        best: Optional[int] = None
        for k in range(1, self.L):
            thr = self._phase_threshold(k)
            if thr <= 0 or hi < thr:
                continue
            start = max(lo, thr)
            first_multiple = ((start + thr - 1) // thr) * thr
            if first_multiple <= hi:
                best = k
        return best

    def _rebuild_suffix(self, start_level_1based: int) -> List[int]:
        """
        Rebuild one hierarchy level from the level above, then copy it downward.

        This follows Section 4: C_{L-j+1} <- ContractGraph(C_{L-j}) and
        C_i <- C_{L-j+1} for all lower levels.
        """
        if start_level_1based > self.active_levels:
            return []

        start = int(start_level_1based)
        if start <= 1:
            self.initialize_structure()
            return list(range(1, int(self.active_levels) + 1))

        self._ensure_level_contracted_current(start - 1)
        parent = self.levels[start - 2]
        if parent is None:
            return []

        rebuilt_state = self._build_level_from_graph(
            level_idx_1based=start,
            graph_for_level=parent.semi.contracted.contracted_graph,
            input_base_map=parent.base_node_map,
            use_graph_in_place=False,
        )
        self.levels[start - 1] = rebuilt_state

        rebuilt: List[int] = [start]
        for lev in range(start + 1, self.active_levels + 1):
            self.levels[lev - 1] = rebuilt_state
            rebuilt.append(lev)

        return rebuilt

    def _scheduler_increment(self, applied_now: int) -> int:
        """
        Return how much scheduler counter delta_t should advance for this update.
        """
        applied_now = int(applied_now)
        if applied_now <= 0:
            return 0
        if self.scheduler_update_unit == "batch":
            return 1
        return applied_now

    def _notify_pre_rebuild(self, reason: str, start_level: int) -> None:
        callback = self.pre_rebuild_callback
        if callback is None:
            return
        callback(
            {
                "reason": str(reason),
                "start_level": int(start_level),
                "active_levels": int(self.active_levels),
                "delta_t": int(self.delta_t),
                "n_t_prime": int(self.n_t_prime),
                "level_sizes": self._current_level_sizes(),
            }
        )

    @staticmethod
    def _accumulate_level_stats(
        infos: Sequence[dict],
        level_applied: List[int],
        level_unfolded: List[int],
        level_runtime: List[float],
    ) -> None:
        """Accumulate per-level counters from semi.update(...) info dicts."""
        for idx, info in enumerate(infos):
            level_applied[idx] += int(info.get("num_applied", 0))
            level_unfolded[idx] += int(info.get("unfolded_clusters", 0))
            level_runtime[idx] += float(info.get("update_runtime_s", 0.0))

    def _build_update_result(
        self,
        *,
        num_applied: int,
        attempted_edges: int,
        unique_edges: int,
        phase_k: Optional[int],
        rebuilt_levels: Sequence[int],
        restarted: bool,
        level_applied: Optional[Sequence[int]] = None,
        level_unfolded: Optional[Sequence[int]] = None,
        level_runtime: Optional[Sequence[float]] = None,
    ) -> dict:
        """Build standardized update response payload."""
        if level_applied is None:
            level_applied = [0] * self.L
        if level_unfolded is None:
            level_unfolded = [0] * self.L
        if level_runtime is None:
            level_runtime = [0.0] * self.L

        return {
            "num_applied": int(num_applied),
            "attempted_edges": int(attempted_edges),
            "unique_edges": int(unique_edges),
            "updates_since_t_prime": int(self.delta_t),
            "n_at_t_prime": int(self.n_t_prime),
            "phase_k": phase_k,
            "rebuilt_levels": sorted(set(int(x) for x in rebuilt_levels)),
            "restarted": bool(restarted),
            "active_levels": int(self.active_levels),
            "level_sizes": self._current_level_sizes(),
            "level_applied": [int(x) for x in level_applied],
            "level_unfolded": [int(x) for x in level_unfolded],
            "level_alpha": self._current_level_alpha(),
            "level_update_runtime_s": [float(x) for x in level_runtime],
        }

    def update_hierarchy(self, new_edges: Iterable[Tuple[int, int]]) -> dict:
        """Apply a batch of insertions, run scheduler, and return update statistics."""
        raw_edges = list(new_edges)
        if len(raw_edges) == 0:
            return self._build_update_result(
                num_applied=0,
                attempted_edges=0,
                unique_edges=0,
                phase_k=None,
                rebuilt_levels=[],
                restarted=False,
            )

        attempted_total = 0
        unique_total = 0
        applied_total = 0
        phase_k_last = None
        rebuilt_levels_all: List[int] = []
        restarted = False

        level_applied = [0] * self.L
        level_unfolded = [0] * self.L
        level_runtime = [0.0] * self.L

        if self.batch_update_mode and len(raw_edges) > 1:
            infos = self._update_all_levels_from_base_edges(raw_edges)
            if len(infos) > 0:
                self._accumulate_level_stats(
                    infos, level_applied, level_unfolded, level_runtime
                )

                base_info = infos[0]
                attempted_total += int(base_info.get("attempted_edges", 0))
                unique_total += int(base_info.get("unique_edges", 0))
                applied_now = int(base_info.get("num_applied", 0))
                applied_total += applied_now

                delta_inc = self._scheduler_increment(applied_now)
                if delta_inc > 0:
                    prev_delta = int(self.delta_t)
                    new_delta = int(prev_delta + delta_inc)
                    self.delta_t = new_delta

                    if self.delta_t >= self.n_t_prime:
                        self._notify_pre_rebuild("restart", 1)
                        self.initialize_structure()
                        rebuilt_levels_all.extend(range(1, int(self.active_levels) + 1))
                        restarted = True
                    elif new_delta >= self._phase_threshold(1):
                        phase_k = self._argmax_phase_triggered_in_interval(
                            prev_delta + 1, new_delta
                        )
                        if phase_k is not None:
                            start = self.L - phase_k + 1
                            self._notify_pre_rebuild("phase", start)
                            rebuilt = self._rebuild_suffix(start)
                            if len(rebuilt) > 0:
                                rebuilt_levels_all.extend(int(x) for x in rebuilt)
                            phase_k_last = int(phase_k)
        else:
            for edge in raw_edges:
                infos = self._update_all_levels_from_base_edges([edge])
                if len(infos) == 0:
                    continue

                self._accumulate_level_stats(
                    infos, level_applied, level_unfolded, level_runtime
                )

                base_info = infos[0]
                attempted_total += int(base_info.get("attempted_edges", 0))
                unique_total += int(base_info.get("unique_edges", 0))
                applied_now = int(base_info.get("num_applied", 0))
                applied_total += applied_now

                delta_inc = self._scheduler_increment(applied_now)
                if delta_inc <= 0:
                    continue

                self.delta_t += delta_inc

                if self.delta_t >= self.n_t_prime:
                    self._notify_pre_rebuild("restart", 1)
                    self.initialize_structure()
                    rebuilt_levels_all.extend(range(1, int(self.active_levels) + 1))
                    restarted = True
                    continue

                if self.delta_t >= self._phase_threshold(1):
                    phase_k = self._argmax_divisible_phase(self.delta_t)
                    if phase_k is not None:
                        start = self.L - phase_k + 1
                        self._notify_pre_rebuild("phase", start)
                        rebuilt = self._rebuild_suffix(start)
                        if len(rebuilt) > 0:
                            rebuilt_levels_all.extend(int(x) for x in rebuilt)
                        phase_k_last = int(phase_k)

        return self._build_update_result(
            num_applied=applied_total,
            attempted_edges=attempted_total,
            unique_edges=unique_total,
            phase_k=phase_k_last,
            rebuilt_levels=rebuilt_levels_all,
            restarted=restarted,
            level_applied=level_applied,
            level_unfolded=level_unfolded,
            level_runtime=level_runtime,
        )

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

        self._ensure_level_base_map(query_level)
        state = self.levels[query_level - 1]
        if state is None:
            raise RuntimeError("Requested hierarchy level is not active.")

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
