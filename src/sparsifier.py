from graph_tool import Graph
import graph_tool as gt
import numpy as np
import random
import time

from spectral_clustering import *
from datasets import *

from sklearn.metrics import adjusted_rand_score


class DynamicGraphSparsifier:
    def __init__(self, graph: Graph, sampling_constant: int = 100) -> None:
        """
        Initialize the Graph Sparsifier for Graph-tool graphs.

        Args:
        graph (Graph): The original graph to be sparsified, from Graph-tool.
        sampling_constant (int): A constant used in the calculation of the sampling probability.
        """
        self.original_graph = graph
        self.sparsified_graph = Graph(directed=False)
        self.sampling_constant = sampling_constant

        # Initialize property maps for degrees and sampling probabilities
        self.degree_array = self._degree_array(self.original_graph)
        self.used_sampling_parameters = None
        # (min(u,v), max(u,v)) -> sampling decision and unscaled edge mass.
        # Tracking unsampled edges too prevents a weighted update from drawing a
        # fresh Bernoulli sample on every increment.
        self._edge_sampling_state = {}

    @staticmethod
    def _degree_array(graph: Graph) -> np.ndarray:
        if "edge_weight" in graph.edge_properties:
            return graph.degree_property_map(
                "total", weight=graph.ep["edge_weight"]
            ).a.copy()
        return graph.degree_property_map("total").a.copy()

    @staticmethod
    def _norm_edge(u, v):
        u, v = int(u), int(v)
        return (u, v) if u <= v else (v, u)

    @classmethod
    def _original_edge_weights(cls, graph: Graph):
        """Aggregate the original (not Horvitz--Thompson) mass by edge pair."""
        if "edge_weight" in graph.edge_properties:
            rows = np.asarray(graph.get_edges([graph.ep["edge_weight"]]), dtype=float)
        else:
            endpoints = np.asarray(graph.get_edges(), dtype=float)
            if endpoints.size == 0:
                return {}
            rows = np.column_stack((endpoints[:, :2], np.ones(len(endpoints))))
        if rows.size == 0:
            return {}

        out = {}
        for u, v, weight in rows[:, :3]:
            key = cls._norm_edge(u, v)
            out[key] = out.get(key, 0.0) + float(weight)
        return out

    def _sampling_probability(self, u, v, log_n, original_weight=1.0):
        u, v = int(u), int(v)
        if u == v:
            return 1.0
        mass = float(original_weight)
        prob_u = min(
            1.0,
            self.sampling_constant * mass * log_n / max(self.degree_array[u], 1),
        )
        prob_v = min(
            1.0,
            self.sampling_constant * mass * log_n / max(self.degree_array[v], 1),
        )
        return float(prob_u + prob_v - prob_u * prob_v)

    def _store_sampled_edge(self, key, original_weight, probability):
        """Store one sampled edge without ever reweighting an estimator weight."""
        u, v = key
        probability = float(probability)
        if u == v:
            probability = 1.0
        if probability <= 0:
            return
        edge = self.sparsified_graph.add_edge(u, v)
        self.sparsified_graph.ep["original_edge_weight"][edge] = float(original_weight)
        self.sparsified_graph.ep["sampling_probability"][edge] = probability
        self.sparsified_graph.ep["edge_weight"][edge] = (
            float(original_weight) if u == v else float(original_weight) / probability
        )

    def _sample_edge(self, key, original_weight, log_n):
        probability = self._sampling_probability(
            key[0], key[1], log_n, original_weight=original_weight
        )
        sampled = key[0] == key[1] or float(np.random.rand()) < probability
        self._edge_sampling_state[key] = {
            "sampled": bool(sampled),
            "probability": float(1.0 if key[0] == key[1] else probability),
            "original_weight": float(original_weight),
        }
        if sampled:
            self._store_sampled_edge(key, original_weight, probability)

    def create_sparsifier(self):

        self.sparsified_graph = Graph(directed=False)
        self.sparsified_graph.set_fast_edge_removal(fast=True)
        edge_weight = self.sparsified_graph.new_edge_property("double")
        original_edge_weight = self.sparsified_graph.new_edge_property("double")
        sampling_probability = self.sparsified_graph.new_edge_property("double")
        self.sparsified_graph.edge_properties["edge_weight"] = edge_weight
        self.sparsified_graph.edge_properties["original_edge_weight"] = (
            original_edge_weight
        )
        self.sparsified_graph.edge_properties["sampling_probability"] = (
            sampling_probability
        )

        # Initialize degree array (weighted if edge weights are present).
        self.degree_array = self._degree_array(self.original_graph)

        n_vertices = int(self.original_graph.num_vertices())
        log_n = np.log(max(1, n_vertices))
        is_weighted_input = "edge_weight" in self.original_graph.edge_properties
        self._edge_sampling_state = {}

        # Add all vertices in one call (much faster than per-vertex loops).
        if n_vertices > 0:
            self.sparsified_graph.add_vertex(n_vertices)

        if is_weighted_input:
            # Contracted graphs are weighted and comparatively small. Retain
            # state for sampled and unsampled edges so repeated mass updates do
            # not redraw or reweight an existing Bernoulli decision.
            original_weights = self._original_edge_weights(self.original_graph)
            for key, original_weight_value in original_weights.items():
                self._sample_edge(key, original_weight_value, log_n)
        else:
            # Large base graphs are unweighted and receive only new-edge
            # insertions. Keep their construction vectorized and retain state
            # only for sampled edges.
            edges = np.asarray(self.original_graph.get_edges(), dtype=int)
            if edges.size > 0:
                endpoints = edges[:, :2]
                u, v = endpoints[:, 0], endpoints[:, 1]
                prob_u = np.minimum(
                    1.0,
                    self.sampling_constant
                    * log_n
                    / np.maximum(self.degree_array[u], 1),
                )
                prob_v = np.minimum(
                    1.0,
                    self.sampling_constant
                    * log_n
                    / np.maximum(self.degree_array[v], 1),
                )
                probabilities = prob_u + prob_v - prob_u * prob_v
                loops = u == v
                probabilities[loops] = 1.0
                add_mask = loops | (np.random.rand(len(endpoints)) < probabilities)
                if np.any(add_mask):
                    selected_endpoints = endpoints[add_mask]
                    selected_probabilities = probabilities[add_mask]
                    original_values = np.ones(len(selected_endpoints), dtype=float)
                    estimator_values = original_values / selected_probabilities
                    rows = np.column_stack(
                        (
                            selected_endpoints,
                            estimator_values,
                            original_values,
                            selected_probabilities,
                        )
                    )
                    self.sparsified_graph.add_edge_list(
                        rows,
                        eprops=[
                            edge_weight,
                            original_edge_weight,
                            sampling_probability,
                        ],
                    )
                    for (edge_u, edge_v), probability in zip(
                        selected_endpoints, selected_probabilities
                    ):
                        key = self._norm_edge(edge_u, edge_v)
                        self._edge_sampling_state[key] = {
                            "sampled": True,
                            "probability": float(probability),
                            "original_weight": 1.0,
                        }

        self.used_sampling_parameters = log_n / np.maximum(self.degree_array, 1)

    def print_verbose(self, message, start):
        print(f"{message}: {round(time.time() - start, 2)}s")

    def update_sparsifier(self, incoming_edges, verbose=True):
        def print_verbose(message, start):
            if verbose:
                print(f"{message}: {round(time.time() - start, 2)}s")

        start_time = time.time()
        n = int(self.original_graph.num_vertices())
        n_sparse_old = self.sparsified_graph.num_vertices()
        log_n = np.log(max(1, n))

        incoming_edges = np.asarray(incoming_edges)
        if incoming_edges.size == 0:
            edges_to_resample = np.zeros((0, 2), dtype=int)
            incoming_weights = np.zeros(0, dtype=float)
        else:
            if incoming_edges.ndim == 1:
                incoming_edges = incoming_edges.reshape(1, -1)
            edges_to_resample = incoming_edges[:, :2].astype(int, copy=False)
            incoming_weights = (
                incoming_edges[:, 2].astype(float, copy=False)
                if incoming_edges.shape[1] >= 3
                else np.ones(edges_to_resample.shape[0], dtype=float)
            )

        # Update degrees incrementally. Insertions only affect endpoints, and
        # graph-tool counts undirected self-loops twice.
        if len(self.degree_array) < n:
            self.degree_array = np.concatenate(
                (self.degree_array, np.zeros(n - len(self.degree_array), dtype=float))
            )
        elif len(self.degree_array) > n:
            self.degree_array = self.degree_array[:n].copy()

        if edges_to_resample.shape[0] > 0:
            valid = (
                (edges_to_resample[:, 0] >= 0)
                & (edges_to_resample[:, 1] >= 0)
                & (edges_to_resample[:, 0] < n)
                & (edges_to_resample[:, 1] < n)
            )
            if np.any(valid):
                valid_edges = edges_to_resample[valid]
                valid_weights = incoming_weights[valid]
                np.add.at(self.degree_array, valid_edges[:, 0], valid_weights)
                np.add.at(self.degree_array, valid_edges[:, 1], valid_weights)

        print_verbose("Adding nodes to sparse graph processing time", start_time)
        if self.used_sampling_parameters is None:
            self.used_sampling_parameters = np.array(
                log_n / np.maximum(self.degree_array, 1)
            )

        if n_sparse_old < n:
            num_new = int(n - n_sparse_old)
            self.sparsified_graph.add_vertex(num_new)
            new_idx = np.arange(int(n_sparse_old), int(n), dtype=int)
            new_params = log_n / np.maximum(self.degree_array[new_idx], 1)
            self.used_sampling_parameters = np.concatenate(
                (self.used_sampling_parameters, new_params.astype(float))
            )

        start_time = time.time()

        # Sampling parameter tracked for rebuild triggers (without the global sampling constant).
        current_sampling_parameter_array = log_n / np.maximum(self.degree_array, 1)

        used_sampling_parameter_array = self.used_sampling_parameters

        condition1 = (
            current_sampling_parameter_array > 10 * used_sampling_parameter_array
        )
        condition2 = (
            current_sampling_parameter_array < 0.1 * used_sampling_parameter_array
        )
        combined_condition = np.logical_or(condition1, condition2)
        vertices_to_resample = np.where(combined_condition)[0]

        rebuild_threshold = max(1, int(np.ceil(n / 100)))
        if len(vertices_to_resample) >= rebuild_threshold:
            # Because edge addition is much faster than edge removal in graph-tool,
            # if too many vertices have to be resampled we reconstruct the sparsifier.
            self.create_sparsifier()
            return

        if verbose:
            print("num of vertices to resample", len(vertices_to_resample))

        # Aggregate deltas so each logical weighted edge is updated once.
        incoming_deltas = {}
        for (u, v), weight in zip(edges_to_resample, incoming_weights):
            key = self._norm_edge(u, v)
            incoming_deltas[key] = incoming_deltas.get(key, 0.0) + float(weight)

        resampled_vertices = set(int(v) for v in vertices_to_resample)
        if resampled_vertices:
            current_weights = self._original_edge_weights(self.original_graph)
            affected_keys = {
                key
                for key in current_weights
                if key[0] in resampled_vertices or key[1] in resampled_vertices
            }
            for vertex in sorted(resampled_vertices):
                self.sparsified_graph.clear_vertex(self.sparsified_graph.vertex(vertex))
            for key in list(self._edge_sampling_state):
                if key[0] in resampled_vertices or key[1] in resampled_vertices:
                    del self._edge_sampling_state[key]
            for key in sorted(affected_keys):
                self._sample_edge(key, current_weights[key], log_n)
            self.used_sampling_parameters[vertices_to_resample] = log_n / np.maximum(
                self.degree_array[vertices_to_resample], 1
            )

        # Preserve the prior Bernoulli decision for ordinary weighted increments.
        for key, delta_weight in incoming_deltas.items():
            if key[0] in resampled_vertices or key[1] in resampled_vertices:
                continue
            state = self._edge_sampling_state.get(key)
            if state is None:
                self._sample_edge(key, delta_weight, log_n)
                continue

            state["original_weight"] = float(state["original_weight"] + delta_weight)
            if not state["sampled"]:
                continue
            edge = self.sparsified_graph.edge(key[0], key[1])
            if edge is None:
                raise RuntimeError(
                    "Sampled-edge state is inconsistent with the sparsifier."
                )
            probability = float(state["probability"])
            original_weight_value = float(state["original_weight"])
            self.sparsified_graph.ep["original_edge_weight"][
                edge
            ] = original_weight_value
            self.sparsified_graph.ep["sampling_probability"][edge] = probability
            self.sparsified_graph.ep["edge_weight"][edge] = (
                original_weight_value
                if key[0] == key[1]
                else original_weight_value / probability
            )

    def get_sparsified_graph(self) -> Graph:
        """
        Return the sparsified graph.

        Returns:
        Graph: The sparsified graph from Graph-tool.
        """
        return self.sparsified_graph
