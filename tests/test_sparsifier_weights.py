from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import graph_tool as gt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hierarchy_contracted import HierarchicalContractedStructure
from sparsifier import DynamicGraphSparsifier
from spectral_clustering import spectral_clustering


def weighted_graph(num_vertices: int, edges: list[tuple[int, int, float]]) -> gt.Graph:
    graph = gt.Graph(directed=False)
    graph.add_vertex(num_vertices)
    weight = graph.new_edge_property("double")
    graph.edge_properties["edge_weight"] = weight
    if edges:
        graph.add_edge_list(np.asarray(edges, dtype=float), eprops=[weight])
    return graph


def edge_values(graph: gt.Graph, u: int, v: int) -> tuple[float, float, float]:
    edge = graph.edge(u, v)
    if edge is None:
        raise AssertionError(f"Missing edge ({u}, {v}).")
    return (
        float(graph.ep["edge_weight"][edge]),
        float(graph.ep["original_edge_weight"][edge]),
        float(graph.ep["sampling_probability"][edge]),
    )


def weighted_volume(graph: gt.Graph) -> float:
    return float(
        np.asarray(
            graph.degree_property_map("total", weight=graph.ep["edge_weight"]).a
        ).sum()
    )


class SparsifierWeightTests(unittest.TestCase):
    def test_probability_one_preserves_original_weight(self) -> None:
        graph = weighted_graph(2, [(0, 1, 7.5)])
        sparsifier = DynamicGraphSparsifier(graph, sampling_constant=10_000)
        sparsifier.create_sparsifier()

        sampled, original, probability = edge_values(sparsifier.sparsified_graph, 0, 1)
        self.assertAlmostEqual(probability, 1.0)
        self.assertAlmostEqual(original, 7.5)
        self.assertAlmostEqual(sampled, 7.5)

    def test_nonloop_uses_original_weight_over_probability(self) -> None:
        graph = weighted_graph(2, [(0, 1, 4.0)])
        with mock.patch.object(np.random, "rand", return_value=0.0):
            sparsifier = DynamicGraphSparsifier(graph, sampling_constant=1)
            sparsifier.create_sparsifier()

        sampled, original, probability = edge_values(sparsifier.sparsified_graph, 0, 1)
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)
        self.assertAlmostEqual(original, 4.0)
        self.assertAlmostEqual(sampled, original / probability)

    def test_weighted_mass_scales_inclusion_probability(self) -> None:
        graph = weighted_graph(2, [(0, 1, 100.0)])
        sparsifier = DynamicGraphSparsifier(graph, sampling_constant=2)
        sparsifier.create_sparsifier()

        sampled, original, probability = edge_values(sparsifier.sparsified_graph, 0, 1)
        self.assertAlmostEqual(probability, 1.0)
        self.assertAlmostEqual(original, 100.0)
        self.assertAlmostEqual(sampled, 100.0)

    def test_self_loop_is_always_exact(self) -> None:
        graph = weighted_graph(2, [(0, 0, 1234.0), (0, 1, 2.0)])
        sparsifier = DynamicGraphSparsifier(graph, sampling_constant=0)
        sparsifier.create_sparsifier()

        sampled, original, probability = edge_values(sparsifier.sparsified_graph, 0, 0)
        self.assertAlmostEqual(probability, 1.0)
        self.assertAlmostEqual(original, 1234.0)
        self.assertAlmostEqual(sampled, 1234.0)
        self.assertIsNone(sparsifier.sparsified_graph.edge(0, 1))

    def test_weighted_increment_updates_original_mass_not_estimator_mass(self) -> None:
        graph = weighted_graph(2, [(0, 1, 2.0)])
        sparsifier = DynamicGraphSparsifier(graph, sampling_constant=10_000)
        sparsifier.create_sparsifier()

        graph.ep["edge_weight"][graph.edge(0, 1)] += 3.0
        sparsifier.update_sparsifier(np.asarray([[0, 1, 3.0]]), verbose=False)

        sampled, original, probability = edge_values(sparsifier.sparsified_graph, 0, 1)
        self.assertAlmostEqual(probability, 1.0)
        self.assertAlmostEqual(original, 5.0)
        self.assertAlmostEqual(sampled, 5.0)

    def test_weighted_increment_survives_vertex_resampling(self) -> None:
        graph = weighted_graph(300, [(0, 1, 1.0)])
        sparsifier = DynamicGraphSparsifier(graph, sampling_constant=10_000)
        sparsifier.create_sparsifier()

        graph.ep["edge_weight"][graph.edge(0, 1)] += 20.0
        sparsifier.update_sparsifier(np.asarray([[0, 1, 20.0]]), verbose=False)

        sampled, original, probability = edge_values(sparsifier.sparsified_graph, 0, 1)
        self.assertAlmostEqual(probability, 1.0)
        self.assertAlmostEqual(original, 21.0)
        self.assertAlmostEqual(sampled, 21.0)

    def test_self_loop_weighted_update_remains_exact(self) -> None:
        graph = weighted_graph(3, [(0, 0, 2.0)])
        sparsifier = DynamicGraphSparsifier(graph, sampling_constant=0)
        sparsifier.create_sparsifier()

        graph.ep["edge_weight"][graph.edge(0, 0)] += 1.0
        sparsifier.update_sparsifier(np.asarray([[0, 0, 1.0]]), verbose=False)

        sampled, original, probability = edge_values(sparsifier.sparsified_graph, 0, 0)
        self.assertAlmostEqual(probability, 1.0)
        self.assertAlmostEqual(original, 3.0)
        self.assertAlmostEqual(sampled, 3.0)

    def test_probability_one_preserves_total_volume_with_loops(self) -> None:
        graph = weighted_graph(
            3,
            [(0, 0, 11.0), (0, 1, 3.0), (1, 2, 5.0), (2, 2, 7.0)],
        )
        sparsifier = DynamicGraphSparsifier(graph, sampling_constant=10_000)
        sparsifier.create_sparsifier()

        self.assertAlmostEqual(
            weighted_volume(sparsifier.sparsified_graph), weighted_volume(graph)
        )

    def test_c3_to_c4_reconstruction_keeps_loop_volume_and_k(self) -> None:
        # A contracted C3-like graph: four massive old-cluster loops and a new star.
        edges = [(old, old, 1000.0) for old in range(4)]
        edges += [(u, v, 2.0) for u in range(4) for v in range(u + 1, 4)]
        edges += [(0, 4, 1.0)] + [(4, leaf, 1.0) for leaf in range(5, 15)]
        c3 = weighted_graph(15, edges)

        hierarchy = object.__new__(HierarchicalContractedStructure)
        hierarchy.sampling_constant = 80
        hierarchy.degree_trigger = 0.5
        hierarchy.k = 5
        hierarchy.verbose = False

        state = hierarchy._build_level_from_graph(
            level_idx_1based=4,
            graph_for_level=c3,
            input_base_map=[{vertex} for vertex in range(15)],
        )

        sparse_edges = state.semi.sparsified_graph.get_edges(
            [state.semi.sparsified_graph.ep["edge_weight"]]
        )
        sparse_loop_weight = float(
            sparse_edges[sparse_edges[:, 0] == sparse_edges[:, 1], 2].sum()
        )
        c4 = state.semi.contracted.contracted_graph
        c4_edges = c4.get_edges([c4.ep["edge_weight"]])
        c4_loop_weight = float(c4_edges[c4_edges[:, 0] == c4_edges[:, 1], 2].sum())

        self.assertEqual(state.semi.last_num_clusters, 5)
        self.assertEqual(int(c4.num_vertices()), 5)
        self.assertAlmostEqual(sparse_loop_weight, 4000.0)
        self.assertGreaterEqual(c4_loop_weight, 4000.0)

    def test_weighted_star_rounding_keeps_the_fifth_cluster_together(self) -> None:
        edges = [(old, old, 1000.0) for old in range(4)]
        edges += [(u, v, 10.0) for u in range(4) for v in range(u + 1, 4)]
        edges += [(0, 4, 1.0)] + [(4, leaf, 1.0) for leaf in range(5, 15)]
        graph = weighted_graph(15, edges)

        clusters = spectral_clustering(graph, 5)
        labels = np.full(15, -1, dtype=int)
        for label, cluster in enumerate(clusters):
            labels[np.asarray(cluster, dtype=int)] = label

        self.assertEqual(len(clusters), 5)
        self.assertEqual(len(set(labels[4:].tolist())), 1)
        self.assertEqual(len(set(labels[:4].tolist())), 4)
        self.assertNotIn(int(labels[4]), set(labels[:4].tolist()))


if __name__ == "__main__":
    unittest.main()
