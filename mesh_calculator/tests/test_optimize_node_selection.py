"""
Tests for optimize_node_selection fixes.

Fix #5: Selected nodes must preserve corridor order.
    Bug: after scoring and sorting by score (descending), the selected nodes
    are returned in score order instead of corridor order. Downstream code
    (install_nodes, visibility graph) expects corridor order.

Fix #6: Removing a node must not break chain connectivity.
    Bug: top-N selection picks highest-scored nodes regardless of whether
    consecutive pairs in the result have LOS to each other. This can create
    a disconnected chain where towers can't see their neighbors.
"""
import unittest
from unittest.mock import patch

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..optimization.corridor import optimize_node_selection


def make_cells_with_elevations(elevations):
    """Create cells with specified elevations: cell_0..cell_n-1."""
    cells = {}
    for i, elev in enumerate(elevations):
        h3_idx = f"cell_{i}"
        cells[h3_idx] = H3Cell(
            h3_index=h3_idx,
            lat=40.0 + i * 0.005,
            lon=44.0 + i * 0.005,
            elevation=elev,
            has_road=True,
        )
    return cells


def make_los_func(los_pairs):
    """Create a has_los mock from a dict of (src, dst) → bool (symmetric)."""
    def los_check(src, dst, cells_arg, config, cache=None,
                  elevation_provider=None):
        return los_pairs.get((src, dst), los_pairs.get((dst, src), False))
    return los_check


# ---------- Fix #5: Corridor Order ----------

class TestCorridorOrderPreservation(unittest.TestCase):
    """Fix #5: Selected nodes must be returned in corridor order."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
        )

    @patch('mesh_calculator.optimization.corridor.has_los')
    def test_selected_nodes_in_corridor_order(self, mock_los):
        """After optimization, nodes must follow corridor order, not score order.

        7 nodes → reduce to 5. All LOS clear, so scores = elevation only.
        Elevations: cell_4 > cell_2 > cell_1 > cell_3 > cell_5
        Score order:  [cell_4, cell_2, cell_1, cell_3, cell_5]
        Corridor order: [cell_1, cell_2, cell_4] (top 3)

        Bug: old code returns [cell_0, cell_4, cell_2, cell_1, cell_6] (score order)
        Fix: must return in corridor order
        """
        elevations = [100, 300, 400, 200, 500, 100, 100]
        cells = make_cells_with_elevations(elevations)
        nodes = [f"cell_{i}" for i in range(7)]

        mock_los.return_value = True

        result = optimize_node_selection(
            nodes, max_nodes=5, cells=cells, config=self.config
        )

        corridor_indices = [nodes.index(n) for n in result]
        self.assertEqual(corridor_indices, sorted(corridor_indices),
            f"Result must be in corridor order, got: {result}")
        self.assertEqual(result[0], "cell_0")
        self.assertEqual(result[-1], "cell_6")

    @patch('mesh_calculator.optimization.corridor.has_los')
    def test_order_preserved_with_varied_scores(self, mock_los):
        """Verify ordering with 9 nodes where score and corridor order differ.

        9 nodes → 5. Highest-scored: cell_7(600), cell_3(500), cell_1(400).
        Corridor order of selection: [cell_1, cell_3, cell_7].
        """
        elevations = [100, 400, 100, 500, 100, 100, 100, 600, 100]
        cells = make_cells_with_elevations(elevations)
        nodes = [f"cell_{i}" for i in range(9)]

        mock_los.return_value = True

        result = optimize_node_selection(
            nodes, max_nodes=5, cells=cells, config=self.config
        )

        corridor_indices = [nodes.index(n) for n in result]
        self.assertEqual(corridor_indices, sorted(corridor_indices),
            f"Result must be in corridor order, got: {result}")


# ---------- Fix #6: Chain Connectivity ----------

class TestChainConnectivity(unittest.TestCase):
    """Fix #6: Node removal must not break chain connectivity."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
        )

    @patch('mesh_calculator.optimization.corridor.has_los')
    def test_bridge_node_kept_when_removal_breaks_los(self, mock_los):
        """A low-scored node that bridges an LOS gap must be kept.

        6 nodes → reduce to 4.
        cell_2 has lowest score but cell_0→cell_2 has no LOS and
        cell_1→cell_3 has no LOS, making both cell_1 and cell_2 critical.

        Bug: old code drops cell_2 (lowest score) → chain broken.
        Fix: cell_2 kept because removing it would disconnect cell_1 from cell_3.
        """
        elevations = [100, 500, 50, 400, 200, 100]
        cells = make_cells_with_elevations(elevations)
        nodes = [f"cell_{i}" for i in range(6)]

        los_pairs = {
            # Consecutive: all OK
            ("cell_0", "cell_1"): True,
            ("cell_1", "cell_2"): True,
            ("cell_2", "cell_3"): True,
            ("cell_3", "cell_4"): True,
            ("cell_4", "cell_5"): True,
            # Skip-1
            ("cell_0", "cell_2"): False,  # cell_1 NOT removable
            ("cell_1", "cell_3"): False,  # cell_2 NOT removable
            ("cell_2", "cell_4"): True,   # cell_3 removable
            ("cell_3", "cell_5"): True,   # cell_4 removable
            # Longer hops
            ("cell_1", "cell_4"): True,
            ("cell_2", "cell_5"): True,
        }
        mock_los.side_effect = make_los_func(los_pairs)

        result = optimize_node_selection(
            nodes, max_nodes=4, cells=cells, config=self.config
        )

        # cell_2 must be kept (bridge)
        self.assertIn("cell_2", result,
            "cell_2 must be kept — removing it breaks LOS between cell_1 and cell_3")

        # Chain must be connected
        for i in range(len(result) - 1):
            pair_los = los_pairs.get(
                (result[i], result[i+1]),
                los_pairs.get((result[i+1], result[i]), False)
            )
            self.assertTrue(pair_los,
                f"LOS gap between consecutive nodes {result[i]} and {result[i+1]}")

    @patch('mesh_calculator.optimization.corridor.has_los')
    def test_force_reduces_to_limit_when_no_los_safe_removal(self, mock_los):
        """When every node is a critical bridge, force-remove to reach the hard limit.

        5 nodes → target 3. Only consecutive pairs have LOS.
        No middle node can be safely removed without breaking the LOS chain,
        so the lowest-scored nodes are removed unconditionally.
        """
        elevations = [100, 200, 100, 300, 100]
        cells = make_cells_with_elevations(elevations)
        nodes = [f"cell_{i}" for i in range(5)]

        # Only consecutive pairs have LOS — no skip-one LOS
        los_pairs = {
            ("cell_0", "cell_1"): True,
            ("cell_1", "cell_2"): True,
            ("cell_2", "cell_3"): True,
            ("cell_3", "cell_4"): True,
        }
        mock_los.side_effect = make_los_func(los_pairs)

        result = optimize_node_selection(
            nodes, max_nodes=3, cells=cells, config=self.config
        )

        # Hard limit must be enforced even though LOS chain breaks
        self.assertEqual(len(result), 3,
            "Force-removal must reduce to the hard max_nodes limit")
        # Endpoints are always preserved
        self.assertEqual(result[0], "cell_0", "Start node must be kept")
        self.assertEqual(result[-1], "cell_4", "End node must be kept")

    @patch('mesh_calculator.optimization.corridor.has_los')
    def test_consecutive_pairs_always_have_los(self, mock_los):
        """General case: all consecutive pairs in result must have LOS.

        8 nodes → 5. Complex LOS topology with gaps.
        """
        elevations = [100, 200, 300, 100, 500, 100, 400, 100]
        cells = make_cells_with_elevations(elevations)
        nodes = [f"cell_{i}" for i in range(8)]

        los_pairs = {
            # Consecutive: all OK
            ("cell_0", "cell_1"): True,
            ("cell_1", "cell_2"): True,
            ("cell_2", "cell_3"): True,
            ("cell_3", "cell_4"): True,
            ("cell_4", "cell_5"): True,
            ("cell_5", "cell_6"): True,
            ("cell_6", "cell_7"): True,
            # Skip-1
            ("cell_0", "cell_2"): True,
            ("cell_1", "cell_3"): False,  # cell_2 critical bridge
            ("cell_2", "cell_4"): True,
            ("cell_3", "cell_5"): True,
            ("cell_4", "cell_6"): True,
            ("cell_5", "cell_7"): True,
            # Longer
            ("cell_0", "cell_3"): False,
            ("cell_1", "cell_4"): True,
            ("cell_2", "cell_5"): True,
            ("cell_3", "cell_6"): True,
            ("cell_4", "cell_7"): True,
            ("cell_2", "cell_6"): True,
            ("cell_3", "cell_7"): True,
        }
        mock_los.side_effect = make_los_func(los_pairs)

        result = optimize_node_selection(
            nodes, max_nodes=5, cells=cells, config=self.config
        )

        # Every consecutive pair must have LOS
        for i in range(len(result) - 1):
            pair_los = los_pairs.get(
                (result[i], result[i+1]),
                los_pairs.get((result[i+1], result[i]), False)
            )
            self.assertTrue(pair_los,
                f"LOS gap between consecutive nodes {result[i]} and {result[i+1]}")

        self.assertEqual(result[0], "cell_0")
        self.assertEqual(result[-1], "cell_7")


# ---------- Edge Cases ----------

class TestOptimizeEdgeCases(unittest.TestCase):
    """Edge cases for optimize_node_selection."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
        )

    def test_nodes_under_limit_unchanged(self):
        """If nodes <= max_nodes, return unchanged."""
        cells = make_cells_with_elevations([100, 200, 300])
        nodes = ["cell_0", "cell_1", "cell_2"]

        result = optimize_node_selection(
            nodes, max_nodes=5, cells=cells, config=self.config
        )
        self.assertEqual(result, nodes)

    def test_max_two_returns_endpoints(self):
        """max_nodes=2 should return just start and end."""
        cells = make_cells_with_elevations([100, 200, 300, 400])
        nodes = [f"cell_{i}" for i in range(4)]

        result = optimize_node_selection(
            nodes, max_nodes=2, cells=cells, config=self.config
        )
        self.assertEqual(result, ["cell_0", "cell_3"])

    @patch('mesh_calculator.optimization.corridor.has_los')
    def test_reduces_to_target_when_possible(self, mock_los):
        """When all removals are safe, reduce to exactly max_nodes."""
        elevations = [100, 200, 300, 400, 500, 200, 100]
        cells = make_cells_with_elevations(elevations)
        nodes = [f"cell_{i}" for i in range(7)]

        # All pairs have LOS — every node is safely removable
        mock_los.return_value = True

        result = optimize_node_selection(
            nodes, max_nodes=4, cells=cells, config=self.config
        )

        self.assertEqual(len(result), 4,
            "Should reduce to exactly max_nodes when all removals are safe")
        self.assertEqual(result[0], "cell_0")
        self.assertEqual(result[-1], "cell_6")


if __name__ == '__main__':
    unittest.main()
