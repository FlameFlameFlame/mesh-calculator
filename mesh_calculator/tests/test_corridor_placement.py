"""
Tests for corridor placement algorithm fixes.

Fix #2: Fallback next-cell in corridor is accepted without LOS verification.
        The scan loop starts at current_idx+2, leaving current_idx+1 unchecked.
Fix #3: Greedy scan breaks at first LOS failure instead of scanning further.
Fix #4: Final endpoint is forced without LOS check to last placed node.
"""
import unittest
from unittest.mock import patch, Mock

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..network.graph import MeshSurface
from ..optimization.corridor import place_nodes_along_corridor


def make_cells(n, elevation=100.0):
    """Create n fake H3 cells labeled cell_0..cell_n-1."""
    cells = {}
    for i in range(n):
        h3_idx = f"cell_{i}"
        cells[h3_idx] = H3Cell(
            h3_index=h3_idx,
            lat=40.0 + i * 0.005,
            lon=44.0 + i * 0.005,
            elevation=elevation,
            has_road=True,
        )
    return cells


def make_corridor(n):
    """Create a corridor: ['cell_0', 'cell_1', ..., 'cell_n-1']."""
    return [f"cell_{i}" for i in range(n)]


def make_los_func(los_pairs):
    """Create a has_los mock function from a dict of (src, dst) → bool."""
    def los_check(src, dst, cells_arg, config, cache=None,
                  elevation_provider=None):
        return los_pairs.get((src, dst), False)
    return los_check


# ---------- Fix #2 ----------

class TestFallbackNextCellLOS(unittest.TestCase):
    """Fix #2: The immediate next cell must be LOS-checked, not assumed visible.

    Bug: old code sets corridor[current+1] as the default "furthest visible"
    and starts the scan at current+2. If the first scan cell has no LOS, it
    breaks and falls back to current+1 — which was never LOS-checked.
    """

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
            tower_separation_m=0.0,
        )

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_fallback_skipped_when_further_cell_has_los(
        self, mock_distance, mock_los
    ):
        """When cell_1 has no LOS but cell_3 does, place at cell_3 (not cell_1).

        Corridor: [0, 1, 2, 3, 4]
        - 0->1: no LOS
        - 0->2: no LOS (in old code, this is first scan cell, causes break)
        - 0->3: LOS
        - 0->4: no LOS

        Old behavior: break at 0->2, fall back to cell_1 (unchecked) → places cell_1
        Fixed: scans 0->1 (fail), 0->2 (fail), 0->3 (LOS!) → places cell_3
        """
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_los.side_effect = make_los_func({
            ("cell_0", "cell_1"): False,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): True,
            ("cell_0", "cell_4"): False,
            ("cell_3", "cell_4"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertNotIn("cell_1", nodes,
            "cell_1 should NOT be placed — it has no LOS from cell_0")
        self.assertIn("cell_3", nodes,
            "cell_3 should be placed — it has LOS from cell_0")

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_los_checked_for_immediate_neighbor(
        self, mock_distance, mock_los
    ):
        """Backward scan: farthest cell is checked first; immediate neighbor
        is only checked when all farther cells fail LOS."""
        corridor = make_corridor(4)
        cells = make_cells(4)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        # Only the immediate next cell (cell_1) has LOS — all farther cells fail.
        mock_los.side_effect = make_los_func({
            ("cell_0", "cell_3"): False,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_1"): True,
            ("cell_1", "cell_3"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        # cell_1 must be chosen (it's the only hop with LOS from cell_0)
        self.assertIn("cell_1", nodes,
            "cell_1 must be placed when it is the only cell with LOS from cell_0")


# ---------- Fix #3 ----------

class TestGreedyScanContinuesPastFailure(unittest.TestCase):
    """Fix #3: Don't break on first LOS failure — scan further along corridor.

    Bug: `else: break` stops scanning when a cell has no LOS. But a cell
    further along might be on higher ground with clear LOS.
    """

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
            tower_separation_m=0.0,
        )

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_scan_past_valley_to_hilltop(self, mock_distance, mock_los):
        """Valley at cell_3 blocks LOS, but hilltop at cell_4 is visible.

        Corridor: [0, 1, 2, 3, 4, 5]
        - 0->1: LOS, 0->2: LOS, 0->3: no LOS (valley), 0->4: LOS (hilltop)

        Old behavior: places at cell_2, breaks at cell_3 (never checks cell_4)
        Fixed: continues scanning, finds cell_4, places there
        """
        corridor = make_corridor(6)
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_los.side_effect = make_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): True,
            ("cell_0", "cell_3"): False,  # valley
            ("cell_0", "cell_4"): True,   # hilltop
            ("cell_0", "cell_5"): False,
            ("cell_4", "cell_5"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_4", nodes,
            "Should scan past valley (cell_3) and find hilltop (cell_4)")
        self.assertNotIn("cell_2", nodes,
            "cell_2 should be skipped — cell_4 is further and visible")

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_multiple_gaps_scanned_through(self, mock_distance, mock_los):
        """Multiple LOS failures between visible cells should be scanned through.

        Corridor: [0, 1, 2, 3, 4, 5, 6, 7]
        - 0->1: LOS, 0->2: no, 0->3: no, 0->4: LOS, 0->5: no, 0->6: LOS
        """
        corridor = make_corridor(8)
        cells = make_cells(8)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_los.side_effect = make_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): False,
            ("cell_0", "cell_4"): True,
            ("cell_0", "cell_5"): False,
            ("cell_0", "cell_6"): True,   # Furthest visible from 0
            ("cell_0", "cell_7"): False,
            ("cell_6", "cell_7"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_6", nodes,
            "Should find cell_6 by scanning through multiple gaps")
        self.assertNotIn("cell_4", nodes,
            "cell_4 should be skipped — cell_6 is further and visible")

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_distance_limit_still_stops_scan(self, mock_distance, mock_los):
        """Scan should stop when distance exceeds max_visibility_m."""
        corridor = make_corridor(6)
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)

        def distance_fn(src, dst):
            src_idx = int(src.split("_")[1])
            dst_idx = int(dst.split("_")[1])
            return abs(dst_idx - src_idx) * 20000.0  # 20km per hop

        mock_distance.side_effect = distance_fn
        mock_los.return_value = True

        nodes = place_nodes_along_corridor(corridor, surface)

        # 20km/hop, 70km max → can see 3 hops ahead → need intermediate nodes
        self.assertGreater(len(nodes), 2,
            "Distance limit should force intermediate node placement")


# ---------- Fix #4 ----------

class TestEndpointLOSVerification(unittest.TestCase):
    """Fix #4: Verify LOS to the corridor endpoint.

    With fixes #2 and #3, the scan loop naturally reaches the endpoint.
    This tests that the endpoint is properly connected through the scan
    rather than blindly appended.
    """

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
            tower_separation_m=0.0,
        )

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_endpoint_reached_through_bridge_node(self, mock_distance, mock_los):
        """When last placed can't see endpoint, scan finds a bridge.

        Corridor: [0, 1, 2, 3, 4, 5, 6, 7]
        - 0 sees up to 3 → place 3
        - 3 can't see 4,5,6,7 → forced advance to 4
        - 4 sees 6 → place 6
        - 6 sees 7 → place 7

        Without fix #3: from cell_3, would break at cell_4 (no LOS),
        and the loop would force advance one cell at a time without
        finding the bridge at cell_6.
        """
        corridor = make_corridor(8)
        cells = make_cells(8)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_los.side_effect = make_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): True,
            ("cell_0", "cell_3"): True,
            ("cell_0", "cell_4"): False,
            ("cell_0", "cell_5"): False,
            ("cell_0", "cell_6"): False,
            ("cell_0", "cell_7"): False,
            ("cell_3", "cell_4"): False,
            ("cell_3", "cell_5"): False,
            ("cell_3", "cell_6"): False,
            ("cell_3", "cell_7"): False,
            # After forced advance to 4:
            ("cell_4", "cell_5"): False,
            ("cell_4", "cell_6"): True,   # Bridge!
            ("cell_4", "cell_7"): False,
            ("cell_6", "cell_7"): True,   # Bridge to endpoint
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_6", nodes,
            "Bridge node cell_6 should be found after forced advance")
        self.assertEqual(nodes[-1], "cell_7",
            "Endpoint should be the last placed node")
        # The link 6->7 should have LOS
        self.assertIn("cell_7", nodes)

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_endpoint_included_when_all_los_clear(self, mock_distance, mock_los):
        """When all LOS is clear, endpoint should be included."""
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_los.return_value = True

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], "cell_0")
        self.assertEqual(nodes[-1], "cell_4")

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_short_corridor_still_works(self, mock_distance, mock_los):
        """Corridor of 2 cells should produce both cells."""
        corridor = make_corridor(2)
        cells = make_cells(2)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_los.return_value = True

        nodes = place_nodes_along_corridor(corridor, surface)
        self.assertEqual(nodes, ["cell_0", "cell_1"])


if __name__ == '__main__':
    unittest.main()
