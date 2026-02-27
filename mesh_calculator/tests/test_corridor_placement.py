"""
Tests for corridor placement algorithm (MaxMin DP).

The greedy _walk_segment algorithm has been replaced with a MaxMin Bottleneck
Path DP that finds the globally optimal tower chain maximising minimum Fresnel
clearance.  These tests verify the DP's core behaviours:

  - Skips cells that have no LOS to the current tower
  - Finds the furthest/best visible cell (not just the immediate neighbour)
  - Always includes both corridor endpoints
  - Falls back to peak-based placement when no feasible chain exists
  - Short corridors (2 cells) are handled correctly
"""
import unittest
from unittest.mock import patch

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..data.cache import LOSResult
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


def make_compute_los_func(los_pairs, default_visible=False, clearance=10.0):
    """
    Return a compute_los mock that reads visibility from los_pairs.

    Keys are (src, dst) tuples; value is True/False.  Pairs not in the dict
    default to default_visible.  Always returns a LOSResult.
    """
    def _compute_los(src, dst, cells_arg, config, cache=None,
                     elevation_provider=None):
        is_vis = los_pairs.get((src, dst), default_visible)
        return LOSResult(
            clearance_m=clearance if is_vis else -999.0,
            path_loss_db=50.0,
            distance_m=1000.0,
            is_visible=is_vis,
        )
    return _compute_los


# ---------- DP: non-adjacent LOS ----------

class TestDPNonAdjacentLOS(unittest.TestCase):
    """DP correctly connects through the best visible cell, skipping others."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_skips_nonvisible_finds_further_visible(
        self, mock_distance, mock_compute_los
    ):
        """
        When intermediate cells have no LOS but a further cell does,
        the DP should select the further cell, not fall back to adjacent.

        Corridor: [0, 1, 2, 3, 4]
        LOS: 0→3 only (not 0→1, 0→2, 0→4); 3→4.
        Expected: [cell_0, cell_3, cell_4]  (cell_1 and cell_2 absent)
        """
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): False,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): True,
            ("cell_0", "cell_4"): False,
            ("cell_3", "cell_4"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertNotIn("cell_1", nodes,
            "cell_1 has no LOS from cell_0 and should be absent")
        self.assertNotIn("cell_2", nodes,
            "cell_2 has no LOS from cell_0 and should be absent")
        self.assertIn("cell_3", nodes,
            "cell_3 has LOS from cell_0 and should be selected")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_immediate_neighbour_chosen_when_only_option(
        self, mock_distance, mock_compute_los
    ):
        """
        When the immediate neighbour is the only cell with LOS, it is chosen.

        Corridor: [0, 1, 2, 3]
        LOS: 0→1 only; 1→3.
        """
        corridor = make_corridor(4)
        cells = make_cells(4)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): False,
            ("cell_1", "cell_3"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_1", nodes,
            "cell_1 is the only cell with LOS from cell_0 and must be chosen")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_finds_peak_past_valley(self, mock_distance, mock_compute_los):
        """
        DP should find a peak cell past a valley (no-LOS cell).

        Corridor: [0, 1, 2, 3, 4, 5]
        LOS: 0→1, 0→2, 0→4 (hilltop, past valley 3); 4→5.
        Expected: cell_4 chosen, not cell_2 (cell_4 is further and visible).
        """
        corridor = make_corridor(6)
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): True,
            ("cell_0", "cell_3"): False,   # valley
            ("cell_0", "cell_4"): True,    # hilltop
            ("cell_0", "cell_5"): False,
            ("cell_4", "cell_5"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_4", nodes,
            "Hilltop cell_4 should be selected; it has LOS from cell_0")
        self.assertNotIn("cell_2", nodes,
            "cell_2 should be skipped — cell_4 is a further, visible relay")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_scans_through_multiple_gaps(self, mock_distance, mock_compute_los):
        """
        DP finds the best relay even when several cells between them lack LOS.

        Corridor: [0..7]; 0→1, 0→4, 0→6 (furthest); 6→7.
        Expected: cell_6 chosen over cell_4.
        """
        corridor = make_corridor(8)
        cells = make_cells(8)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): False,
            ("cell_0", "cell_4"): True,
            ("cell_0", "cell_5"): False,
            ("cell_0", "cell_6"): True,    # furthest visible from cell_0
            ("cell_0", "cell_7"): False,
            ("cell_6", "cell_7"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_6", nodes,
            "cell_6 (furthest visible from cell_0) should be selected")
        self.assertNotIn("cell_4", nodes,
            "cell_4 should be skipped — cell_6 provides a better relay")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_direct_los_skips_intermediates(
        self, mock_distance, mock_compute_los
    ):
        """
        When all cells have LOS to each other, the DP picks only endpoints
        (max clearance path uses fewest hops).
        """
        corridor = make_corridor(6)
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)

        def distance_fn(src, dst):
            si = int(src.split("_")[1])
            di = int(dst.split("_")[1])
            return abs(di - si) * 20000.0

        mock_distance.side_effect = distance_fn
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], corridor[0], "First endpoint always included")
        self.assertEqual(nodes[-1], corridor[-1], "Last endpoint always included")


# ---------- DP: endpoints ----------

class TestEndpointHandling(unittest.TestCase):
    """Both corridor endpoints are always included."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_endpoints_included_all_los_clear(
        self, mock_distance, mock_compute_los
    ):
        """When all LOS is clear, both endpoints are in the result."""
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], "cell_0")
        self.assertEqual(nodes[-1], "cell_4")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_short_corridor_both_cells_returned(
        self, mock_distance, mock_compute_los
    ):
        """A 2-cell corridor returns exactly both cells."""
        corridor = make_corridor(2)
        cells = make_cells(2)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes, ["cell_0", "cell_1"])

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_fallback_includes_endpoints_when_no_los(
        self, mock_distance, mock_compute_los
    ):
        """
        When no feasible chain exists (all LOS blocked), the peak-based
        fallback still includes both corridor endpoints.
        """
        corridor = make_corridor(8)
        cells = make_cells(8)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=-999.0, path_loss_db=999.0,
            distance_m=1000.0, is_visible=False,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_0", nodes, "Start endpoint must always be present")
        self.assertIn("cell_7", nodes, "End endpoint must always be present")


if __name__ == '__main__':
    unittest.main()
