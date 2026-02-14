"""
Tests for corridor constraint enforcement.

Fix #9: tower_separation_m — skip cells closer than minimum separation.
Fix #10: hop_limit — warn when corridor exceeds hop limit.
"""
import unittest
from unittest.mock import patch

from mesh_calculator.core.config import MeshConfig
from mesh_calculator.core.grid import H3Cell
from mesh_calculator.network.graph import MeshSurface
from mesh_calculator.optimization.corridor import place_nodes_along_corridor


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


def distance_per_cell(spacing_m):
    """Return a distance function where distance = cell index diff * spacing."""
    def dist_fn(src, dst):
        src_idx = int(src.split("_")[1])
        dst_idx = int(dst.split("_")[1])
        return abs(dst_idx - src_idx) * spacing_m
    return dist_fn


# ---------- Fix #9: Tower Separation ----------

class TestTowerSeparation(unittest.TestCase):
    """Fix #9: tower_separation_m enforced during corridor placement."""

    def setUp(self):
        self.config = MeshConfig(
            tower_separation_m=5000.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
        )

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_close_cells_skipped(self, mock_distance, mock_los):
        """Cells closer than tower_separation_m are skipped."""
        # 10 cells, 1km apart. Separation=5km. All have LOS.
        # From cell_0: cells 1-4 are too close (<5km), cell_5+ are OK.
        # Expected: cell_0, cell_5 (or further), ..., cell_9
        corridor = make_corridor(10)
        cells = make_cells(10)
        surface = MeshSurface(cells, self.config)

        mock_distance.side_effect = distance_per_cell(1000.0)
        mock_los.return_value = True

        nodes = place_nodes_along_corridor(corridor, surface)

        # cell_1 through cell_4 should be skipped (< 5km from cell_0)
        for i in range(1, 5):
            self.assertNotIn(f"cell_{i}", nodes,
                             f"cell_{i} is only {i}km from cell_0, "
                             f"should be skipped (separation=5km)")

        # cell_0 and cell_9 must always be present
        self.assertEqual(nodes[0], "cell_0")
        self.assertEqual(nodes[-1], "cell_9")

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_endpoint_always_allowed(self, mock_distance, mock_los):
        """Corridor endpoint is included even if within separation distance."""
        # 4 cells, 2km apart. Total length = 6km.
        # Separation=5km. From cell_0, cell_1 and cell_2 are too close.
        # cell_3 (endpoint, 6km away) should be included.
        corridor = make_corridor(4)
        cells = make_cells(4)
        surface = MeshSurface(cells, self.config)

        mock_distance.side_effect = distance_per_cell(2000.0)
        mock_los.return_value = True

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_0", nodes)
        self.assertIn("cell_3", nodes)

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_separation_zero_no_filtering(self, mock_distance, mock_los):
        """tower_separation_m=0 means no separation filtering."""
        config = MeshConfig(
            tower_separation_m=0.0,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
        )
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, config)

        mock_distance.side_effect = distance_per_cell(1000.0)
        mock_los.return_value = True

        nodes = place_nodes_along_corridor(corridor, surface)

        # With separation=0 and all LOS, furthest visible is endpoint
        self.assertIn("cell_0", nodes)
        self.assertIn("cell_4", nodes)

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_forced_advance_still_works(self, mock_distance, mock_los):
        """When no LOS found, forced advance works regardless of separation."""
        # All cells within separation, no LOS at all.
        # Forced advance must still move forward to avoid infinite loop.
        corridor = make_corridor(4)
        cells = make_cells(4)
        surface = MeshSurface(cells, self.config)

        mock_distance.side_effect = distance_per_cell(1000.0)
        mock_los.return_value = False

        nodes = place_nodes_along_corridor(corridor, surface)

        # Should still include start and end
        self.assertEqual(nodes[0], "cell_0")
        self.assertEqual(nodes[-1], "cell_3")


# ---------- Fix #10: Hop Limit ----------

class TestHopLimit(unittest.TestCase):
    """Fix #10: hop_limit validated after corridor placement."""

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    @patch('mesh_calculator.optimization.corridor.logger')
    def test_within_hop_limit_no_warning(
        self, mock_logger, mock_distance, mock_los
    ):
        """Corridor with <= hop_limit hops produces no warning."""
        config = MeshConfig(
            hop_limit=5,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
            tower_separation_m=0.0,
        )
        # 4 cells = 3 hops, hop_limit=5 → OK
        corridor = make_corridor(4)
        cells = make_cells(4)
        surface = MeshSurface(cells, config)

        mock_distance.return_value = 1000.0
        mock_los.return_value = True

        place_nodes_along_corridor(corridor, surface)

        # Check no warning about hop limit
        for call in mock_logger.warning.call_args_list:
            self.assertNotIn("hop", str(call).lower(),
                             "Should not warn when within hop limit")

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    @patch('mesh_calculator.optimization.corridor.logger')
    def test_exceeds_hop_limit_warns(
        self, mock_logger, mock_distance, mock_los
    ):
        """Corridor with > hop_limit hops logs a warning."""
        config = MeshConfig(
            hop_limit=2,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
            tower_separation_m=0.0,
        )
        # 6 cells, each only sees next cell → 5 hops. hop_limit=2 → warn.
        corridor = make_corridor(6)
        cells = make_cells(6)
        surface = MeshSurface(cells, config)

        mock_distance.return_value = 1000.0

        def only_adjacent_los(src, dst, *args, **kwargs):
            si = int(src.split("_")[1])
            di = int(dst.split("_")[1])
            return abs(si - di) == 1

        mock_los.side_effect = only_adjacent_los

        place_nodes_along_corridor(corridor, surface)

        # Should have warned about hop limit
        warning_calls = [
            str(c) for c in mock_logger.warning.call_args_list
        ]
        hop_warnings = [c for c in warning_calls if "hop" in c.lower()]
        self.assertTrue(len(hop_warnings) > 0,
                        "Should warn when corridor exceeds hop limit")

    @patch('mesh_calculator.optimization.corridor.has_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    @patch('mesh_calculator.optimization.corridor.logger')
    def test_exceeds_hop_limit_still_returns_all_nodes(
        self, mock_logger, mock_distance, mock_los
    ):
        """Even when hop limit exceeded, all nodes are still returned."""
        config = MeshConfig(
            hop_limit=2,
            max_visibility_m=70000.0,
            max_nodes_per_road=100,
            tower_separation_m=0.0,
        )
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, config)

        mock_distance.return_value = 1000.0

        def only_adjacent_los(src, dst, *args, **kwargs):
            si = int(src.split("_")[1])
            di = int(dst.split("_")[1])
            return abs(si - di) == 1

        mock_los.side_effect = only_adjacent_los

        nodes = place_nodes_along_corridor(corridor, surface)

        # All nodes should be present (not truncated)
        self.assertEqual(nodes[0], "cell_0")
        self.assertEqual(nodes[-1], "cell_4")
        self.assertEqual(len(nodes), 5)
