"""
Tests for corridor tower reuse in place_nodes_along_corridor().

Before the DP runs, any existing surface tower within max_visibility_m of
the corridor (sampled every ~20th cell) is injected as a relay candidate.
Once injected the tower is treated as a free waypoint (the corridor is
split there) so the DP can route through it.

When no existing towers are nearby the function should behave identically
to the baseline.
"""
import unittest
from unittest.mock import patch

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..data.cache import LOSResult
from ..network.graph import MeshSurface, Tower
from ..optimization.corridor import place_nodes_along_corridor


# ---------------------------------------------------------------------------
# Helpers (mirrors test_corridor_placement.py conventions)
# ---------------------------------------------------------------------------

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
    return [f"cell_{i}" for i in range(n)]


def make_compute_los_func(los_pairs, default_visible=False, clearance=10.0):
    def _compute_los(src, dst, cells_arg, config, cache=None,
                     elevation_provider=None, corridor_cells=None):
        is_vis = los_pairs.get((src, dst), default_visible)
        return LOSResult(
            clearance_m=clearance if is_vis else -999.0,
            path_loss_db=50.0,
            distance_m=1000.0,
            is_visible=is_vis,
        )
    return _compute_los


def _inject_tower(surface, h3_index, lat, lon, elevation=100.0):
    """Register a pre-existing tower on the surface, adding cell if needed."""
    if h3_index not in surface.cells:
        surface.cells[h3_index] = H3Cell(
            h3_index=h3_index, lat=lat, lon=lon,
            elevation=elevation, has_road=False, has_tower=True,
        )
    tower = Tower(
        tower_id=surface._next_tower_id,
        h3_index=h3_index,
        lat=lat,
        lon=lon,
        source='seed',
    )
    surface.towers[tower.tower_id] = tower
    surface.tower_by_h3[h3_index] = tower
    surface.visibility_graph.add_tower(tower)
    surface._next_tower_id += 1
    return tower


# ---------------------------------------------------------------------------
# Tests: nearby tower IS injected
# ---------------------------------------------------------------------------

class TestTowerReuseInjected(unittest.TestCase):
    """An existing tower near the corridor is injected as a relay candidate."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_nearby_tower_appears_in_result(
        self, mock_distance, mock_compute_los
    ):
        """
        A relay tower within max_visibility_m of the corridor should be
        injected and appear in the selected nodes.

        Setup:
          Corridor: [cell_0, cell_1, cell_2, cell_3, cell_4, cell_5]
          Existing tower 'relay_tower' is near cell_3 (injected there).
          h3_distance returns 500 m for relay pairs (within visibility),
          9999 m for all other direct corridor cell pairs — just large
          enough to block direct LOS beyond immediate neighbours but still
          within max_visibility_m so the relay is picked up.

        After injection, relay_tower sits in the middle of the corridor.
        Because it is already in surface.tower_by_h3 it becomes a free
        waypoint and the corridor is split there; the DP for each half
        can use all-visible LOS (default_visible=True) to find a path.
        """
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)

        relay_h3 = 'relay_tower'
        # Place relay between cell_2 and cell_3 (lat between them)
        _inject_tower(surface, relay_h3, lat=40.0125, lon=44.0125)

        def dist_fn(a, b):
            if 'relay' in a or 'relay' in b:
                return 500.0   # within max_visibility_m
            return 9999.0      # still within visibility but large

        mock_distance.side_effect = dist_fn

        # All LOS calls return visible so both halves of the corridor
        # can be resolved by the DP.
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        corridor = make_corridor(6)
        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn(
            relay_h3, nodes,
            "Injected relay tower must appear in the selected nodes",
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_endpoints_preserved_with_relay_in_middle(
        self, mock_distance, mock_compute_los
    ):
        """
        When a relay tower is injected in the middle of the corridor the
        original corridor endpoints (cell_0 and cell_N-1) are still
        present in the result.

        Distance mock is tuned so the relay is closest to cell_3
        (200 m away), making best_pos insert it near the middle of the
        corridor rather than at position 0.  After insertion:
          [cell_0, .., cell_2, relay_tower, cell_3, .., cell_5]
        relay_tower is in tower_by_h3 so the corridor splits there.
        With all LOS visible the DP returns endpoints of each segment,
        giving a result that contains cell_0, relay_tower, and cell_5.
        """
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)

        relay_h3 = 'relay_tower'
        _inject_tower(surface, relay_h3, lat=40.0125, lon=44.0125)

        def dist_fn(a, b):
            # relay is 200 m from cell_3; distance grows for farther cells
            if 'relay' in a or 'relay' in b:
                other = b if ('relay' in a) else a
                try:
                    idx = int(other.split('_')[1])
                    return 200.0 + abs(idx - 3) * 500.0
                except (ValueError, IndexError):
                    return 500.0
            return 1000.0   # corridor cell pairs within visibility

        mock_distance.side_effect = dist_fn
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        corridor = make_corridor(6)
        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn('cell_0', nodes, "Start endpoint must be present")
        self.assertIn('cell_5', nodes, "End endpoint must be present")


# ---------------------------------------------------------------------------
# Tests: no nearby towers — baseline unchanged
# ---------------------------------------------------------------------------

class TestTowerReuseNoNearbyTowers(unittest.TestCase):
    """When no existing towers are within range, behavior is unchanged."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_no_existing_towers_preserves_endpoints(
        self, mock_distance, mock_compute_los
    ):
        """With an empty surface.towers, no injection happens."""
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)   # no towers

        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        corridor = make_corridor(5)
        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], 'cell_0', "First endpoint always included")
        self.assertEqual(
            nodes[-1], 'cell_4', "Last endpoint always included"
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_far_tower_not_injected(
        self, mock_distance, mock_compute_los
    ):
        """A tower beyond max_visibility_m must not be injected."""
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)

        _inject_tower(surface, 'far_tower', lat=60.0, lon=80.0)

        max_vis = self.config.max_visibility_m
        # All distances exceed max_visibility_m
        mock_distance.return_value = max_vis + 100_000.0

        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        corridor = make_corridor(5)
        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertNotIn(
            'far_tower', nodes,
            "Tower beyond max_visibility_m must not appear in nodes",
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_tower_already_on_corridor_not_double_injected(
        self, mock_distance, mock_compute_los
    ):
        """
        A tower whose h3_index is already in the corridor is skipped
        during injection (corridor_set check) so it never appears twice.
        """
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)

        existing_cell = cells['cell_2']
        _inject_tower(
            surface, 'cell_2',
            lat=existing_cell.lat, lon=existing_cell.lon,
        )

        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        corridor = make_corridor(5)
        nodes = place_nodes_along_corridor(corridor, surface)

        count = nodes.count('cell_2')
        self.assertLessEqual(
            count, 1,
            "'cell_2' is in the corridor already; must not be injected again",
        )


if __name__ == '__main__':
    unittest.main()
