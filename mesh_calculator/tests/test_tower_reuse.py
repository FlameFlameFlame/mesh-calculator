"""
Tests for on-corridor tower reuse in place_nodes_along_corridor().

Existing towers whose H3 cell falls directly on the corridor are used as
free waypoints — the corridor is split at each such tower and each segment
is processed independently.  Off-corridor relay injection was removed.
"""
import unittest
from unittest.mock import patch

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..data.cache import LOSResult
from ..network.graph import MeshSurface, Tower
from ..optimization.corridor import place_nodes_along_corridor


# ---------------------------------------------------------------------------
# Helpers
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
# Tests: on-corridor tower reuse
# ---------------------------------------------------------------------------

class TestOnCorridorTowerReuse(unittest.TestCase):
    """Existing towers on the corridor are reused as free waypoints."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_endpoints_preserved_no_existing_towers(
        self, mock_distance, mock_compute_los
    ):
        """With an empty surface.towers, endpoints are always preserved."""
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)

        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        corridor = make_corridor(5)
        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], 'cell_0', "First endpoint always included")
        self.assertEqual(nodes[-1], 'cell_4', "Last endpoint always included")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_tower_on_corridor_used_as_waypoint(
        self, mock_distance, mock_compute_los
    ):
        """A tower whose H3 cell is on the corridor splits it there."""
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)

        # cell_3 is on the corridor and has an existing tower
        existing_cell = cells['cell_3']
        _inject_tower(
            surface, 'cell_3',
            lat=existing_cell.lat, lon=existing_cell.lon,
        )

        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        corridor = make_corridor(6)
        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn('cell_0', nodes, "Start endpoint must be present")
        self.assertIn('cell_5', nodes, "End endpoint must be present")
        self.assertIn('cell_3', nodes, "On-corridor tower must appear")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_tower_already_on_corridor_not_double_counted(
        self, mock_distance, mock_compute_los
    ):
        """A tower on the corridor must not appear more than once."""
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
            "'cell_2' is on the corridor; must not appear more than once",
        )


if __name__ == '__main__':
    unittest.main()
