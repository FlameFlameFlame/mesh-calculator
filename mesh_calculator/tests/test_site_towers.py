"""
Tests for Fix #11: Place tower at every site location.

Every site must have a tower placed at its h3_index with source='site',
even if the site is a lone P1 site or all corridor searches fail.
"""
import unittest
from unittest.mock import patch, MagicMock
import networkx as nx

from mesh_calculator.core.config import MeshConfig
from mesh_calculator.core.grid import H3Cell
from mesh_calculator.data.sites import Site
from mesh_calculator.network.graph import MeshSurface
from mesh_calculator.optimization.hierarchical import connect_sites_by_priority


def make_cells(h3_indices, elevation=100.0):
    """Create cells dict for given h3 indices."""
    cells = {}
    for i, h3_idx in enumerate(h3_indices):
        cells[h3_idx] = H3Cell(
            h3_index=h3_idx,
            lat=40.0 + i * 0.01,
            lon=44.0 + i * 0.01,
            elevation=elevation,
            has_road=True,
            is_in_boundary=True,
        )
    return cells


class TestSiteTowersPlaced(unittest.TestCase):
    """Fix #11: Every site gets a tower at its h3_index."""

    def test_lone_p1_site_gets_tower(self):
        """A single P1 site (no mesh possible) still gets a tower."""
        cells = make_cells(['site_cell'])
        config = MeshConfig()
        surface = MeshSurface(cells, config)
        routing_graph = nx.DiGraph()

        site = Site(name='Lonely', lat=40.0, lon=44.0, priority=1,
                    h3_index='site_cell')

        connect_sites_by_priority([site], surface, routing_graph)

        self.assertIn('site_cell', surface.tower_by_h3)
        self.assertEqual(surface.tower_by_h3['site_cell'].source, 'site')

    @patch('mesh_calculator.optimization.hierarchical.find_road_corridor',
           return_value=None)
    def test_all_sites_get_towers_when_corridors_fail(self, mock_corridor):
        """All sites get towers even when all corridor searches fail."""
        cell_ids = ['p1_cell_a', 'p1_cell_b', 'p2_cell']
        cells = make_cells(cell_ids)
        config = MeshConfig()
        surface = MeshSurface(cells, config)
        routing_graph = nx.DiGraph()

        sites = [
            Site(name='CityA', lat=40.0, lon=44.0, priority=1,
                 h3_index='p1_cell_a'),
            Site(name='CityB', lat=40.01, lon=44.01, priority=1,
                 h3_index='p1_cell_b'),
            Site(name='Town', lat=40.02, lon=44.02, priority=2,
                 h3_index='p2_cell'),
        ]

        connect_sites_by_priority(sites, surface, routing_graph)

        for cell_id in cell_ids:
            self.assertIn(cell_id, surface.tower_by_h3,
                          f"Tower missing at {cell_id}")
            self.assertEqual(surface.tower_by_h3[cell_id].source, 'site')

    @patch('mesh_calculator.optimization.hierarchical.find_road_corridor',
           return_value=None)
    def test_site_tower_not_duplicated(self, mock_corridor):
        """place_tower is idempotent — no duplicate tower created."""
        cells = make_cells(['cell_a', 'cell_b'])
        config = MeshConfig()
        surface = MeshSurface(cells, config)
        routing_graph = nx.DiGraph()

        sites = [
            Site(name='A', lat=40.0, lon=44.0, priority=1, h3_index='cell_a'),
            Site(name='B', lat=40.01, lon=44.01, priority=1, h3_index='cell_b'),
        ]

        connect_sites_by_priority(sites, surface, routing_graph)

        # Should have exactly 2 towers, not more
        self.assertEqual(len(surface.towers), 2)
