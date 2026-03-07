"""
Tests for Fix #11: Place tower at every site location.
Tests for Fix #7: update_visibility_edges wiring.

Every site must have a tower placed at its h3_index with source='site',
even if the site is a lone P1 site or all corridor searches fail.
After tower placement, update_visibility_edges must compute LOS edges.
"""
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import networkx as nx

from mesh_calculator.core.config import MeshConfig
from mesh_calculator.core.grid import H3Cell
from mesh_calculator.data.sites import Site
from mesh_calculator.data.cache import LOSCache, LOSResult
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


# ---------- Fix #7: Visibility Edges ----------

class TestVisibilityEdgesWiring(unittest.TestCase):
    """Fix #7: update_visibility_edges adds edges between LOS-connected towers."""

    @patch('mesh_calculator.network.graph.compute_los')
    def test_edges_added_between_visible_towers(self, mock_compute_los):
        """After placing towers, update_visibility_edges adds edges."""
        cells = make_cells(['a', 'b', 'c'])
        config = MeshConfig()
        surface = MeshSurface(cells, config)

        surface.place_tower('a', source='site')
        surface.place_tower('b', source='site')
        surface.place_tower('c', source='site')

        mock_compute_los.return_value = LOSResult(
            clearance_m=15.0, path_loss_db=120.0,
            distance_m=5000.0, is_visible=True,
        )

        surface.update_visibility_edges()

        self.assertGreater(
            surface.visibility_graph.edge_count(), 0,
            "Visibility graph should have edges after update")

    @patch('mesh_calculator.network.graph.compute_los')
    def test_clusters_reflect_connectivity(self, mock_compute_los):
        """Connected towers form clusters, not singletons."""
        cells = make_cells(['a', 'b', 'c'])
        config = MeshConfig()
        surface = MeshSurface(cells, config)

        surface.place_tower('a', source='site')
        surface.place_tower('b', source='site')
        surface.place_tower('c', source='site')

        mock_compute_los.return_value = LOSResult(
            clearance_m=15.0, path_loss_db=120.0,
            distance_m=5000.0, is_visible=True,
        )

        surface.update_visibility_edges()

        clusters = surface.get_tower_clusters()
        num_clusters = len(set(clusters.values()))
        self.assertEqual(num_clusters, 1,
                         "All 3 towers with mutual LOS should form 1 cluster")

    @patch('mesh_calculator.network.graph.compute_los')
    def test_edges_store_clearance_and_path_loss(self, mock_compute_los):
        """Visibility edges include clearance_m and path_loss_db."""
        cells = make_cells(['a', 'b'])
        config = MeshConfig()
        surface = MeshSurface(cells, config)

        surface.place_tower('a', source='site')
        surface.place_tower('b', source='site')

        mock_compute_los.return_value = LOSResult(
            clearance_m=22.5, path_loss_db=115.3,
            distance_m=8000.0, is_visible=True,
        )

        surface.update_visibility_edges()

        edges = list(surface.visibility_graph.graph.edges(data=True))
        self.assertEqual(len(edges), 1)
        data = edges[0][2]
        self.assertAlmostEqual(data['clearance_m'], 22.5)
        self.assertAlmostEqual(data['path_loss_db'], 115.3)
        self.assertAlmostEqual(data['distance_m'], 8000.0)

    @patch('mesh_calculator.network.graph.compute_los')
    def test_fspl_radius_cap_prunes_impossible_pairs(self, mock_compute_los):
        """KD-tree candidate radius is capped by FSPL budget bound."""
        cells = {
            'a': H3Cell('a', lat=40.0, lon=44.0, elevation=100.0, has_road=True, is_in_boundary=True),
            # ~70 km away: outside FSPL cap for very low link budget.
            'b': H3Cell('b', lat=40.63, lon=44.0, elevation=100.0, has_road=True, is_in_boundary=True),
        }
        config = MeshConfig(
            tx_power_mw=1.0,
            antenna_gain_dbi=0.0,
            receiver_sensitivity_dbm=-90.0,
        )
        surface = MeshSurface(cells, config)
        surface.place_tower('a', source='site')
        surface.place_tower('b', source='site')

        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=80.0, distance_m=1000.0, is_visible=True
        )

        with patch.object(MeshConfig, "max_visibility_m", new_callable=PropertyMock, return_value=200000.0):
            surface.update_visibility_edges()

        self.assertEqual(surface.visibility_graph.edge_count(), 0)
        self.assertFalse(mock_compute_los.called)


# ---------- Cell Coverage ----------

class TestCellCoverage(unittest.TestCase):
    """compute_cell_coverage populates visible_tower_count and related fields."""

    @patch('mesh_calculator.network.graph.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_visible_tower_count_populated(self, mock_distance, mock_compute_los):
        """Cells near visible towers get visible_tower_count > 0."""
        cells = make_cells(['t1', 't2', 'c1'])
        config = MeshConfig()
        surface = MeshSurface(cells, config)

        surface.place_tower('t1', source='site')
        surface.place_tower('t2', source='site')

        mock_distance.return_value = 5000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=110.0,
            distance_m=5000.0, is_visible=True,
        )

        surface.compute_cell_coverage()

        # Tower cells see themselves + the other tower
        self.assertEqual(cells['t1'].visible_tower_count, 2)
        self.assertEqual(cells['t2'].visible_tower_count, 2)
        # Non-tower cell sees both towers
        self.assertEqual(cells['c1'].visible_tower_count, 2)
        self.assertAlmostEqual(cells['c1'].distance_to_closest_tower, 5000.0)
        self.assertAlmostEqual(cells['c1'].path_loss, 110.0)
        self.assertAlmostEqual(cells['c1'].clearance, 10.0)

    @patch('mesh_calculator.network.graph.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_no_los_leaves_zero(self, mock_distance, mock_compute_los):
        """Cells with no LOS to any tower keep visible_tower_count=0."""
        cells = make_cells(['t1', 'c1'])
        config = MeshConfig()
        surface = MeshSurface(cells, config)

        surface.place_tower('t1', source='site')

        mock_distance.return_value = 5000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=-5.0, path_loss_db=999.0,
            distance_m=5000.0, is_visible=False,
        )

        surface.compute_cell_coverage()

        # Tower cell sees itself (distance 0, no LOS check needed)
        self.assertEqual(cells['t1'].visible_tower_count, 1)
        # Non-tower cell has no LOS
        self.assertEqual(cells['c1'].visible_tower_count, 0)


class TestHierarchicalNestedLOSWorkers(unittest.TestCase):
    """Hierarchical outer thread pools should force inner LOS workers to 1."""

    @patch('mesh_calculator.optimization.hierarchical.wire_corridor_edges')
    @patch('mesh_calculator.optimization.hierarchical.place_nodes_along_corridor')
    @patch('mesh_calculator.optimization.hierarchical.find_road_corridor')
    def test_priority2_parallel_corridor_calls_use_single_inner_worker(
        self,
        mock_find_corridor,
        mock_place_nodes,
        mock_wire,
    ):
        cells = make_cells(['p1_cell', 'p2_cell'])
        config = MeshConfig(los_parallel_workers=8)
        surface = MeshSurface(cells, config)
        routing_graph = nx.DiGraph()
        sites = [
            Site(name='P1', lat=40.0, lon=44.0, priority=1, h3_index='p1_cell'),
            Site(name='P2', lat=40.01, lon=44.01, priority=2, h3_index='p2_cell'),
        ]

        mock_find_corridor.return_value = ['p2_cell', 'p1_cell']
        mock_place_nodes.return_value = ['p2_cell', 'p1_cell']
        mock_wire.return_value = None

        connect_sites_by_priority(sites, surface, routing_graph)

        self.assertGreaterEqual(mock_place_nodes.call_count, 1)
        for call in mock_place_nodes.call_args_list:
            self.assertEqual(call.kwargs.get('los_max_workers'), 1)

    @patch('mesh_calculator.optimization.hierarchical.wire_corridor_edges')
    @patch('mesh_calculator.optimization.hierarchical.place_nodes_along_corridor')
    @patch('mesh_calculator.optimization.hierarchical.find_road_corridor')
    def test_priority1_mesh_calls_use_single_inner_worker(
        self,
        mock_find_corridor,
        mock_place_nodes,
        mock_wire,
    ):
        cells = make_cells(['a', 'b'])
        config = MeshConfig(los_parallel_workers=8)
        surface = MeshSurface(cells, config)
        routing_graph = nx.DiGraph()
        sites = [
            Site(name='A', lat=40.0, lon=44.0, priority=1, h3_index='a'),
            Site(name='B', lat=40.01, lon=44.01, priority=1, h3_index='b'),
        ]

        mock_find_corridor.return_value = ['a', 'b']
        mock_place_nodes.return_value = ['a', 'b']
        mock_wire.return_value = None

        connect_sites_by_priority(sites, surface, routing_graph)

        self.assertGreaterEqual(mock_place_nodes.call_count, 1)
        for call in mock_place_nodes.call_args_list:
            self.assertEqual(call.kwargs.get('los_max_workers'), 1)
