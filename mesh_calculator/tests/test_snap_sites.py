"""
Tests for Fix #8: Snap sites to nearest road cell.

Sites loaded from GeoJSON get h3_index via h3.latlng_to_cell which may not
be a road cell. snap_sites_to_roads must find the nearest road cell.
"""
import unittest
from unittest.mock import patch

from mesh_calculator.data.sites import Site, snap_sites_to_roads
from mesh_calculator.core.grid import H3Cell


def make_road_cells(cell_specs):
    """Create cells dict from a list of (h3_index, lat, lon, elevation) tuples."""
    cells = {}
    for h3_idx, lat, lon, elev in cell_specs:
        cells[h3_idx] = H3Cell(
            h3_index=h3_idx, lat=lat, lon=lon,
            elevation=elev, has_road=True, is_in_boundary=True,
        )
    return cells


class TestSnapSitesToRoads(unittest.TestCase):
    """Fix #8: Sites must be snapped to nearest road cell."""

    def test_site_already_on_road_unchanged(self):
        """If site's h3_index is already a road cell, it stays unchanged."""
        cells = make_road_cells([
            ('cell_A', 40.0, 44.0, 100.0),
            ('cell_B', 40.01, 44.01, 200.0),
        ])
        site = Site(name='Yerevan', lat=40.0, lon=44.0, priority=1,
                    h3_index='cell_A')

        snap_sites_to_roads([site], cells)

        self.assertEqual(site.h3_index, 'cell_A')

    def test_site_off_road_snapped_to_nearest(self):
        """Site not on a road cell gets snapped to nearest road cell."""
        cells = make_road_cells([
            ('road_1', 40.0, 44.0, 100.0),   # closer to site
            ('road_2', 41.0, 45.0, 200.0),   # farther
        ])
        site = Site(name='Village', lat=40.001, lon=44.001, priority=2,
                    h3_index='off_road_cell')

        snap_sites_to_roads([site], cells)

        self.assertEqual(site.h3_index, 'road_1')

    def test_snaps_to_closest_among_multiple(self):
        """Among multiple road cells, site snaps to the closest one."""
        cells = make_road_cells([
            ('far_cell', 41.0, 45.0, 100.0),
            ('mid_cell', 40.05, 44.05, 100.0),
            ('near_cell', 40.001, 44.001, 100.0),
        ])
        site = Site(name='Town', lat=40.0, lon=44.0, priority=1,
                    h3_index='not_a_road')

        snap_sites_to_roads([site], cells)

        self.assertEqual(site.h3_index, 'near_cell')

    def test_no_road_cells_leaves_unchanged(self):
        """If no road cells exist, log warning and leave h3_index as-is."""
        cells = {}
        site = Site(name='Isolated', lat=40.0, lon=44.0, priority=1,
                    h3_index='original')

        snap_sites_to_roads([site], cells)

        self.assertEqual(site.h3_index, 'original')
