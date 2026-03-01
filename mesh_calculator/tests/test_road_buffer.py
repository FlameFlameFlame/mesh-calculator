"""
Tests for road buffer expansion in generate_road_grid().

When config.road_buffer_m > 0, the function expands road cells by sampling at
H3 resolution 10 within the buffer radius and mapping back to the main
resolution.  All expanded cells are returned with has_road=True.

Strategy: call generate_road_grid() directly with a real (tiny) road geometry
and a mock ElevationProvider.  We use a real Shapely LineString + GeoDataFrame
so we exercise the actual expansion code rather than mocking h3 internals.
"""
import unittest
from unittest.mock import MagicMock

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from ..core.config import MeshConfig
from ..core.grid import generate_road_grid, H3Cell


def _make_elevation_provider(elevation=100.0):
    """Return a mock ElevationProvider that always returns a fixed elevation."""
    prov = MagicMock()
    prov.get_elevation.return_value = elevation
    return prov


def _make_roads_gdf(line_coords):
    """Build a one-row GeoDataFrame with a single LineString road."""
    geom = LineString(line_coords)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


def _make_boundary(center_lon, center_lat, half_deg=0.3):
    """Return a square Shapely Polygon large enough to contain test roads."""
    return Polygon([
        (center_lon - half_deg, center_lat - half_deg),
        (center_lon + half_deg, center_lat - half_deg),
        (center_lon + half_deg, center_lat + half_deg),
        (center_lon - half_deg, center_lat + half_deg),
        (center_lon - half_deg, center_lat - half_deg),
    ])


# A small road segment in Armenia (matches the project domain)
_ROAD_LINE = [(44.50, 40.18), (44.52, 40.18)]
_CENTER_LON = 44.51
_CENTER_LAT = 40.18


class TestRoadBufferZero(unittest.TestCase):
    """With road_buffer_m=0, no buffer expansion occurs."""

    def test_no_buffer_returns_road_only_cells(self):
        config = MeshConfig(h3_resolution=8, road_buffer_m=0.0)
        boundary = _make_boundary(_CENTER_LON, _CENTER_LAT)
        roads_gdf = _make_roads_gdf(_ROAD_LINE)
        elev = _make_elevation_provider()

        cells_no_buf = generate_road_grid(boundary, roads_gdf, elev, config)

        # All returned cells must have has_road=True
        for cell in cells_no_buf.values():
            self.assertTrue(
                cell.has_road,
                f"Cell {cell.h3_index} should have has_road=True",
            )

        # Record baseline count for comparison tests
        self._no_buf_count = len(cells_no_buf)

    def test_zero_buffer_cells_all_have_road_flag(self):
        config = MeshConfig(h3_resolution=8, road_buffer_m=0.0)
        boundary = _make_boundary(_CENTER_LON, _CENTER_LAT)
        roads_gdf = _make_roads_gdf(_ROAD_LINE)
        elev = _make_elevation_provider()

        cells = generate_road_grid(boundary, roads_gdf, elev, config)
        road_flags = [c.has_road for c in cells.values()]
        self.assertTrue(all(road_flags))


class TestRoadBufferPositive(unittest.TestCase):
    """With road_buffer_m > 0, more cells are produced and all are road-flagged."""

    def _run(self, buffer_m):
        config = MeshConfig(h3_resolution=8, road_buffer_m=buffer_m)
        boundary = _make_boundary(_CENTER_LON, _CENTER_LAT)
        roads_gdf = _make_roads_gdf(_ROAD_LINE)
        elev = _make_elevation_provider()
        return generate_road_grid(boundary, roads_gdf, elev, config)

    def test_buffer_300_produces_more_cells_than_no_buffer(self):
        config_no_buf = MeshConfig(h3_resolution=8, road_buffer_m=0.0)
        boundary = _make_boundary(_CENTER_LON, _CENTER_LAT)
        roads_gdf = _make_roads_gdf(_ROAD_LINE)
        elev = _make_elevation_provider()

        cells_no_buf = generate_road_grid(
            boundary, roads_gdf, elev, config_no_buf
        )
        cells_with_buf = self._run(300.0)

        self.assertGreater(
            len(cells_with_buf),
            len(cells_no_buf),
            "Buffer should expand the road cell set",
        )

    def test_buffer_cells_have_road_flag_true(self):
        cells = self._run(300.0)
        for cell in cells.values():
            self.assertTrue(
                cell.has_road,
                f"Expanded buffer cell {cell.h3_index} must have has_road=True",
            )

    def test_larger_buffer_produces_more_cells_than_smaller_buffer(self):
        cells_small = self._run(100.0)
        cells_large = self._run(600.0)
        self.assertGreaterEqual(
            len(cells_large),
            len(cells_small),
            "A larger buffer radius should yield at least as many cells",
        )

    def test_buffer_cells_are_H3Cell_instances(self):
        cells = self._run(300.0)
        for cell in cells.values():
            self.assertIsInstance(cell, H3Cell)


if __name__ == '__main__':
    unittest.main()
