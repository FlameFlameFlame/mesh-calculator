"""
Unit tests for rasterized H3 cell anchor-point resolution.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import math

import h3
import numpy as np
import rasterio
from rasterio.transform import from_origin, rowcol
from shapely.geometry import Point, Polygon

from ..core.elevation import ElevationProvider


def _make_dem(path: Path, width: int, height: int, west: float, north: float, pixel_deg: float):
    transform = from_origin(west, north, pixel_deg, pixel_deg)
    data = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(data, 1)
    return transform


def _set_pixel(path: Path, lat: float, lon: float, value: float):
    with rasterio.open(path, "r+") as ds:
        r, c = rowcol(ds.transform, lon, lat)
        if 0 <= r < ds.height and 0 <= c < ds.width:
            band = ds.read(1)
            band[int(r), int(c)] = float(value)
            ds.write(band, 1)


class TestElevationAnchor(unittest.TestCase):
    def test_anchor_point_returns_highest_feasible_pixel(self):
        h3_idx = h3.latlng_to_cell(40.0, 44.0, 10)
        lat_c, lon_c = h3.cell_to_latlng(h3_idx)
        tif_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tif_dir.cleanup)
        tif_path = Path(tif_dir.name) / "dem.tif"
        _make_dem(
            tif_path,
            width=800,
            height=800,
            west=lon_c - 0.02,
            north=lat_c + 0.02,
            pixel_deg=0.00005,
        )
        _set_pixel(tif_path, lat_c, lon_c, 1234.0)

        provider = ElevationProvider(str(tif_path))
        self.addCleanup(provider.close)
        a_lat, a_lon, a_elev = provider.get_h3_cell_anchor_point(h3_idx, margin_m=10.0)

        boundary = h3.cell_to_boundary(h3_idx)
        poly = Polygon([(lon, lat) for lat, lon in boundary])
        self.assertTrue(poly.contains(Point(a_lon, a_lat)))
        self.assertAlmostEqual(a_elev, 1234.0, places=2)

    def test_anchor_margin_falls_back_to_zero_margin(self):
        h3_idx = h3.latlng_to_cell(40.0, 44.0, 11)
        lat_c, lon_c = h3.cell_to_latlng(h3_idx)
        tif_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tif_dir.cleanup)
        tif_path = Path(tif_dir.name) / "dem.tif"
        _make_dem(
            tif_path,
            width=600,
            height=600,
            west=lon_c - 0.01,
            north=lat_c + 0.01,
            pixel_deg=0.00004,
        )
        _set_pixel(tif_path, lat_c, lon_c, 777.0)

        provider = ElevationProvider(str(tif_path))
        self.addCleanup(provider.close)
        huge_margin = provider.get_h3_cell_anchor_point(h3_idx, margin_m=1000.0)
        zero_margin = provider.get_h3_cell_anchor_point(h3_idx, margin_m=0.0)

        self.assertAlmostEqual(huge_margin[2], zero_margin[2], places=2)
        self.assertAlmostEqual(huge_margin[0], zero_margin[0], places=6)
        self.assertAlmostEqual(huge_margin[1], zero_margin[1], places=6)

    def test_anchor_falls_back_to_centroid_when_dem_has_no_overlap(self):
        h3_idx = h3.latlng_to_cell(40.0, 44.0, 10)
        lat_c, lon_c = h3.cell_to_latlng(h3_idx)
        tif_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tif_dir.cleanup)
        tif_path = Path(tif_dir.name) / "dem.tif"
        _make_dem(
            tif_path,
            width=40,
            height=40,
            west=0.0,
            north=1.0,
            pixel_deg=0.0001,
        )

        provider = ElevationProvider(str(tif_path))
        self.addCleanup(provider.close)
        a_lat, a_lon, a_elev = provider.get_h3_cell_anchor_point(h3_idx, margin_m=10.0)

        self.assertAlmostEqual(a_lat, lat_c, places=6)
        self.assertAlmostEqual(a_lon, lon_c, places=6)
        self.assertAlmostEqual(a_elev, 0.0, places=6)

    def test_anchor_cache_hit_avoids_recomputation(self):
        h3_idx = h3.latlng_to_cell(40.0, 44.0, 10)
        lat_c, lon_c = h3.cell_to_latlng(h3_idx)
        tif_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tif_dir.cleanup)
        tif_path = Path(tif_dir.name) / "dem.tif"
        _make_dem(
            tif_path,
            width=600,
            height=600,
            west=lon_c - 0.01,
            north=lat_c + 0.01,
            pixel_deg=0.00004,
        )
        _set_pixel(tif_path, lat_c, lon_c, 500.0)

        provider = ElevationProvider(str(tif_path))
        self.addCleanup(provider.close)
        calls = {"count": 0}
        original = provider._max_pixel_in_polygon

        def _wrapped(poly):
            calls["count"] += 1
            return original(poly)

        provider._max_pixel_in_polygon = _wrapped
        provider.get_h3_cell_anchor_point(h3_idx, margin_m=10.0)
        provider.get_h3_cell_anchor_point(h3_idx, margin_m=10.0)
        self.assertEqual(calls["count"], 1)

    def test_lockless_regression_paths_do_not_fail(self):
        h3_idx = h3.latlng_to_cell(40.0, 44.0, 10)
        lat_c, lon_c = h3.cell_to_latlng(h3_idx)
        neighbor = next(nb for nb in h3.grid_disk(h3_idx, 1) if nb != h3_idx)
        n_lat, n_lon = h3.cell_to_latlng(neighbor)

        tif_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tif_dir.cleanup)
        tif_path = Path(tif_dir.name) / "dem.tif"
        _make_dem(
            tif_path,
            width=900,
            height=900,
            west=min(lon_c, n_lon) - 0.02,
            north=max(lat_c, n_lat) + 0.02,
            pixel_deg=0.00005,
        )
        _set_pixel(tif_path, lat_c, lon_c, 400.0)
        _set_pixel(tif_path, n_lat, n_lon, 420.0)

        provider = ElevationProvider(str(tif_path))
        self.addCleanup(provider.close)

        # Regression guard: methods must still work if lock attribute is absent.
        delattr(provider, "_lock")

        max_elev = provider.get_h3_cell_max_elevation(h3_idx)
        anchor = provider.get_h3_cell_anchor_point(h3_idx, margin_m=10.0)
        peak = provider.get_line_peak_elevation(lat_c, lon_c, n_lat, n_lon)

        self.assertTrue(math.isfinite(max_elev))
        self.assertEqual(len(anchor), 3)
        self.assertTrue(all(math.isfinite(float(v)) for v in anchor))
        self.assertEqual(len(peak), 4)
        self.assertTrue(all(math.isfinite(float(v)) for v in peak))


if __name__ == "__main__":
    unittest.main()
