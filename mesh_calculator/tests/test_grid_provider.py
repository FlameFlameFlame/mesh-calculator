"""Tests for grid-provider bundle metadata and lazy cell access."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h3
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from ..core.config import MeshConfig
from ..core.grid_provider import GridProvider


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


def _boundary_geojson():
    poly = box(43.8, 39.8, 44.2, 40.2)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {},
            }
        ],
    }


def _roads_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[43.82, 39.95], [44.18, 40.05]],
                },
                "properties": {"osm_way_id": 1},
            }
        ],
    }


def test_grid_bundle_metadata_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tif = tmp / "elevation.tif"
        bundle = tmp / "grid_bundle.json"
        _make_dem(tif, width=800, height=800, west=43.7, north=40.3, pixel_deg=0.001)

        payload = GridProvider.build_bundle(
            bundle_path=str(bundle),
            elevation_path=str(tif),
            boundary_geojson=_boundary_geojson(),
            roads_geojson=_roads_geojson(),
            resolutions=(8, 9),
        )

        meta = payload.get("metadata") or {}
        assert meta.get("resolution_set") == [8, 9]
        assert isinstance(meta.get("boundary_sha256"), str) and meta["boundary_sha256"]
        assert isinstance(meta.get("roads_sha256"), str) and meta["roads_sha256"]
        assert meta.get("elevation_file", {}).get("size_bytes", 0) > 0

        with open(bundle) as f:
            on_disk = json.load(f)
        assert on_disk.get("version") >= 1
        assert "metadata" in on_disk

        provider = GridProvider.from_bundle(str(bundle), elevation_path=str(tif))
        try:
            assert provider.available_resolutions() == [8, 9]
            assert provider.bundle_metadata.get("resolution_set") == [8, 9]
            full = provider.get_full_cells(8)
            roads = provider.get_road_cells(8)
            assert full
            assert roads
            assert roads.issubset(full)
        finally:
            provider.close()


def test_lazy_road_cells_lookup_for_unbundled_resolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tif = tmp / "elevation.tif"
        _make_dem(tif, width=500, height=500, west=43.7, north=40.3, pixel_deg=0.001)

        provider = GridProvider.from_inputs(
            elevation_path=str(tif),
            boundary_geojson=_boundary_geojson(),
            roads_geojson=_roads_geojson(),
            resolutions=(8,),
        )
        try:
            roads_at_10 = provider.get_road_cells(10)
            assert isinstance(roads_at_10, set)
            assert roads_at_10
        finally:
            provider.close()


def test_adaptive_full_grid_has_no_parent_child_overlap():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tif = tmp / "elevation.tif"
        _make_dem(tif, width=800, height=800, west=43.7, north=40.3, pixel_deg=0.001)

        provider = GridProvider.from_inputs(
            elevation_path=str(tif),
            boundary_geojson=_boundary_geojson(),
            roads_geojson=_roads_geojson(),
            resolutions=(8, 9, 10, 11),
        )
        try:
            # Force steep terrain so adaptive refinement is exercised.
            provider.get_h3_cell_max_elevation = lambda h3_idx: h3.cell_to_latlng(h3_idx)[0] * 10000.0
            cfg = MeshConfig(h3_resolution=8, auto_refine_h3_on_gradient=True, auto_refine_h3_max_resolution=11)
            adaptive = provider.get_adaptive_full_cells(8, cfg)
            assert adaptive
            for idx in adaptive:
                res = int(h3.get_resolution(idx))
                for parent_res in range(8, res):
                    assert h3.cell_to_parent(idx, parent_res) not in adaptive
        finally:
            provider.close()


def test_adaptive_ladder_and_radius_query_monotonic():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tif = tmp / "elevation.tif"
        _make_dem(tif, width=500, height=500, west=43.7, north=40.3, pixel_deg=0.001)

        provider = GridProvider.from_inputs(
            elevation_path=str(tif),
            boundary_geojson=_boundary_geojson(),
            roads_geojson=_roads_geojson(),
            resolutions=(8, 9, 10, 11),
        )
        try:
            cfg = MeshConfig(h3_resolution=8, auto_refine_h3_on_gradient=True, auto_refine_h3_max_resolution=11)
            assert provider._ladder_target_resolution(120.0, 8, cfg) == 11
            assert provider._ladder_target_resolution(80.0, 8, cfg) == 10
            assert provider._ladder_target_resolution(55.0, 8, cfg) == 9
            assert provider._ladder_target_resolution(10.0, 8, cfg) == 8

            # Use a deterministic steep-elevation proxy to trigger mixed cells.
            provider.get_h3_cell_max_elevation = lambda h3_idx: h3.cell_to_latlng(h3_idx)[0] * 10000.0
            cells = provider.get_adaptive_full_cells(8, cfg)
            center = next(iter(cells))
            near = provider.adaptive_cells_within_radius(center, 1000.0, 8, cfg, candidate_cells=cells)
            far = provider.adaptive_cells_within_radius(center, 2000.0, 8, cfg, candidate_cells=cells)
            assert near
            assert near.issubset(far)
        finally:
            provider.close()
