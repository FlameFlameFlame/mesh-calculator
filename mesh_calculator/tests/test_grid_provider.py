"""Tests for grid-provider bundle metadata and materialization paths."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
        assert on_disk.get("version") == 3
        assert "metadata" in on_disk
        assert "cell_static" in on_disk["resolutions"]["8"]

        provider = GridProvider.from_bundle(str(bundle), elevation_path=str(tif))
        try:
            assert provider.available_resolutions() == [8, 9]
            assert provider.bundle_metadata.get("resolution_set") == [8, 9]
            full = provider.get_full_cells(8)
            roads = provider.get_road_cells(8)
            assert full
            assert roads
            assert roads.issubset(full)
            mat, stats = provider.materialize_cells(
                list(sorted(list(full))[:5]),
                MeshConfig(h3_resolution=8),
                include_stats=True,
            )
            assert len(mat) == 5
            assert stats["from_static"] >= 1
        finally:
            provider.close()


def test_strict_bundle_loader_rejects_old_version():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tif = tmp / "elevation.tif"
        bundle = tmp / "grid_bundle.json"
        _make_dem(tif, width=200, height=200, west=43.7, north=40.3, pixel_deg=0.001)

        payload = {
            "version": 2,
            "elevation_path": str(tif),
            "boundary_geojson": _boundary_geojson(),
            "roads_geojson": _roads_geojson(),
            "resolutions": {"8": {"full_cells": [], "road_cells": []}},
        }
        with open(bundle, "w") as f:
            json.dump(payload, f)

        try:
            GridProvider.from_bundle(str(bundle), elevation_path=str(tif))
            raise AssertionError("Expected strict loader to reject old bundle version")
        except ValueError as exc:
            assert "Unsupported grid bundle version" in str(exc)


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
