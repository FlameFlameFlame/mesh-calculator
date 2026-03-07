"""Tests for grid-provider bundle metadata and materialization paths."""
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
    # Keep test geometry small to reduce H3 materialization cost.
    poly = box(43.94, 40.00, 44.06, 40.12)
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
                    "coordinates": [[43.95, 40.02], [44.05, 40.10]],
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
        _make_dem(tif, width=180, height=180, west=43.90, north=40.18, pixel_deg=0.001)

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
        _make_dem(tif, width=100, height=100, west=43.90, north=40.18, pixel_deg=0.001)

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
        _make_dem(tif, width=150, height=150, west=43.90, north=40.18, pixel_deg=0.001)

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


def test_adaptive_small_buffer_includes_touching_neighbors():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tif = tmp / "elevation.tif"
        _make_dem(tif, width=150, height=150, west=43.90, north=40.18, pixel_deg=0.001)

        provider = GridProvider.from_inputs(
            elevation_path=str(tif),
            boundary_geojson=_boundary_geojson(),
            roads_geojson=_roads_geojson(),
            resolutions=(8, 9),
        )
        try:
            cfg = MeshConfig(h3_resolution=8, auto_refine_h3_on_gradient=False)
            full = provider.get_adaptive_full_cells(8, cfg)
            center = None
            for h3_idx in full:
                ring1 = set(h3.grid_disk(h3_idx, 1))
                if ring1.issubset(full):
                    center = h3_idx
                    break
            assert center is not None

            nearby = provider.adaptive_cells_within_radius(
                center,
                100.0,
                8,
                cfg,
                candidate_cells=full,
            )
            ring1_in_full = set(h3.grid_disk(center, 1)) & full
            assert center in nearby
            assert len((ring1_in_full - {center}) & nearby) >= 1
        finally:
            provider.close()
