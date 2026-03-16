"""
Grid provider abstraction for route optimization and runtime coverage.

This module centralizes grid/elevation access so planners can be resolution-agnostic
and consume prebuilt grid bundles instead of rebuilding grid internals ad hoc.
"""
from __future__ import annotations

import json
import os
import hashlib
import math
from functools import lru_cache
from typing import Dict, Iterable, Optional

import geopandas as gpd
import h3
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, shape as shapely_shape

import structlog

from .config import MeshConfig, RouteSpec
from .elevation import ElevationProvider
from .geometry import great_circle_distance
from .grid import (
    H3Cell,
    find_h3_cells_on_roads,
    resolve_cell_profile,
    shapely_to_h3_cells,
)
from .road_corridor import road_geojson_to_h3_corridor

logger = structlog.get_logger(__name__)

_DEFAULT_BUNDLE_RESOLUTIONS = (8, 9)
_GRID_BUNDLE_VERSION = 3
_SUPPORTED_BUNDLE_VERSIONS = {3}
_EARTH_R = 6_371_000.0


@lru_cache(maxsize=32)
def _cell_edge_length_m(resolution: int) -> float:
    """Cached average H3 edge length for a resolution."""
    return float(h3.average_hexagon_edge_length(int(resolution), unit="m"))


def _boundary_polygon_from_geojson(boundary_geojson: Optional[dict]) -> Optional[Polygon]:
    """Extract merged polygon geometry from boundary GeoJSON."""
    if not boundary_geojson:
        return None
    try:
        if boundary_geojson.get("type") == "FeatureCollection":
            geoms = [
                shapely_shape(f["geometry"])
                for f in boundary_geojson.get("features", [])
                if f.get("geometry")
            ]
            if not geoms:
                return None
            poly = geoms[0]
            for g in geoms[1:]:
                poly = poly.union(g)
            return poly
        if boundary_geojson.get("type") == "Feature":
            geom = boundary_geojson.get("geometry")
            return shapely_shape(geom) if geom else None
        if boundary_geojson.get("type"):
            return shapely_shape(boundary_geojson)
    except Exception:
        logger.warning("Failed to parse boundary geometry", exc_info=True)
        return None
    return None


def _normalize_resolutions(resolutions: Iterable[int]) -> list[int]:
    vals = sorted({int(r) for r in resolutions})
    return [r for r in vals if 0 <= r <= 15]


def _stable_json_sha256(payload: Optional[dict]) -> str:
    """Return deterministic SHA-256 for JSON payloads."""
    if payload is None:
        return ""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _file_metadata(path: str) -> dict:
    """Return coarse source-file metadata for bundle invalidation checks."""
    try:
        st = os.stat(path)
        return {
            "size_bytes": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    except OSError:
        return {
            "size_bytes": 0,
            "mtime_ns": 0,
        }


def _to_xyz(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Project lat/lon arrays to approximate ECEF xyz for KD-tree range search."""
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    cos_lat = np.cos(lats_rad)
    return np.column_stack([
        cos_lat * np.cos(lons_rad),
        cos_lat * np.sin(lons_rad),
        np.sin(lats_rad),
    ]) * _EARTH_R


class GridProvider:
    """
    Multi-resolution grid/elevation provider.

    Responsibilities:
    - Bundle build/load for resolutions 8..9 (or configured set)
    - H3 full/road cell lookup by resolution
    - Radius-to-ring conversions and disk expansion helpers
    - Terrain/elevation proxy methods used by LOS calculations
    - On-demand H3Cell materialization with cached cell profiles
    """

    def __init__(
        self,
        *,
        elevation_path: str,
        boundary_geojson: dict,
        roads_geojson: Optional[dict] = None,
        full_cells_by_res: Optional[Dict[int, set[str]]] = None,
        road_cells_by_res: Optional[Dict[int, set[str]]] = None,
        cell_static_by_res: Optional[Dict[int, dict[str, dict[str, float]]]] = None,
        bundle_path: Optional[str] = None,
    ):
        if not elevation_path or not os.path.isfile(elevation_path):
            raise ValueError("GridProvider requires a valid elevation_path")
        self.elevation_path = os.path.abspath(elevation_path)
        self.boundary_geojson = boundary_geojson
        self.roads_geojson = roads_geojson or {"type": "FeatureCollection", "features": []}
        self.boundary_polygon = _boundary_polygon_from_geojson(boundary_geojson)
        if self.boundary_polygon is None:
            raise ValueError("GridProvider requires valid boundary geometry")
        self.bundle_path = bundle_path
        self.bundle_metadata: dict = {}

        self._elevation_provider = ElevationProvider(self.elevation_path)
        self._full_cells_by_res: Dict[int, set[str]] = full_cells_by_res or {}
        self._road_cells_by_res: Dict[int, set[str]] = road_cells_by_res or {}
        self._cell_static_by_res: Dict[int, dict[str, dict[str, float]]] = cell_static_by_res or {}
        self._cell_cache_by_res: Dict[int, Dict[str, H3Cell]] = {}
        self._adaptive_mesh_cache: Dict[tuple, dict] = {}
        self._roads_gdf_cache = None

    @classmethod
    def from_bundle(cls, bundle_path: str, elevation_path: Optional[str] = None) -> "GridProvider":
        """Load a provider from a persisted grid bundle JSON."""
        with open(bundle_path) as f:
            payload = json.load(f)
        version = int(payload.get("version", 0))
        if version not in _SUPPORTED_BUNDLE_VERSIONS:
            raise ValueError(
                f"Unsupported grid bundle version {version}; expected one of "
                f"{sorted(_SUPPORTED_BUNDLE_VERSIONS)}"
            )
        full_cells_by_res = {
            int(res): set(vals.get("full_cells", []))
            for res, vals in (payload.get("resolutions") or {}).items()
        }
        road_cells_by_res = {
            int(res): set(vals.get("road_cells", []))
            for res, vals in (payload.get("resolutions") or {}).items()
        }
        missing_static = [
            res for res, vals in (payload.get("resolutions") or {}).items()
            if "cell_static" not in (vals or {})
        ]
        if missing_static:
            raise ValueError(
                "Grid bundle is missing required cell_static attributes for resolutions: "
                + ", ".join(str(r) for r in missing_static)
            )
        cell_static_by_res = {
            int(res): vals.get("cell_static", {})
            for res, vals in (payload.get("resolutions") or {}).items()
        }
        elev_path = elevation_path or payload.get("elevation_path")
        if elev_path and not os.path.isabs(elev_path):
            elev_path = os.path.join(os.path.dirname(os.path.abspath(bundle_path)), elev_path)
        provider = cls(
            elevation_path=elev_path,
            boundary_geojson=payload.get("boundary_geojson"),
            roads_geojson=payload.get("roads_geojson"),
            full_cells_by_res=full_cells_by_res,
            road_cells_by_res=road_cells_by_res,
            cell_static_by_res=cell_static_by_res,
            bundle_path=os.path.abspath(bundle_path),
        )
        provider.bundle_metadata = payload.get("metadata") or {}
        return provider

    @classmethod
    def build_bundle(
        cls,
        *,
        bundle_path: str,
        elevation_path: str,
        boundary_geojson: dict,
        roads_geojson: Optional[dict] = None,
        resolutions: Iterable[int] = _DEFAULT_BUNDLE_RESOLUTIONS,
    ) -> dict:
        """
        Build and persist a multi-resolution grid bundle.

        Bundle stores:
        - boundary/roads GeoJSON payloads
        - full and road H3 cell sets for each resolution
        - source elevation path (relative to bundle dir when possible)
        """
        boundary_poly = _boundary_polygon_from_geojson(boundary_geojson)
        if boundary_poly is None:
            raise ValueError("Cannot build grid bundle without valid boundary geometry")

        res_list = _normalize_resolutions(resolutions)
        if not res_list:
            raise ValueError("No valid resolutions provided for grid bundle")

        roads_fc = roads_geojson or {"type": "FeatureCollection", "features": []}
        roads_gdf = None
        if roads_fc.get("features"):
            try:
                roads_gdf = gpd.GeoDataFrame.from_features(roads_fc["features"], crs="EPSG:4326")
            except Exception:
                logger.warning("Failed to build roads GeoDataFrame for bundle", exc_info=True)
                roads_gdf = None

        elev_provider = ElevationProvider(elevation_path)
        by_res = {}
        for res in res_list:
            full_cells = set(shapely_to_h3_cells(boundary_poly, res))
            road_cells = set()
            if roads_gdf is not None and len(roads_gdf):
                road_cells = set(find_h3_cells_on_roads(roads_gdf, res)) & full_cells
            static_attrs: dict[str, dict[str, float]] = {}
            for h3_idx in full_cells:
                lat, lon = h3.cell_to_latlng(h3_idx)
                elev, los_lat, los_lon = resolve_cell_profile(
                    elev_provider,
                    h3_idx,
                    lat,
                    lon,
                )
                static_attrs[h3_idx] = {
                    "elevation_max_m": float(elev),
                    "los_anchor_lat": float(los_lat),
                    "los_anchor_lon": float(los_lon),
                    "h3_resolution": int(h3.get_resolution(h3_idx)),
                }
            by_res[str(res)] = {
                "full_cells": sorted(full_cells),
                "road_cells": sorted(road_cells),
                "cell_static": static_attrs,
            }
            logger.info(
                "Grid bundle resolution built",
                resolution=res,
                full_cells=len(full_cells),
                road_cells=len(road_cells),
            )
        elev_provider.close()

        bundle_dir = os.path.dirname(os.path.abspath(bundle_path))
        os.makedirs(bundle_dir, exist_ok=True)

        elev_rel = elevation_path
        try:
            elev_rel = os.path.relpath(os.path.abspath(elevation_path), bundle_dir)
        except Exception:
            pass
        payload = {
            "version": _GRID_BUNDLE_VERSION,
            "elevation_path": elev_rel,
            "boundary_geojson": boundary_geojson,
            "roads_geojson": roads_fc,
            "resolutions": by_res,
            "metadata": {
                "resolution_set": res_list,
                "boundary_sha256": _stable_json_sha256(boundary_geojson),
                "roads_sha256": _stable_json_sha256(roads_fc),
                "elevation_path_abs": os.path.abspath(elevation_path),
                "elevation_file": _file_metadata(elevation_path),
            },
        }
        with open(bundle_path, "w") as f:
            json.dump(payload, f)
        logger.info(
            "Grid bundle saved",
            path=bundle_path,
            resolutions=res_list,
        )
        return payload

    @classmethod
    def from_inputs(
        cls,
        *,
        elevation_path: str,
        boundary_geojson: dict,
        roads_geojson: Optional[dict] = None,
        resolutions: Iterable[int] = _DEFAULT_BUNDLE_RESOLUTIONS,
    ) -> "GridProvider":
        """
        Build in-memory provider from raw inputs (without writing bundle).
        """
        res_list = _normalize_resolutions(resolutions)
        boundary_poly = _boundary_polygon_from_geojson(boundary_geojson)
        if boundary_poly is None:
            raise ValueError("Cannot initialize GridProvider without valid boundary geometry")
        roads_fc = roads_geojson or {"type": "FeatureCollection", "features": []}
        roads_gdf = None
        if roads_fc.get("features"):
            roads_gdf = gpd.GeoDataFrame.from_features(roads_fc["features"], crs="EPSG:4326")
        full_cells_by_res: Dict[int, set[str]] = {}
        road_cells_by_res: Dict[int, set[str]] = {}
        cell_static_by_res: Dict[int, dict[str, dict[str, float]]] = {}
        elev_provider = ElevationProvider(elevation_path)
        for res in res_list:
            full = set(shapely_to_h3_cells(boundary_poly, res))
            full_cells_by_res[res] = full
            if roads_gdf is not None and len(roads_gdf):
                road_cells_by_res[res] = set(find_h3_cells_on_roads(roads_gdf, res)) & full
            else:
                road_cells_by_res[res] = set()
            static_attrs: dict[str, dict[str, float]] = {}
            for h3_idx in full:
                lat, lon = h3.cell_to_latlng(h3_idx)
                elev, los_lat, los_lon = resolve_cell_profile(
                    elev_provider,
                    h3_idx,
                    lat,
                    lon,
                )
                static_attrs[h3_idx] = {
                    "elevation_max_m": float(elev),
                    "los_anchor_lat": float(los_lat),
                    "los_anchor_lon": float(los_lon),
                    "h3_resolution": int(h3.get_resolution(h3_idx)),
                }
            cell_static_by_res[res] = static_attrs
        elev_provider.close()
        return cls(
            elevation_path=elevation_path,
            boundary_geojson=boundary_geojson,
            roads_geojson=roads_fc,
            full_cells_by_res=full_cells_by_res,
            road_cells_by_res=road_cells_by_res,
            cell_static_by_res=cell_static_by_res,
        )

    def close(self) -> None:
        self._elevation_provider.close()

    def __enter__(self) -> "GridProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- Elevation/LOS proxy methods ---
    def get_elevation(self, lat: float, lon: float) -> float:
        return self._elevation_provider.get_elevation(lat, lon)

    def get_elevation_bilinear(self, lat: float, lon: float) -> float:
        return self._elevation_provider.get_elevation_bilinear(lat, lon)

    def get_elevation_bulk(self, coords):
        return self._elevation_provider.get_elevation_bulk(coords)

    def get_elevation_bilinear_bulk(self, coords):
        return self._elevation_provider.get_elevation_bilinear_bulk(coords)

    def get_h3_cell_max_elevation(self, h3_index: str) -> float:
        return self._elevation_provider.get_h3_cell_max_elevation(h3_index)

    def get_h3_cell_anchor_point(
        self, h3_index: str, margin_m: float = 10.0
    ) -> tuple[float, float, float]:
        return self._elevation_provider.get_h3_cell_anchor_point(h3_index, margin_m=margin_m)

    def get_line_peak_elevation(
        self, src_lat: float, src_lon: float, dst_lat: float, dst_lon: float
    ) -> tuple[float, float, float, float]:
        return self._elevation_provider.get_line_peak_elevation(src_lat, src_lon, dst_lat, dst_lon)

    def get_line_peak_elevation_batch(
        self,
        lines: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        return self._elevation_provider.get_line_peak_elevation_batch(lines)

    def cache_stats(self) -> dict:
        return self._elevation_provider.cache_stats()

    # --- Grid lookup helpers ---
    def available_resolutions(self) -> list[int]:
        keys = set(self._full_cells_by_res.keys()) | set(self._road_cells_by_res.keys())
        return sorted(int(k) for k in keys)

    def _get_roads_gdf(self):
        if self._roads_gdf_cache is None:
            try:
                self._roads_gdf_cache = gpd.GeoDataFrame.from_features(
                    self.roads_geojson.get("features", []),
                    crs="EPSG:4326",
                )
            except Exception:
                self._roads_gdf_cache = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return self._roads_gdf_cache

    def get_full_cells(self, resolution: int) -> set[str]:
        res = int(resolution)
        if res not in self._full_cells_by_res:
            self._full_cells_by_res[res] = set(shapely_to_h3_cells(self.boundary_polygon, res))
        return self._full_cells_by_res[res]

    def get_road_cells(self, resolution: int) -> set[str]:
        res = int(resolution)
        if res not in self._road_cells_by_res:
            roads_gdf = self._get_roads_gdf()
            if len(roads_gdf):
                self._road_cells_by_res[res] = (
                    set(find_h3_cells_on_roads(roads_gdf, res))
                    & set(self.get_full_cells(res))
                )
            else:
                self._road_cells_by_res[res] = set()
        return self._road_cells_by_res[res]

    def corridor_from_features(
        self,
        features: list,
        resolution: int,
        site1: Optional[dict] = None,
        site2: Optional[dict] = None,
        config: Optional[MeshConfig] = None,
    ) -> list[str]:
        if config is not None and bool(getattr(config, "auto_refine_h3_on_gradient", False)):
            return self.adaptive_corridor_from_features(
                features,
                int(resolution),
                config,
                site1=site1,
                site2=site2,
            )
        return road_geojson_to_h3_corridor(features, int(resolution), site1=site1, site2=site2)

    def radius_m_to_ring(self, radius_m: float, resolution: int, minimum_one: bool = False) -> int:
        edge_m = h3.average_hexagon_edge_length(int(resolution), unit="m")
        rings = int(round(max(float(radius_m), 0.0) / edge_m)) if edge_m > 0 else 0
        if minimum_one and radius_m > 0:
            rings = max(1, rings)
        return max(0, rings)

    def expand_disk(self, h3_index: str, rings: int) -> set[str]:
        return set(h3.grid_disk(h3_index, int(rings)))

    def _adaptive_cache_key(self, base_resolution: int, config: MeshConfig) -> tuple:
        return (
            int(base_resolution),
            bool(getattr(config, "auto_refine_h3_on_gradient", False)),
            int(getattr(config, "auto_refine_h3_max_resolution", 9)),
        )

    def _ladder_target_resolution(self, gradient_m_per_km: float, base_resolution: int, config: MeshConfig) -> int:
        target = int(base_resolution)
        if gradient_m_per_km > 75.0:
            target = 9
        target = max(int(base_resolution), target)
        target = min(target, int(getattr(config, "auto_refine_h3_max_resolution", 9)))
        return target

    def _base_cell_local_gradient(self, base_h3: str, base_full: set[str]) -> float:
        """Max ring-1 slope (m/km) around a base-resolution H3 cell."""
        lat_a, lon_a = h3.cell_to_latlng(base_h3)
        elev_a = self.get_h3_cell_max_elevation(base_h3)
        max_grad = 0.0
        for nb in h3.grid_disk(base_h3, 1):
            if nb == base_h3 or nb not in base_full:
                continue
            lat_b, lon_b = h3.cell_to_latlng(nb)
            dist_m = great_circle_distance(lat_a, lon_a, lat_b, lon_b)
            if dist_m <= 1.0:
                continue
            elev_b = self.get_h3_cell_max_elevation(nb)
            grad = abs(elev_b - elev_a) / (dist_m / 1000.0)
            if grad > max_grad:
                max_grad = grad
        return float(max_grad)

    def _build_adaptive_mesh(self, base_resolution: int, config: MeshConfig) -> dict:
        """Build a non-overlapping mixed-resolution partition for the whole boundary."""
        key = self._adaptive_cache_key(base_resolution, config)
        cached = self._adaptive_mesh_cache.get(key)
        if cached is not None:
            return cached

        base_res = int(base_resolution)
        base_full = set(self.get_full_cells(base_res))
        base_road = set(self.get_road_cells(base_res)) & base_full

        full_cells_out: set[str] = set()
        road_cells_out: set[str] = set()
        cell_meta: dict[str, dict] = {}
        by_res_full: dict[int, set[str]] = {}
        by_res_road: dict[int, set[str]] = {}
        cell_lats: list[float] = []
        cell_lons: list[float] = []
        cell_ids: list[str] = []
        cell_radii: list[float] = []

        if not bool(getattr(config, "auto_refine_h3_on_gradient", False)):
            for h3_idx in base_full:
                full_cells_out.add(h3_idx)
                if h3_idx in base_road:
                    road_cells_out.add(h3_idx)
                cell_meta[h3_idx] = {
                    "base_h3_resolution": base_res,
                    "target_h3_resolution": base_res,
                    "gradient_m_per_km": 0.0,
                    "adaptive_refined": False,
                }
        else:
            full_by_res_cache: dict[int, set[str]] = {base_res: base_full}
            road_by_res_cache: dict[int, set[str]] = {base_res: base_road}
            for parent in base_full:
                gradient = self._base_cell_local_gradient(parent, base_full)
                target_res = self._ladder_target_resolution(gradient, base_res, config)
                if target_res == base_res:
                    full_cells_out.add(parent)
                    if parent in base_road:
                        road_cells_out.add(parent)
                    cell_meta[parent] = {
                        "base_h3_resolution": base_res,
                        "target_h3_resolution": target_res,
                        "gradient_m_per_km": gradient,
                        "adaptive_refined": False,
                    }
                    continue

                target_full = full_by_res_cache.get(target_res)
                if target_full is None:
                    target_full = set(self.get_full_cells(target_res))
                    full_by_res_cache[target_res] = target_full
                target_road = road_by_res_cache.get(target_res)
                if target_road is None:
                    target_road = set(self.get_road_cells(target_res))
                    road_by_res_cache[target_res] = target_road

                for child in h3.cell_to_children(parent, target_res):
                    if child not in target_full:
                        continue
                    full_cells_out.add(child)
                    if child in target_road:
                        road_cells_out.add(child)
                    cell_meta[child] = {
                        "base_h3_resolution": base_res,
                        "target_h3_resolution": target_res,
                        "gradient_m_per_km": gradient,
                        "adaptive_refined": True,
                    }

        for h3_idx in full_cells_out:
            res = int(h3.get_resolution(h3_idx))
            by_res_full.setdefault(res, set()).add(h3_idx)
            if h3_idx in road_cells_out:
                by_res_road.setdefault(res, set()).add(h3_idx)
            lat, lon = h3.cell_to_latlng(h3_idx)
            cell_lats.append(lat)
            cell_lons.append(lon)
            cell_ids.append(h3_idx)
            cell_radii.append(_cell_edge_length_m(res))

        kdtree = None
        xyz = None
        cell_radii_arr = None
        max_cell_radius_m = 0.0
        if cell_ids:
            xyz = _to_xyz(np.asarray(cell_lats), np.asarray(cell_lons))
            kdtree = cKDTree(xyz)
            cell_radii_arr = np.asarray(cell_radii, dtype=float)
            max_cell_radius_m = float(np.max(cell_radii_arr))

        counts: dict[int, int] = {}
        for h3_idx in full_cells_out:
            res = int(h3.get_resolution(h3_idx))
            counts[res] = counts.get(res, 0) + 1

        mesh = {
            "base_resolution": base_res,
            "full_cells": full_cells_out,
            "road_cells": road_cells_out,
            "full_by_res": by_res_full,
            "road_by_res": by_res_road,
            "cell_meta": cell_meta,
            "cells_by_resolution": counts,
            "effective_h3_resolution_min": min(counts.keys()) if counts else base_res,
            "effective_h3_resolution_max": max(counts.keys()) if counts else base_res,
            "cell_ids": cell_ids,
            "xyz": xyz,
            "kdtree": kdtree,
            "cell_radii": cell_radii_arr,
            "max_cell_radius_m": max_cell_radius_m,
        }
        self._adaptive_mesh_cache[key] = mesh
        return mesh

    def get_adaptive_full_cells(self, base_resolution: int, config: MeshConfig) -> set[str]:
        return set(self._build_adaptive_mesh(base_resolution, config)["full_cells"])

    def get_adaptive_road_cells(self, base_resolution: int, config: MeshConfig) -> set[str]:
        return set(self._build_adaptive_mesh(base_resolution, config)["road_cells"])

    def get_adaptive_cell_metadata(
        self,
        h3_index: str,
        base_resolution: int,
        config: MeshConfig,
    ) -> dict:
        mesh = self._build_adaptive_mesh(base_resolution, config)
        meta = mesh["cell_meta"].get(h3_index)
        if meta is not None:
            return dict(meta)
        res = int(h3.get_resolution(h3_index))
        return {
            "base_h3_resolution": int(base_resolution),
            "target_h3_resolution": res,
            "gradient_m_per_km": 0.0,
            "adaptive_refined": False,
        }

    def adaptive_resolution_summary(self, base_resolution: int, config: MeshConfig) -> dict:
        mesh = self._build_adaptive_mesh(base_resolution, config)
        return {
            "h3_resolution_mode": "adaptive_mixed",
            "base_h3_resolution": int(base_resolution),
            "effective_h3_resolution_min": mesh["effective_h3_resolution_min"],
            "effective_h3_resolution_max": mesh["effective_h3_resolution_max"],
            "cells_by_resolution": dict(sorted(mesh["cells_by_resolution"].items())),
        }

    def locate_adaptive_cell(
        self,
        lat: float,
        lon: float,
        base_resolution: int,
        config: MeshConfig,
        *,
        prefer_road: bool = False,
    ) -> str:
        mesh = self._build_adaptive_mesh(base_resolution, config)
        by_res = mesh["road_by_res"] if prefer_road else mesh["full_by_res"]
        if not by_res:
            return h3.latlng_to_cell(lat, lon, int(base_resolution))
        for res in sorted(by_res.keys(), reverse=True):
            idx = h3.latlng_to_cell(lat, lon, int(res))
            if idx in by_res.get(int(res), set()):
                return idx
        return h3.latlng_to_cell(lat, lon, int(base_resolution))

    def adaptive_cells_within_radius(
        self,
        center_h3: str,
        radius_m: float,
        base_resolution: int,
        config: MeshConfig,
        *,
        candidate_cells: Optional[set[str]] = None,
    ) -> set[str]:
        if radius_m <= 0:
            return {center_h3}
        mesh = self._build_adaptive_mesh(base_resolution, config)
        lat, lon = h3.cell_to_latlng(center_h3)
        center_radius = _cell_edge_length_m(h3.get_resolution(center_h3))
        max_cell_radius_m = float(mesh.get("max_cell_radius_m", 0.0))
        xyz = mesh.get("xyz")
        cell_ids = mesh.get("cell_ids") or []
        cell_radii = mesh.get("cell_radii")

        if candidate_cells is None:
            kdtree = mesh.get("kdtree")
            if kdtree is None:
                return set()
            q = _to_xyz(np.asarray([lat]), np.asarray([lon]))[0]
            idxs = kdtree.query_ball_point(
                q, r=float(radius_m + center_radius + max_cell_radius_m)
            )
            out: set[str] = set()
            for i in idxs:
                cand_r = (
                    float(cell_radii[i])
                    if cell_radii is not None
                    else _cell_edge_length_m(h3.get_resolution(cell_ids[i]))
                )
                if float(np.linalg.norm(xyz[i] - q)) <= float(radius_m + center_radius + cand_r):
                    out.add(cell_ids[i])
            return out

        candidate_count = len(candidate_cells)
        full_count = len(mesh.get("full_cells", []))
        kdtree = mesh.get("kdtree")
        if (
            kdtree is not None
            and full_count > 0
            and candidate_count >= int(full_count * 0.5)
        ):
            q = _to_xyz(np.asarray([lat]), np.asarray([lon]))[0]
            idxs = kdtree.query_ball_point(
                q, r=float(radius_m + center_radius + max_cell_radius_m)
            )
            out: set[str] = set()
            for i in idxs:
                h3_idx = cell_ids[i]
                if h3_idx not in candidate_cells:
                    continue
                cand_r = (
                    float(cell_radii[i])
                    if cell_radii is not None
                    else _cell_edge_length_m(h3.get_resolution(h3_idx))
                )
                if float(np.linalg.norm(xyz[i] - q)) <= float(radius_m + center_radius + cand_r):
                    out.add(h3_idx)
            return out

        out: set[str] = set()
        for h3_idx in candidate_cells:
            lat_b, lon_b = h3.cell_to_latlng(h3_idx)
            cand_radius = _cell_edge_length_m(h3.get_resolution(h3_idx))
            if great_circle_distance(lat, lon, lat_b, lon_b) <= float(
                radius_m + center_radius + cand_radius
            ):
                out.add(h3_idx)
        return out

    def adaptive_union_within_radius(
        self,
        centers: Iterable[str],
        radius_m: float,
        base_resolution: int,
        config: MeshConfig,
        *,
        candidate_cells: Optional[set[str]] = None,
    ) -> set[str]:
        """Return union of adaptive cells within radius from multiple center cells."""
        centers = list(centers)
        if not centers:
            return set()
        if radius_m <= 0:
            return set(centers)
        mesh = self._build_adaptive_mesh(base_resolution, config)
        kdtree = mesh.get("kdtree")
        if kdtree is not None:
            union_ids: set[str] = set()
            xyz = mesh.get("xyz")
            cell_ids = mesh.get("cell_ids") or []
            cell_radii = mesh.get("cell_radii")
            max_cell_radius_m = float(mesh.get("max_cell_radius_m", 0.0))
            for h3_idx in centers:
                lat, lon = h3.cell_to_latlng(h3_idx)
                q = _to_xyz(np.asarray([lat]), np.asarray([lon]))[0]
                center_radius = _cell_edge_length_m(h3.get_resolution(h3_idx))
                query_r = float(radius_m + center_radius + max_cell_radius_m)
                idxs = kdtree.query_ball_point(q, r=query_r)
                if not idxs:
                    continue
                for i in idxs:
                    cid = cell_ids[i]
                    if candidate_cells is not None and cid not in candidate_cells:
                        continue
                    cand_r = (
                        float(cell_radii[i])
                        if cell_radii is not None
                        else _cell_edge_length_m(h3.get_resolution(cid))
                    )
                    if float(np.linalg.norm(xyz[i] - q)) <= float(radius_m + center_radius + cand_r):
                        union_ids.add(cid)
            # Ensure query centers are preserved when they are part of candidate domain.
            if candidate_cells is None:
                union_ids.update(centers)
            else:
                union_ids.update(c for c in centers if c in candidate_cells)
            return union_ids

        out: set[str] = set()
        for center_h3 in centers:
            out.update(
                self.adaptive_cells_within_radius(
                    center_h3,
                    radius_m,
                    base_resolution,
                    config,
                    candidate_cells=candidate_cells,
                )
            )
        return out

    def adaptive_corridor_from_features(
        self,
        features: list,
        base_resolution: int,
        config: MeshConfig,
        site1: Optional[dict] = None,
        site2: Optional[dict] = None,
    ) -> list[str]:
        """
        Build an ordered mixed-resolution corridor by projecting high-res road samples
        onto the adaptive road mesh and deduplicating in path order.
        """
        high_res = max(self.available_resolutions() or [int(base_resolution)])
        ordered_high = road_geojson_to_h3_corridor(features, high_res, site1=site1, site2=site2)
        if not ordered_high:
            ordered_high = road_geojson_to_h3_corridor(
                features, int(base_resolution), site1=site1, site2=site2
            )
        mapped: list[str] = []
        seen: set[str] = set()
        for h3_idx in ordered_high:
            lat, lon = h3.cell_to_latlng(h3_idx)
            adaptive = self.locate_adaptive_cell(
                lat,
                lon,
                int(base_resolution),
                config,
                prefer_road=True,
            )
            if adaptive not in seen:
                mapped.append(adaptive)
                seen.add(adaptive)

        if site1 and "lat" in site1 and "lon" in site1:
            start = self.locate_adaptive_cell(
                float(site1["lat"]),
                float(site1["lon"]),
                int(base_resolution),
                config,
                prefer_road=True,
            )
            if mapped and mapped[0] != start:
                mapped.insert(0, start)
            elif not mapped:
                mapped.append(start)

        if site2 and "lat" in site2 and "lon" in site2:
            end = self.locate_adaptive_cell(
                float(site2["lat"]),
                float(site2["lon"]),
                int(base_resolution),
                config,
                prefer_road=True,
            )
            if mapped and mapped[-1] != end:
                mapped.append(end)
            elif not mapped:
                mapped.append(end)

        return mapped

    def get_or_create_cell(
        self,
        h3_index: str,
        config: MeshConfig,
        *,
        has_road: bool = False,
        is_in_boundary: bool = True,
    ) -> H3Cell:
        try:
            res = int(h3.get_resolution(h3_index))
        except Exception:
            res = int(config.h3_resolution)
        by_res = self._cell_cache_by_res.setdefault(res, {})
        cell = by_res.get(h3_index)
        if cell is not None:
            cell.has_road = bool(cell.has_road or has_road)
            cell.is_in_boundary = bool(cell.is_in_boundary or is_in_boundary)
            meta = self.get_adaptive_cell_metadata(h3_index, config.h3_resolution, config)
            setattr(cell, "base_h3_resolution", meta["base_h3_resolution"])
            setattr(cell, "target_h3_resolution", meta["target_h3_resolution"])
            setattr(cell, "gradient_m_per_km", meta["gradient_m_per_km"])
            setattr(cell, "adaptive_refined", meta["adaptive_refined"])
            return cell
        lat, lon = h3.cell_to_latlng(h3_index)
        static = self._cell_static_by_res.get(res, {}).get(h3_index)
        if static is not None:
            elev = float(static.get("elevation_max_m", 0.0))
            los_lat = float(static.get("los_anchor_lat", lat))
            los_lon = float(static.get("los_anchor_lon", lon))
        else:
            elev, los_lat, los_lon = resolve_cell_profile(
                self._elevation_provider,
                h3_index,
                lat,
                lon,
                anchor_margin_m=config.cell_anchor_margin_m,
            )
        cell = H3Cell(
            h3_index=h3_index,
            lat=lat,
            lon=lon,
            elevation=elev,
            has_road=has_road,
            is_in_boundary=is_in_boundary,
            los_lat=los_lat,
            los_lon=los_lon,
        )
        meta = self.get_adaptive_cell_metadata(h3_index, config.h3_resolution, config)
        setattr(cell, "base_h3_resolution", meta["base_h3_resolution"])
        setattr(cell, "target_h3_resolution", meta["target_h3_resolution"])
        setattr(cell, "gradient_m_per_km", meta["gradient_m_per_km"])
        setattr(cell, "adaptive_refined", meta["adaptive_refined"])
        by_res = self._cell_cache_by_res.setdefault(res, {})
        prev = by_res.get(h3_index)
        if prev is not None:
            prev.has_road = bool(prev.has_road or has_road)
            prev.is_in_boundary = bool(prev.is_in_boundary or is_in_boundary)
            meta = self.get_adaptive_cell_metadata(h3_index, config.h3_resolution, config)
            setattr(prev, "base_h3_resolution", meta["base_h3_resolution"])
            setattr(prev, "target_h3_resolution", meta["target_h3_resolution"])
            setattr(prev, "gradient_m_per_km", meta["gradient_m_per_km"])
            setattr(prev, "adaptive_refined", meta["adaptive_refined"])
            return prev
        by_res[h3_index] = cell
        return cell

    def materialize_cells(
        self,
        h3_indices: Iterable[str],
        config: MeshConfig,
        *,
        road_cells: Optional[set[str]] = None,
        is_in_boundary: bool = True,
        include_stats: bool = False,
    ):
        road_set = road_cells or set()
        indices = list(dict.fromkeys(h3_indices))
        out: Dict[str, H3Cell] = {}
        stats = {
            "requested": len(indices),
            "cache_hits": 0,
            "from_static": 0,
            "from_dem": 0,
        }
        adaptive_mesh = self._build_adaptive_mesh(config.h3_resolution, config)
        adaptive_meta = adaptive_mesh.get("cell_meta", {})

        for h3_idx in indices:
            try:
                res = int(h3.get_resolution(h3_idx))
            except Exception:
                res = int(config.h3_resolution)
            has_road = h3_idx in road_set

            by_res = self._cell_cache_by_res.setdefault(res, {})
            cell = by_res.get(h3_idx)
            if cell is not None:
                cell.has_road = bool(cell.has_road or has_road)
                cell.is_in_boundary = bool(cell.is_in_boundary or is_in_boundary)
                meta = adaptive_meta.get(h3_idx)
                if meta is None:
                    meta = {
                        "base_h3_resolution": int(config.h3_resolution),
                        "target_h3_resolution": int(h3.get_resolution(h3_idx)),
                        "gradient_m_per_km": 0.0,
                        "adaptive_refined": False,
                    }
                setattr(cell, "base_h3_resolution", meta["base_h3_resolution"])
                setattr(cell, "target_h3_resolution", meta["target_h3_resolution"])
                setattr(cell, "gradient_m_per_km", meta["gradient_m_per_km"])
                setattr(cell, "adaptive_refined", meta["adaptive_refined"])
                out[h3_idx] = cell
                stats["cache_hits"] += 1
                continue

            lat, lon = h3.cell_to_latlng(h3_idx)
            static = self._cell_static_by_res.get(res, {}).get(h3_idx)
            if static is not None:
                elev = float(static.get("elevation_max_m", 0.0))
                los_lat = float(static.get("los_anchor_lat", lat))
                los_lon = float(static.get("los_anchor_lon", lon))
                stats["from_static"] += 1
            else:
                elev, los_lat, los_lon = resolve_cell_profile(
                    self._elevation_provider,
                    h3_idx,
                    lat,
                    lon,
                    anchor_margin_m=config.cell_anchor_margin_m,
                )
                stats["from_dem"] += 1
            cell = H3Cell(
                h3_index=h3_idx,
                lat=lat,
                lon=lon,
                elevation=elev,
                has_road=has_road,
                is_in_boundary=is_in_boundary,
                los_lat=los_lat,
                los_lon=los_lon,
            )
            meta = adaptive_meta.get(h3_idx)
            if meta is None:
                meta = {
                    "base_h3_resolution": int(config.h3_resolution),
                    "target_h3_resolution": int(h3.get_resolution(h3_idx)),
                    "gradient_m_per_km": 0.0,
                    "adaptive_refined": False,
                }
            setattr(cell, "base_h3_resolution", meta["base_h3_resolution"])
            setattr(cell, "target_h3_resolution", meta["target_h3_resolution"])
            setattr(cell, "gradient_m_per_km", meta["gradient_m_per_km"])
            setattr(cell, "adaptive_refined", meta["adaptive_refined"])

            by_res = self._cell_cache_by_res.setdefault(res, {})
            by_res[h3_idx] = cell
            out[h3_idx] = cell

        if include_stats:
            stats["materialized"] = len(out)
            return out, stats
        return out

    def build_cells_dict(
        self,
        h3_indices: Iterable[str],
        config: MeshConfig,
        *,
        road_cells: Optional[set[str]] = None,
        is_in_boundary: bool = True,
    ) -> Dict[str, H3Cell]:
        return self.materialize_cells(
            h3_indices,
            config,
            road_cells=road_cells,
            is_in_boundary=is_in_boundary,
        )

    def build_full_boundary_grid(self, config: MeshConfig) -> Dict[str, H3Cell]:
        if bool(getattr(config, "auto_refine_h3_on_gradient", False)):
            full = self.get_adaptive_full_cells(config.h3_resolution, config)
            road = self.get_adaptive_road_cells(config.h3_resolution, config)
        else:
            full = self.get_full_cells(config.h3_resolution)
            road = self.get_road_cells(config.h3_resolution)
        return self.build_cells_dict(full, config, road_cells=road, is_in_boundary=True)

    def resolve_effective_resolution(
        self,
        routes: list[RouteSpec],
        base_resolution: int,
        config: MeshConfig,
    ) -> tuple[int, bool, Optional[str], Optional[float]]:
        """
        Resolve effective resolution via hardcoded slope ladder.

        Ladder (by configured percentile gradient):
        - >75  m/km -> 9
        """
        if not routes:
            return base_resolution, False, None, None
        slopes: list[float] = []
        for route in routes:
            corridor = self.corridor_from_features(
                route.features,
                base_resolution,
                site1=route.site1,
                site2=route.site2,
            )
            if len(corridor) < 2:
                continue
            for a, b in zip(corridor, corridor[1:]):
                lat_a, lon_a = h3.cell_to_latlng(a)
                lat_b, lon_b = h3.cell_to_latlng(b)
                dist_m = great_circle_distance(lat_a, lon_a, lat_b, lon_b)
                if dist_m <= 1.0:
                    continue
                elev_a = self.get_h3_cell_max_elevation(a)
                elev_b = self.get_h3_cell_max_elevation(b)
                slopes.append(abs(elev_b - elev_a) / (dist_m / 1000.0))
        if not slopes:
            return base_resolution, False, None, None

        pctl = float(np.percentile(np.asarray(slopes, dtype=np.float64), config.gradient_refine_percentile))
        target = base_resolution
        if pctl > 75.0:
            target = 9

        target = max(base_resolution, target)
        target = min(target, int(config.auto_refine_h3_max_resolution))
        refined = target > base_resolution
        reason = None
        if refined:
            reason = (
                f"terrain_gradient_p{config.gradient_refine_percentile:.0f}"
                f"={pctl:.1f}m_per_km->h3_res_{target}"
            )
        return target, refined, reason, pctl
