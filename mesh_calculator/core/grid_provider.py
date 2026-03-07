"""
Grid provider abstraction for route optimization and runtime coverage.

This module centralizes grid/elevation access so planners can be resolution-agnostic
and consume prebuilt grid bundles instead of rebuilding grid internals ad hoc.
"""
from __future__ import annotations

import json
import os
import hashlib
import threading
from typing import Dict, Iterable, Optional

import geopandas as gpd
import h3
import numpy as np
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

_DEFAULT_BUNDLE_RESOLUTIONS = (8, 9, 10, 11)
_GRID_BUNDLE_VERSION = 2
_SUPPORTED_BUNDLE_VERSIONS = {1, 2}


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


class GridProvider:
    """
    Multi-resolution grid/elevation provider.

    Responsibilities:
    - Bundle build/load for resolutions 8..11 (or configured set)
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
        self._cell_cache_by_res: Dict[int, Dict[str, H3Cell]] = {}
        self._lock = threading.Lock()
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
        elev_path = elevation_path or payload.get("elevation_path")
        if elev_path and not os.path.isabs(elev_path):
            elev_path = os.path.join(os.path.dirname(os.path.abspath(bundle_path)), elev_path)
        provider = cls(
            elevation_path=elev_path,
            boundary_geojson=payload.get("boundary_geojson"),
            roads_geojson=payload.get("roads_geojson"),
            full_cells_by_res=full_cells_by_res,
            road_cells_by_res=road_cells_by_res,
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

        by_res = {}
        for res in res_list:
            full_cells = set(shapely_to_h3_cells(boundary_poly, res))
            road_cells = set()
            if roads_gdf is not None and len(roads_gdf):
                road_cells = set(find_h3_cells_on_roads(roads_gdf, res)) & full_cells
            by_res[str(res)] = {
                "full_cells": sorted(full_cells),
                "road_cells": sorted(road_cells),
            }
            logger.info(
                "Grid bundle resolution built",
                resolution=res,
                full_cells=len(full_cells),
                road_cells=len(road_cells),
            )

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
        for res in res_list:
            full = set(shapely_to_h3_cells(boundary_poly, res))
            full_cells_by_res[res] = full
            if roads_gdf is not None and len(roads_gdf):
                road_cells_by_res[res] = set(find_h3_cells_on_roads(roads_gdf, res)) & full
            else:
                road_cells_by_res[res] = set()
        return cls(
            elevation_path=elevation_path,
            boundary_geojson=boundary_geojson,
            roads_geojson=roads_fc,
            full_cells_by_res=full_cells_by_res,
            road_cells_by_res=road_cells_by_res,
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
                self._road_cells_by_res[res] = set(find_h3_cells_on_roads(roads_gdf, res))
            else:
                self._road_cells_by_res[res] = set()
        return self._road_cells_by_res[res]

    def corridor_from_features(
        self,
        features: list,
        resolution: int,
        site1: Optional[dict] = None,
        site2: Optional[dict] = None,
    ) -> list[str]:
        return road_geojson_to_h3_corridor(features, int(resolution), site1=site1, site2=site2)

    def radius_m_to_ring(self, radius_m: float, resolution: int, minimum_one: bool = False) -> int:
        edge_m = h3.average_hexagon_edge_length(int(resolution), unit="m")
        rings = int(round(max(float(radius_m), 0.0) / edge_m)) if edge_m > 0 else 0
        if minimum_one and radius_m > 0:
            rings = max(1, rings)
        return max(0, rings)

    def expand_disk(self, h3_index: str, rings: int) -> set[str]:
        return set(h3.grid_disk(h3_index, int(rings)))

    def get_or_create_cell(
        self,
        h3_index: str,
        config: MeshConfig,
        *,
        has_road: bool = False,
        is_in_boundary: bool = True,
    ) -> H3Cell:
        res = int(config.h3_resolution)
        with self._lock:
            by_res = self._cell_cache_by_res.setdefault(res, {})
            cell = by_res.get(h3_index)
            if cell is not None:
                cell.has_road = bool(cell.has_road or has_road)
                cell.is_in_boundary = bool(cell.is_in_boundary or is_in_boundary)
                return cell
        lat, lon = h3.cell_to_latlng(h3_index)
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
        with self._lock:
            by_res = self._cell_cache_by_res.setdefault(res, {})
            prev = by_res.get(h3_index)
            if prev is not None:
                prev.has_road = bool(prev.has_road or has_road)
                prev.is_in_boundary = bool(prev.is_in_boundary or is_in_boundary)
                return prev
            by_res[h3_index] = cell
        return cell

    def build_cells_dict(
        self,
        h3_indices: Iterable[str],
        config: MeshConfig,
        *,
        road_cells: Optional[set[str]] = None,
        is_in_boundary: bool = True,
    ) -> Dict[str, H3Cell]:
        road_set = road_cells or set()
        out: Dict[str, H3Cell] = {}
        for h3_idx in h3_indices:
            out[h3_idx] = self.get_or_create_cell(
                h3_idx,
                config,
                has_road=(h3_idx in road_set),
                is_in_boundary=is_in_boundary,
            )
        return out

    def build_full_boundary_grid(self, config: MeshConfig) -> Dict[str, H3Cell]:
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
        - >100 m/km -> 11
        - >75  m/km -> 10
        - >50  m/km -> 9
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
        if pctl > 100.0:
            target = 11
        elif pctl > 75.0:
            target = 10
        elif pctl > 50.0:
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
