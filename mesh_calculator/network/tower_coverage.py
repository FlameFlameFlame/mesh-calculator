"""
Standalone H3 tower coverage computation.
"""
from __future__ import annotations

import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

import h3
import numpy as np
from scipy.spatial import cKDTree
import structlog

from ..core.config import MeshConfig
from ..core.grid import H3Cell, resolve_cell_profile
from ..data.cache import LOSCache
from ..core.geometry import great_circle_distance
from ..physics.path_loss import fspl_only

logger = structlog.get_logger(__name__)

# Earth radius for approximate Cartesian conversion (meters)
_EARTH_R = 6_371_000


@dataclass(frozen=True)
class CoverageSource:
    """Signal source location for standalone coverage computation."""

    source_id: Union[int, str]
    h3_index: str
    lat: float
    lon: float


def _provider_supports(provider, method_name: str) -> bool:
    """True only if provider type explicitly implements the method."""
    if provider is None:
        return False
    return callable(getattr(type(provider), method_name, None))


def _cell_elevation(elevation_provider, h3_index: str, lat: float, lon: float) -> float:
    if elevation_provider is None:
        return 0.0
    if _provider_supports(elevation_provider, "get_h3_cell_max_elevation"):
        try:
            return float(elevation_provider.get_h3_cell_max_elevation(h3_index))
        except Exception:
            pass
    get_elevation = getattr(elevation_provider, "get_elevation", None)
    if callable(get_elevation):
        try:
            return float(get_elevation(lat, lon))
        except Exception:
            return 0.0
    return 0.0


def _sample_shadow_profile(
    source_cell: H3Cell,
    target_cell: H3Cell,
    distance_m: float,
    config: MeshConfig,
    elevation_provider=None,
) -> list[tuple[float, float]]:
    """Sample terrain elevations along source->target for shadow-casting check."""
    samples: list[tuple[float, float]] = [
        (0.0, float(source_cell.elevation)),
        (1.0, float(target_cell.elevation)),
    ]
    if elevation_provider is None:
        return samples
    get_elevation = getattr(elevation_provider, "get_elevation", None)
    if not callable(get_elevation) or distance_m <= 0.0:
        return samples

    step_m = max(float(config.los_dense_sample_step_m), 1.0)
    max_samples = max(int(config.los_dense_max_samples), 2)
    n_samples = max(2, int(distance_m / step_m))
    n_samples = min(n_samples, max_samples)
    for i in range(1, n_samples):
        frac = i / n_samples
        lat = source_cell.lat + (target_cell.lat - source_cell.lat) * frac
        lon = source_cell.lon + (target_cell.lon - source_cell.lon) * frac
        try:
            elev = float(get_elevation(lat, lon))
        except Exception:
            continue
        samples.append((frac, elev))
    return samples


def _to_xyz(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    cos_lat = np.cos(lats_rad)
    return np.column_stack([
        cos_lat * np.cos(lons_rad),
        cos_lat * np.sin(lons_rad),
        np.sin(lats_rad),
    ]) * _EARTH_R


def _dedupe_sources(sources: Iterable[CoverageSource]) -> list[CoverageSource]:
    unique: Dict[str, CoverageSource] = {}
    for src in sources:
        if src.h3_index not in unique:
            unique[src.h3_index] = src
    return list(unique.values())


def _normalize_sources_to_resolution(
    sources: Iterable[CoverageSource],
    resolution: int,
    config: MeshConfig,
    grid_provider=None,
) -> list[CoverageSource]:
    """Snap all sources to coverage grid (adaptive when provider supports it)."""
    normalized: list[CoverageSource] = []
    use_adaptive = _provider_supports(grid_provider, "locate_adaptive_cell")
    for src in sources:
        if use_adaptive:
            snapped_h3 = grid_provider.locate_adaptive_cell(
                src.lat,
                src.lon,
                resolution,
                config,
            )
        else:
            snapped_h3 = h3.latlng_to_cell(src.lat, src.lon, resolution)
        snapped_lat, snapped_lon = h3.cell_to_latlng(snapped_h3)
        normalized.append(CoverageSource(
            source_id=src.source_id,
            h3_index=snapped_h3,
            lat=snapped_lat,
            lon=snapped_lon,
        ))
    return normalized


def _ring_step_m(resolution: int) -> float:
    """Approximate center-to-center distance for one H3 ring step."""
    edge_m = h3.average_hexagon_edge_length(resolution, unit='m')
    return edge_m * math.sqrt(3.0)


def _compute_shadow_link(
    source_cell: H3Cell,
    target_cell: H3Cell,
    config: MeshConfig,
    grid_provider=None,
) -> tuple[float, float, float, bool]:
    """
    Compute strict terrain-shadow viability for tower coverage.

    This is intentionally stricter than optimization LOS policy:
    - hard geometric LOS requirement (no diffraction acceptance)
    - FSPL-only budget check once LOS is clear
    - receiver endpoint height uses coverage_receiver_height_m
    """
    distance_m = great_circle_distance(
        source_cell.lat, source_cell.lon, target_cell.lat, target_cell.lon
    )
    if distance_m <= 0.0:
        return 0.0, config.coverage_receiver_height_m, 0.0, True

    src_height_asl = source_cell.elevation + config.mast_height_m
    dst_height_asl = target_cell.elevation + config.coverage_receiver_height_m

    profile = _sample_shadow_profile(
        source_cell=source_cell,
        target_cell=target_cell,
        distance_m=distance_m,
        config=config,
        elevation_provider=grid_provider,
    )
    worst_clearance = float("inf")
    for frac, terrain_elev in profile:
        d1 = distance_m * frac
        d2 = distance_m - d1
        line_alt = src_height_asl + (dst_height_asl - src_height_asl) * frac
        earth_curv = (d1 * d2) / (2.0 * config.effective_earth_radius_m)
        clearance = float(line_alt - (terrain_elev + earth_curv))
        if clearance < worst_clearance:
            worst_clearance = clearance

    if worst_clearance < 0.0:
        return distance_m, worst_clearance, float("inf"), False

    path_loss_db = fspl_only(distance_m, config.frequency_hz)
    is_visible = path_loss_db <= config.link_budget_db
    return distance_m, worst_clearance, path_loss_db, is_visible


def compute_h3_tower_coverage(
    sources: Iterable[CoverageSource],
    base_cells: Dict[str, H3Cell],
    config: MeshConfig,
    grid_provider=None,
    los_cache: LOSCache = None,
    max_radius_m: Optional[float] = None,
) -> list[dict]:
    """
    Compute radial H3 coverage for one or more explicit source points.

    Returns all cells within radius (covered and uncovered). The source cell
    itself is always included with distance/path-loss set to 0.
    """
    src_list = _dedupe_sources(
        _normalize_sources_to_resolution(
            sources,
            config.h3_resolution,
            config,
            grid_provider=grid_provider,
        )
    )
    if not src_list:
        return []

    coverage_radius_m = max_radius_m if max_radius_m is not None else config.max_coverage_radius_m
    if coverage_radius_m <= 0:
        max_rings = 0
    else:
        # One ring roughly advances by center-to-center spacing, not edge length.
        # Add a safety ring to ensure boundary cells are included, then filter
        # with exact geodesic radius via KD-tree.
        ring_step_m = _ring_step_m(config.h3_resolution)
        max_rings = max(1, int(math.ceil(coverage_radius_m / ring_step_m)) + 1)

    logger.info(
        "Standalone tower coverage: %d source(s), radius=%.0fm, rings=%d, h3_res=%d",
        len(src_list), coverage_radius_m, max_rings, config.h3_resolution,
    )

    candidate_h3s: set[str] = set()
    if _provider_supports(grid_provider, "adaptive_cells_within_radius"):
        full_pool = set(grid_provider.get_adaptive_full_cells(config.h3_resolution, config))
        for src in src_list:
            candidate_h3s.update(grid_provider.adaptive_cells_within_radius(
                src.h3_index,
                coverage_radius_m,
                config.h3_resolution,
                config,
                candidate_cells=full_pool,
            ))
    else:
        for src in src_list:
            candidate_h3s.update(h3.grid_disk(src.h3_index, max_rings))

    augmented_cells = dict(base_cells)
    for src in src_list:
        if src.h3_index not in augmented_cells:
            elev, los_lat, los_lon = resolve_cell_profile(
                grid_provider,
                src.h3_index,
                src.lat,
                src.lon,
                anchor_margin_m=config.cell_anchor_margin_m,
            )
            augmented_cells[src.h3_index] = H3Cell(
                h3_index=src.h3_index,
                lat=src.lat,
                lon=src.lon,
                elevation=elev,
                has_road=False,
                is_in_boundary=False,
                los_lat=los_lat,
                los_lon=los_lon,
            )
            if _provider_supports(grid_provider, "get_adaptive_cell_metadata"):
                meta = grid_provider.get_adaptive_cell_metadata(
                    src.h3_index,
                    config.h3_resolution,
                    config,
                )
                setattr(augmented_cells[src.h3_index], "base_h3_resolution", meta["base_h3_resolution"])
                setattr(augmented_cells[src.h3_index], "target_h3_resolution", meta["target_h3_resolution"])
                setattr(augmented_cells[src.h3_index], "gradient_m_per_km", meta["gradient_m_per_km"])
                setattr(augmented_cells[src.h3_index], "adaptive_refined", meta["adaptive_refined"])

    for h3_idx in candidate_h3s:
        if h3_idx not in augmented_cells:
            lat, lon = h3.cell_to_latlng(h3_idx)
            elev, los_lat, los_lon = resolve_cell_profile(
                grid_provider,
                h3_idx,
                lat,
                lon,
                anchor_margin_m=config.cell_anchor_margin_m,
            )
            augmented_cells[h3_idx] = H3Cell(
                h3_index=h3_idx,
                lat=lat,
                lon=lon,
                elevation=elev,
                has_road=False,
                is_in_boundary=False,
                los_lat=los_lat,
                los_lon=los_lon,
            )
            if _provider_supports(grid_provider, "get_adaptive_cell_metadata"):
                meta = grid_provider.get_adaptive_cell_metadata(
                    h3_idx,
                    config.h3_resolution,
                    config,
                )
                setattr(augmented_cells[h3_idx], "base_h3_resolution", meta["base_h3_resolution"])
                setattr(augmented_cells[h3_idx], "target_h3_resolution", meta["target_h3_resolution"])
                setattr(augmented_cells[h3_idx], "gradient_m_per_km", meta["gradient_m_per_km"])
                setattr(augmented_cells[h3_idx], "adaptive_refined", meta["adaptive_refined"])

    cand_list = [augmented_cells[h] for h in candidate_h3s]
    src_cells = [augmented_cells[s.h3_index] for s in src_list]

    source_tree = cKDTree(_to_xyz(
        np.array([s.lat for s in src_cells]),
        np.array([s.lon for s in src_cells]),
    ))
    nearby_sources = source_tree.query_ball_point(_to_xyz(
        np.array([c.lat for c in cand_list]),
        np.array([c.lon for c in cand_list]),
    ), r=coverage_radius_m)

    source_by_h3 = {s.h3_index: s for s in src_list}
    source_h3_set = set(source_by_h3.keys())

    los_pairs = []
    cells_with_own_source = set()
    for ci, cell in enumerate(cand_list):
        if cell.h3_index in source_h3_set:
            cells_with_own_source.add(ci)
        for si in nearby_sources[ci]:
            source = src_list[si]
            if source.h3_index == cell.h3_index:
                continue
            los_pairs.append((ci, si))

    logger.info(
        "Standalone tower coverage: candidates=%d, LOS checks=%d",
        len(cand_list), len(los_pairs),
    )

    def _check_pair(pair):
        ci, si = pair
        c = cand_list[ci]
        src_cell = src_cells[si]
        distance_m, clearance_m, path_loss_db, is_visible = _compute_shadow_link(
            source_cell=src_cell,
            target_cell=c,
            config=config,
            grid_provider=grid_provider,
        )
        if is_visible:
            return (ci, si, distance_m, clearance_m, path_loss_db)
        return None

    def _check_pair_batch(batch):
        out = []
        for pair in batch:
            res = _check_pair(pair)
            if res is not None:
                out.append(res)
        return out

    # ci -> aggregate visible-link metrics
    cell_results: Dict[int, list] = {}
    for ci in cells_with_own_source:
        own_source = source_by_h3.get(cand_list[ci].h3_index)
        sid = own_source.source_id if own_source else None
        # [count, closest_dist, closest_sid, best_ploss, best_sid, best_clear, best_dist]
        cell_results[ci] = [1, 0.0, sid, 0.0, sid, 0.0, 0.0]

    def _accumulate_visible_result(ci, si, dist_m, clear_m, ploss_db):
        if ci not in cell_results:
            cell_results[ci] = [0, float('inf'), None, float('inf'), None, None, None]
        entry = cell_results[ci]
        entry[0] += 1
        if dist_m < entry[1]:
            entry[1] = dist_m
            entry[2] = src_list[si].source_id
        if ploss_db < entry[3]:
            entry[3] = ploss_db
            entry[4] = src_list[si].source_id
            entry[5] = clear_m
            entry[6] = dist_m

    if los_pairs:
        max_workers = min(os.cpu_count() or 4, 32)
        # Avoid one-future-per-pair overhead on large high-resolution runs.
        batch_size = max(64, len(los_pairs) // max(max_workers * 8, 1))
        batches = [
            los_pairs[i:i + batch_size]
            for i in range(0, len(los_pairs), batch_size)
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_check_pair_batch, b) for b in batches]
            for future in as_completed(futures):
                for ci, si, dist_m, clear_m, ploss_db in future.result():
                    _accumulate_visible_result(ci, si, dist_m, clear_m, ploss_db)

    nearest_any: Dict[int, tuple[float, Union[int, str, None]]] = {}
    for ci, src_ids in enumerate(nearby_sources):
        best_dist = float("inf")
        best_sid = None
        cell = cand_list[ci]
        for si in src_ids:
            source = src_cells[si]
            d = great_circle_distance(cell.lat, cell.lon, source.lat, source.lon)
            if d < best_dist:
                best_dist = d
                best_sid = src_list[si].source_id
        nearest_any[ci] = (best_dist, best_sid)

    tx_dbm = config.tx_power_dbm
    gain = 2.0 * config.antenna_gain_dbi
    sens = config.receiver_sensitivity_dbm

    results = []
    for ci, cell in enumerate(cand_list):
        count, closest_dist, closest_sid, best_ploss, best_sid, best_clear, best_dist = cell_results.get(
            ci, [0, float('inf'), None, float('inf'), None, None, None]
        )
        nearest_dist, nearest_sid = nearest_any.get(ci, (float('inf'), None))
        if closest_sid is None and nearest_sid is not None:
            closest_sid = nearest_sid
            closest_dist = nearest_dist
        cell = cand_list[ci]
        rx_dbm = None
        is_covered = False
        if best_ploss is not None and best_ploss != float('inf'):
            rx_dbm = tx_dbm + gain - best_ploss
            is_covered = (count > 0 and rx_dbm >= sens)

        results.append({
            'h3_index': cell.h3_index,
            'lat': cell.lat,
            'lon': cell.lon,
            'elevation': cell.elevation,
            'h3_resolution': int(h3.get_resolution(cell.h3_index)),
            'base_h3_resolution': getattr(cell, 'base_h3_resolution', config.h3_resolution),
            'target_h3_resolution': getattr(cell, 'target_h3_resolution', int(h3.get_resolution(cell.h3_index))),
            'gradient_m_per_km': float(getattr(cell, 'gradient_m_per_km', 0.0) or 0.0),
            'adaptive_refined': bool(getattr(cell, 'adaptive_refined', False)),
            'has_road': cell.has_road,
            'visible_tower_count': count,
            'distance_m': (
                best_dist
                if is_covered and best_dist is not None and best_dist != float('inf')
                else None
            ),
            'clearance_m': (
                best_clear
                if is_covered and best_clear is not None and best_clear != float('inf')
                else None
            ),
            'path_loss_db': (
                best_ploss
                if is_covered and best_ploss != float('inf')
                else None
            ),
            'received_power_dbm': round(rx_dbm, 2) if rx_dbm is not None else None,
            'is_covered': is_covered,
            'serving_tower_id': best_sid if is_covered else None,
            'closest_tower_id': closest_sid,
            'closest_distance_m': closest_dist if closest_dist != float('inf') else None,
        })

    covered_count = sum(1 for r in results if r.get('is_covered'))
    logger.info(
        "Standalone tower coverage computed: %d total hexes (%d covered)",
        len(results),
        covered_count,
    )
    return results
