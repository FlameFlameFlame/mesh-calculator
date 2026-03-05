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
from ..core.grid import H3Cell
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
) -> list[CoverageSource]:
    """Snap all sources to the requested coverage H3 resolution."""
    normalized: list[CoverageSource] = []
    for src in sources:
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
    elevation_provider=None,
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

    step_m = max(float(config.los_dense_sample_step_m), 1.0)
    max_samples = max(int(config.los_dense_max_samples), 2)
    n_samples = max(2, int(distance_m / step_m))
    n_samples = min(n_samples, max_samples)

    fracs = np.linspace(0.0, 1.0, n_samples + 1)
    lats = source_cell.lat + (target_cell.lat - source_cell.lat) * fracs
    lons = source_cell.lon + (target_cell.lon - source_cell.lon) * fracs

    if elevation_provider is not None:
        elevs = np.array(
            [elevation_provider.get_elevation(float(lat), float(lon))
             for lat, lon in zip(lats, lons)],
            dtype=np.float64,
        )
    else:
        # Fallback for tests/edge paths without DEM provider.
        elevs = np.linspace(source_cell.elevation, target_cell.elevation, n_samples + 1)

    d1s = distance_m * fracs
    d2s = distance_m - d1s
    line_alts = src_height_asl + (dst_height_asl - src_height_asl) * fracs
    earth_curvs = (d1s * d2s) / (2.0 * config.effective_earth_radius_m)
    clearances = line_alts - (elevs + earth_curvs)
    worst_clearance = float(clearances[np.argmin(clearances)])

    if worst_clearance < 0.0:
        return distance_m, worst_clearance, float("inf"), False

    path_loss_db = fspl_only(distance_m, config.frequency_hz)
    is_visible = path_loss_db <= config.link_budget_db
    return distance_m, worst_clearance, path_loss_db, is_visible


def compute_h3_tower_coverage(
    sources: Iterable[CoverageSource],
    base_cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider=None,
    los_cache: LOSCache = None,
    max_radius_m: Optional[float] = None,
) -> list[dict]:
    """
    Compute radial H3 coverage for one or more explicit source points.

    Returns covered cells only. The source cell itself is always included
    with distance/path-loss set to 0.
    """
    src_list = _dedupe_sources(
        _normalize_sources_to_resolution(sources, config.h3_resolution)
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
    for src in src_list:
        candidate_h3s.update(h3.grid_disk(src.h3_index, max_rings))

    augmented_cells = dict(base_cells)
    for src in src_list:
        if src.h3_index not in augmented_cells:
            elev = elevation_provider.get_elevation(src.lat, src.lon) if elevation_provider else 0.0
            augmented_cells[src.h3_index] = H3Cell(
                h3_index=src.h3_index,
                lat=src.lat,
                lon=src.lon,
                elevation=elev,
                has_road=False,
                is_in_boundary=False,
            )

    for h3_idx in candidate_h3s:
        if h3_idx not in augmented_cells:
            lat, lon = h3.cell_to_latlng(h3_idx)
            elev = elevation_provider.get_elevation(lat, lon) if elevation_provider else 0.0
            augmented_cells[h3_idx] = H3Cell(
                h3_index=h3_idx,
                lat=lat,
                lon=lon,
                elevation=elev,
                has_road=False,
                is_in_boundary=False,
            )

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
            elevation_provider=elevation_provider,
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

    # ci -> aggregate metrics
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

    tx_dbm = config.tx_power_dbm
    gain = 2.0 * config.antenna_gain_dbi
    sens = config.receiver_sensitivity_dbm

    results = []
    for ci, (count, closest_dist, closest_sid, best_ploss, best_sid, best_clear, best_dist) in cell_results.items():
        cell = cand_list[ci]
        rx_dbm = None
        is_covered = False
        if best_ploss is not None and best_ploss != float('inf'):
            rx_dbm = tx_dbm + gain - best_ploss
            is_covered = (count > 0 and rx_dbm >= sens)

        if not is_covered:
            continue

        results.append({
            'h3_index': cell.h3_index,
            'lat': cell.lat,
            'lon': cell.lon,
            'elevation': cell.elevation,
            'has_road': cell.has_road,
            'visible_tower_count': count,
            'distance_m': best_dist if best_dist is not None and best_dist != float('inf') else None,
            'clearance_m': best_clear if best_clear is not None and best_clear != float('inf') else None,
            'path_loss_db': best_ploss if best_ploss != float('inf') else None,
            'received_power_dbm': round(rx_dbm, 2) if rx_dbm is not None else None,
            'is_covered': is_covered,
            'serving_tower_id': best_sid,
            'closest_tower_id': closest_sid,
            'closest_distance_m': closest_dist if closest_dist != float('inf') else None,
        })

    logger.info("Standalone tower coverage computed: %d covered hexes", len(results))
    return results
