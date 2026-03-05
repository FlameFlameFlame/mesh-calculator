"""
Standalone H3 tower coverage computation.
"""
from __future__ import annotations

import os
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
from ..physics.los import compute_los

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
    src_list = _dedupe_sources(sources)
    if not src_list:
        return []

    coverage_radius_m = max_radius_m if max_radius_m is not None else config.max_coverage_radius_m
    edge_m = h3.average_hexagon_edge_length(config.h3_resolution, unit='m')
    max_rings = max(1, int(coverage_radius_m / edge_m))

    logger.info(
        "Standalone tower coverage: %d source(s), radius=%.0fm, rings=%d",
        len(src_list), coverage_radius_m, max_rings,
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
        s = src_list[si]
        result = compute_los(
            c.h3_index, s.h3_index,
            augmented_cells, config, los_cache,
            elevation_provider=elevation_provider,
        )
        if result.is_visible:
            return (ci, si, result.distance_m, result.clearance_m, result.path_loss_db)
        return None

    # ci -> [count, best_dist, best_clear, best_ploss, best_source_id]
    cell_results: Dict[int, list] = {}
    for ci in cells_with_own_source:
        own_source = source_by_h3.get(cand_list[ci].h3_index)
        cell_results[ci] = [1, 0.0, 0.0, 0.0, own_source.source_id if own_source else None]

    max_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_check_pair, p) for p in los_pairs]
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            ci, si, dist_m, clear_m, ploss_db = res
            if ci not in cell_results:
                cell_results[ci] = [0, float('inf'), None, None, None]
            entry = cell_results[ci]
            entry[0] += 1
            if dist_m < entry[1]:
                entry[1] = dist_m
                entry[2] = clear_m
                entry[3] = ploss_db
                entry[4] = src_list[si].source_id

    tx_dbm = config.tx_power_dbm
    gain = 2.0 * config.antenna_gain_dbi
    sens = config.receiver_sensitivity_dbm

    results = []
    for ci, (count, dist, clearance, ploss, closest_sid) in cell_results.items():
        cell = cand_list[ci]
        rx_dbm = None
        is_covered = False
        if ploss is not None:
            rx_dbm = tx_dbm + gain - ploss
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
            'distance_m': dist if dist != float('inf') else None,
            'clearance_m': clearance if clearance is not None and clearance != float('inf') else None,
            'path_loss_db': ploss,
            'received_power_dbm': round(rx_dbm, 2) if rx_dbm is not None else None,
            'is_covered': is_covered,
            'closest_tower_id': closest_sid,
        })

    logger.info("Standalone tower coverage computed: %d covered hexes", len(results))
    return results
