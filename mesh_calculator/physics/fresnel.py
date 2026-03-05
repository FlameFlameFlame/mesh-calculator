"""
Fresnel zone clearance calculation with earth curvature.

Based on h3_visibility_clearance.sql from the original implementation.
"""
import math
from typing import Dict, Tuple
import h3
import numpy as np

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..core.geometry import h3_distance

logger = structlog.get_logger(__name__)


def compute_fresnel_clearance(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider=None,
) -> Tuple[float, float, float, float]:
    """
    Calculate minimum Fresnel clearance between two H3 cells.

    Implements the algorithm from h3_visibility_clearance_compute (lines 116-190).

    Args:
        h3_src: Source H3 cell index
        h3_dst: Destination H3 cell index
        cells: Dictionary of all H3 cells
        config: Mesh configuration

    Returns:
        Tuple of (clearance_m, distance_m, d1_m, d2_m)
        - clearance_m: Worst-case clearance in meters (negative if obstructed)
        - distance_m: Total distance in meters
        - d1_m: Distance from source to worst obstacle
        - d2_m: Distance from worst obstacle to destination

    Reference:
        h3-mesh-placement/functions/h3_visibility_clearance.sql:116-190
    """
    # Get cell data
    if h3_src not in cells or h3_dst not in cells:
        raise ValueError(f"Cells not found: {h3_src}, {h3_dst}")

    src_cell = cells[h3_src]
    dst_cell = cells[h3_dst]

    # Source and destination heights (elevation + mast)
    src_height = src_cell.elevation + config.mast_height_m
    dst_height = dst_cell.elevation + config.mast_height_m

    # Same cell - return mast height as clearance
    if h3_src == h3_dst:
        return (config.mast_height_m, 0.0, 0.0, 0.0)

    # Total distance
    total_distance = h3_distance(h3_src, h3_dst)

    if total_distance <= 0:
        return (config.mast_height_m, 0.0, 0.0, 0.0)

    # Wavelength for Fresnel zone calculation
    wavelength = config.wavelength_m

    # Effective earth radius (accounts for radio refraction)
    effective_radius = config.effective_earth_radius_m

    # Sample terrain along the straight-line RF path between src and dst.
    # Always use h3.grid_path_cells (straight line) — never road corridor cells,
    # which follow roads that curve around ridges and would miss terrain obstacles.
    _FRESNEL_MAX_SAMPLES = 50
    try:
        path_cells = list(h3.grid_path_cells(h3_src, h3_dst))
    except Exception as e:
        logger.warning("Failed to get path cells", error=str(e))
        path_cells = [h3_src, h3_dst]

    if len(path_cells) > _FRESNEL_MAX_SAMPLES:
        step = max(1, len(path_cells) // _FRESNEL_MAX_SAMPLES)
        path_cells = path_cells[::step]
        # Always include the destination cell
        if path_cells[-1] != h3_dst:
            path_cells.append(h3_dst)

    # Precompute src/dst coordinates and direction vector for fraction math.
    # Apply cos(lat) scaling to longitude deltas so the dot-product projection
    # is metric-correct (1° lon ≠ 1° lat at non-equatorial latitudes).
    src_lat, src_lon = src_cell.lat, src_cell.lon
    dst_lat, dst_lon = dst_cell.lat, dst_cell.lon
    _lat_mid = (src_lat + dst_lat) / 2.0
    _cos_lat = math.cos(math.radians(_lat_mid))
    _dx = (dst_lon - src_lon) * _cos_lat
    _dy = dst_lat - src_lat
    _denom = _dx * _dx + _dy * _dy

    # Data-collection loop — unavoidable (dict lookups + H3 calls)
    lats_list, lons_list, elevs_list = [], [], []
    for cell_h3 in path_cells:
        if cell_h3 in cells:
            c = cells[cell_h3]
            lats_list.append(c.lat)
            lons_list.append(c.lon)
            elevs_list.append(c.elevation)
        elif elevation_provider is not None:
            cell_lat, cell_lon = h3.cell_to_latlng(cell_h3)
            lats_list.append(cell_lat)
            lons_list.append(cell_lon)
            elevs_list.append(elevation_provider.get_elevation(cell_lat, cell_lon))

    if not lats_list:
        return (float('inf'), total_distance, total_distance / 2, total_distance / 2)

    # Numpy-vectorized math — replaces per-cell Python arithmetic
    lats = np.array(lats_list, dtype=np.float64)
    lons = np.array(lons_list, dtype=np.float64)
    elevs = np.array(elevs_list, dtype=np.float64)

    if _denom > 1e-12:
        fracs = np.clip(
            ((lons - src_lon) * _cos_lat * _dx + (lats - src_lat) * _dy) / _denom,
            0.0, 1.0
        )
    else:
        fracs = np.zeros(len(lats))

    d1s = total_distance * fracs
    d2s = total_distance - d1s
    line_alts = src_height + (dst_height - src_height) * fracs
    earth_curvs = (d1s * d2s) / (2 * effective_radius)
    valid = (d1s > 0) & (d2s > 0)
    fresnel_rs = np.where(
        valid,
        np.sqrt(np.maximum(wavelength * d1s * d2s / total_distance, 0.0)),
        0.0
    )
    # Clearance = line altitude - (terrain + earth curvature + Fresnel zone)
    clearances = line_alts - (elevs + earth_curvs + fresnel_rs)

    worst_idx = int(np.argmin(clearances))
    return (float(clearances[worst_idx]), total_distance,
            float(d1s[worst_idx]), float(d2s[worst_idx]))


def has_line_of_sight(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider=None,
) -> bool:
    """
    Check if two cells have line-of-sight (LOS).

    Args:
        h3_src: Source H3 cell index
        h3_dst: Destination H3 cell index
        cells: Dictionary of all H3 cells
        config: Mesh configuration

    Returns:
        True if LOS exists (clearance > 0), False otherwise
    """
    # Check distance constraint
    distance = h3_distance(h3_src, h3_dst)
    if distance > config.max_visibility_m:
        return False

    # Calculate clearance
    clearance, _, _, _ = compute_fresnel_clearance(
        h3_src, h3_dst, cells, config,
        elevation_provider=elevation_provider,
    )

    # LOS exists if clearance is positive
    return clearance > 0
