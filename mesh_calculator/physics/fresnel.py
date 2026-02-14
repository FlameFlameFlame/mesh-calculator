"""
Fresnel zone clearance calculation with earth curvature.

Based on h3_visibility_clearance.sql from the original implementation.
"""
import math
from typing import Dict, Tuple
import h3

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..core.geometry import h3_distance, calculate_line_fraction

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

    # Sample all cells along the path
    try:
        path_cells = list(h3.grid_path_cells(h3_src, h3_dst))
    except Exception as e:
        logger.warning("Failed to get path cells", error=str(e))
        # Fallback: just check endpoints
        path_cells = [h3_src, h3_dst]

    # Calculate clearance at each intermediate point
    worst_clearance = float('inf')
    worst_d1 = total_distance / 2
    worst_d2 = total_distance / 2

    for cell_h3 in path_cells:
        # Get terrain elevation: from grid if available, else from provider
        if cell_h3 in cells:
            terrain_elevation = cells[cell_h3].elevation
        elif elevation_provider is not None:
            cell_lat, cell_lon = h3.cell_to_latlng(cell_h3)
            terrain_elevation = elevation_provider.get_elevation(cell_lat, cell_lon)
        else:
            continue

        # Calculate fractional position along line
        frac = calculate_line_fraction(cell_h3, h3_src, h3_dst)

        # Distances to this point
        d1 = total_distance * frac
        d2 = total_distance * (1 - frac)

        # Ideal line altitude at this fraction
        line_altitude = src_height + (dst_height - src_height) * frac

        # Earth curvature term
        # Accounts for the fact that earth curves away from the line
        earth_curvature = (d1 * d2) / (2 * effective_radius)

        # First Fresnel zone radius at this point
        # F1 = sqrt(wavelength * d1 * d2 / total_distance)
        if d1 > 0 and d2 > 0:
            fresnel_radius = math.sqrt(wavelength * d1 * d2 / total_distance)
        else:
            fresnel_radius = 0.0

        # Clearance = line altitude - (terrain + earth curvature + Fresnel zone)
        # Positive clearance = clear path
        # Negative clearance = obstructed
        clearance = line_altitude - (terrain_elevation + earth_curvature + fresnel_radius)

        # Track worst clearance
        if clearance < worst_clearance:
            worst_clearance = clearance
            worst_d1 = d1
            worst_d2 = d2

    return (worst_clearance, total_distance, worst_d1, worst_d2)


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
