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
from ..core.geometry import h3_distance, great_circle_distance

logger = structlog.get_logger(__name__)


def _provider_supports(provider, method_name: str) -> bool:
    """Return True only when method is explicitly implemented on provider type."""
    if provider is None:
        return False
    return callable(getattr(type(provider), method_name, None))


def _fraction_along_line(
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    lat: float,
    lon: float,
) -> float:
    """Project point to src->dst line in lat/lon space with cos-lat correction."""
    lat_mid = (src_lat + dst_lat) / 2.0
    cos_lat = math.cos(math.radians(lat_mid))
    dx = (dst_lon - src_lon) * cos_lat
    dy = dst_lat - src_lat
    denom = (dx * dx) + (dy * dy)
    if denom <= 1e-12:
        return 0.0
    frac = (((lon - src_lon) * cos_lat * dx) + ((lat - src_lat) * dy)) / denom
    return max(0.0, min(1.0, frac))


def compute_fresnel_clearance(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider=None,
    mast_height_src_m: float | None = None,
    mast_height_dst_m: float | None = None,
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

    src_mast = config.mast_height_m if mast_height_src_m is None else mast_height_src_m
    dst_mast = config.mast_height_m if mast_height_dst_m is None else mast_height_dst_m

    # Source and destination heights (elevation + endpoint mast)
    src_height = src_cell.elevation + src_mast
    dst_height = dst_cell.elevation + dst_mast

    # Same cell - return mast height as clearance
    if h3_src == h3_dst:
        return (min(src_mast, dst_mast), 0.0, 0.0, 0.0)

    # Total distance
    total_distance = great_circle_distance(
        src_cell.lat, src_cell.lon, dst_cell.lat, dst_cell.lon
    )

    if total_distance <= 0:
        return (min(src_mast, dst_mast), 0.0, 0.0, 0.0)

    # Wavelength for Fresnel zone calculation
    wavelength = config.wavelength_m

    # Effective earth radius (accounts for radio refraction)
    effective_radius = config.effective_earth_radius_m

    peak_elev = max(src_cell.elevation, dst_cell.elevation)
    peak_lat = src_cell.lat if src_cell.elevation >= dst_cell.elevation else dst_cell.lat
    peak_lon = src_cell.lon if src_cell.elevation >= dst_cell.elevation else dst_cell.lon
    frac = _fraction_along_line(
        src_cell.lat, src_cell.lon, dst_cell.lat, dst_cell.lon, peak_lat, peak_lon
    )

    used_provider_peak = False
    if _provider_supports(elevation_provider, "get_line_peak_elevation"):
        try:
            peak_elev, peak_lat, peak_lon, frac = elevation_provider.get_line_peak_elevation(
                src_cell.lat, src_cell.lon, dst_cell.lat, dst_cell.lon
            )
            used_provider_peak = True
        except Exception as e:
            logger.warning("Failed to get line peak elevation", error=str(e))

    if (not used_provider_peak) and elevation_provider is not None and callable(
        getattr(elevation_provider, "get_elevation", None)
    ):
        # Conservative fallback: sample DEM along line and take maximum.
        step_m = max(float(config.los_dense_sample_step_m), 1.0)
        max_samples = max(int(config.los_dense_max_samples), 2)
        n_samples = max(2, int(total_distance / step_m))
        n_samples = min(n_samples, max_samples)
        # Endpoints already come from cell elevations; sample interior points only.
        for i in range(1, n_samples):
            sample_frac = i / n_samples
            sample_lat = src_cell.lat + (dst_cell.lat - src_cell.lat) * sample_frac
            sample_lon = src_cell.lon + (dst_cell.lon - src_cell.lon) * sample_frac
            try:
                sample_elev = float(elevation_provider.get_elevation(sample_lat, sample_lon))
            except Exception:
                continue
            if sample_elev > peak_elev:
                peak_elev = sample_elev
                peak_lat = sample_lat
                peak_lon = sample_lon
                frac = sample_frac
    elif not used_provider_peak:
        # Final fallback to max over H3 grid-path sample points when DEM profile API is unavailable.
        try:
            path_cells = list(h3.grid_path_cells(h3_src, h3_dst))
        except Exception:
            path_cells = [h3_src, h3_dst]
        for cell_h3 in path_cells:
            c = cells.get(cell_h3)
            if c is None:
                continue
            if c.elevation > peak_elev:
                peak_elev = c.elevation
                peak_lat = c.lat
                peak_lon = c.lon
                frac = _fraction_along_line(
                    src_cell.lat, src_cell.lon, dst_cell.lat, dst_cell.lon, peak_lat, peak_lon
                )

    d1 = total_distance * frac
    d2 = total_distance - d1
    line_alt = src_height + (dst_height - src_height) * frac
    earth_curv = (d1 * d2) / (2 * effective_radius)
    fresnel_r = 0.0
    if d1 > 0 and d2 > 0:
        fresnel_r = math.sqrt(max(wavelength * d1 * d2 / total_distance, 0.0))
    clearance = line_alt - (peak_elev + earth_curv + fresnel_r)
    return (float(clearance), float(total_distance), float(d1), float(d2))


def compute_fresnel_clearance_dense(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider=None,
    sample_step_m: float | None = None,
    max_samples: int | None = None,
    mast_height_src_m: float | None = None,
    mast_height_dst_m: float | None = None,
) -> Tuple[float, float, float, float]:
    """
    Calculate minimum Fresnel clearance using dense DEM profile sampling.

    Samples equidistant points along the straight RF line between endpoint
    coordinates instead of H3 cell centers on grid_path.
    """
    # Peak-point conservative model keeps coarse/dense semantics identical.
    return compute_fresnel_clearance(
        h3_src=h3_src,
        h3_dst=h3_dst,
        cells=cells,
        config=config,
        elevation_provider=elevation_provider,
        mast_height_src_m=mast_height_src_m,
        mast_height_dst_m=mast_height_dst_m,
    )


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
