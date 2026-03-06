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


def _coarse_profile_samples(
    h3_src: str,
    h3_dst: str,
    src_cell: H3Cell,
    dst_cell: H3Cell,
    cells: Dict[str, H3Cell],
    elevation_provider=None,
) -> list[tuple[float, float]]:
    """
    Build coarse terrain samples along the RF line.

    Endpoints come from already-built cell elevations; intermediate points use
    H3 grid-path centers and provider elevation if available.
    """
    samples: list[tuple[float, float]] = [
        (0.0, float(src_cell.elevation)),
        (1.0, float(dst_cell.elevation)),
    ]

    try:
        path_cells = list(h3.grid_path_cells(h3_src, h3_dst))
    except Exception:
        path_cells = [h3_src, h3_dst]

    get_elevation = getattr(elevation_provider, "get_elevation", None)
    use_provider = callable(get_elevation)

    for cell_h3 in path_cells[1:-1]:
        c = cells.get(cell_h3)
        if c is not None:
            frac = _fraction_along_line(
                src_cell.lat, src_cell.lon, dst_cell.lat, dst_cell.lon, c.lat, c.lon
            )
            samples.append((frac, float(c.elevation)))
            continue
        if not use_provider:
            continue
        try:
            lat, lon = h3.cell_to_latlng(cell_h3)
            elev = float(get_elevation(lat, lon))
        except Exception:
            continue
        frac = _fraction_along_line(
            src_cell.lat, src_cell.lon, dst_cell.lat, dst_cell.lon, lat, lon
        )
        samples.append((frac, elev))

    # Merge near-identical fractions conservatively by keeping higher terrain.
    merged: dict[float, float] = {}
    for frac, elev in samples:
        key = round(float(frac), 6)
        prev = merged.get(key)
        if prev is None or elev > prev:
            merged[key] = float(elev)
    return sorted(merged.items(), key=lambda x: x[0])


def _dense_profile_samples(
    src_cell: H3Cell,
    dst_cell: H3Cell,
    total_distance_m: float,
    elevation_provider,
    sample_step_m: float,
    max_samples: int,
) -> list[tuple[float, float]]:
    """Build dense terrain samples over straight RF line using DEM interpolation."""
    get_elevation = getattr(elevation_provider, "get_elevation", None)
    if not callable(get_elevation) or total_distance_m <= 0.0:
        return [
            (0.0, float(src_cell.elevation)),
            (1.0, float(dst_cell.elevation)),
        ]

    n_samples = max(2, int(total_distance_m / max(sample_step_m, 1.0)))
    n_samples = min(n_samples, max(max_samples, 2))

    samples: list[tuple[float, float]] = [
        (0.0, float(src_cell.elevation)),
        (1.0, float(dst_cell.elevation)),
    ]
    for i in range(1, n_samples):
        frac = i / n_samples
        lat = src_cell.lat + (dst_cell.lat - src_cell.lat) * frac
        lon = src_cell.lon + (dst_cell.lon - src_cell.lon) * frac
        try:
            elev = float(get_elevation(lat, lon))
        except Exception:
            continue
        samples.append((frac, elev))
    return samples


def _min_clearance_over_samples(
    samples: list[tuple[float, float]],
    src_height_asl: float,
    dst_height_asl: float,
    total_distance_m: float,
    wavelength_m: float,
    effective_radius_m: float,
) -> tuple[float, float, float]:
    """Evaluate clearance at each sample and return minimum clearance and distances."""
    min_clearance = float("inf")
    best_d1 = 0.0
    best_d2 = total_distance_m

    for frac, terrain_elev in samples:
        d1 = total_distance_m * frac
        d2 = total_distance_m - d1
        line_alt = src_height_asl + (dst_height_asl - src_height_asl) * frac
        earth_curv = (d1 * d2) / (2 * effective_radius_m)
        fresnel_r = 0.0
        if d1 > 0.0 and d2 > 0.0:
            fresnel_r = math.sqrt(max(wavelength_m * d1 * d2 / total_distance_m, 0.0))
        clearance = line_alt - (terrain_elev + earth_curv + fresnel_r)
        if clearance < min_clearance:
            min_clearance = float(clearance)
            best_d1 = float(d1)
            best_d2 = float(d2)

    return float(min_clearance), float(best_d1), float(best_d2)


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

    samples = _coarse_profile_samples(
        h3_src=h3_src,
        h3_dst=h3_dst,
        src_cell=src_cell,
        dst_cell=dst_cell,
        cells=cells,
        elevation_provider=elevation_provider,
    )
    clearance, d1, d2 = _min_clearance_over_samples(
        samples=samples,
        src_height_asl=src_height,
        dst_height_asl=dst_height,
        total_distance_m=total_distance,
        wavelength_m=config.wavelength_m,
        effective_radius_m=config.effective_earth_radius_m,
    )
    return (clearance, float(total_distance), d1, d2)


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
    if h3_src not in cells or h3_dst not in cells:
        raise ValueError(f"Cells not found: {h3_src}, {h3_dst}")

    src_cell = cells[h3_src]
    dst_cell = cells[h3_dst]

    src_mast = config.mast_height_m if mast_height_src_m is None else mast_height_src_m
    dst_mast = config.mast_height_m if mast_height_dst_m is None else mast_height_dst_m

    if h3_src == h3_dst:
        return (min(src_mast, dst_mast), 0.0, 0.0, 0.0)

    src_height = src_cell.elevation + src_mast
    dst_height = dst_cell.elevation + dst_mast
    total_distance = great_circle_distance(
        src_cell.lat, src_cell.lon, dst_cell.lat, dst_cell.lon
    )
    if total_distance <= 0:
        return (min(src_mast, dst_mast), 0.0, 0.0, 0.0)

    dense_samples = _dense_profile_samples(
        src_cell=src_cell,
        dst_cell=dst_cell,
        total_distance_m=total_distance,
        elevation_provider=elevation_provider,
        sample_step_m=(
            config.los_dense_sample_step_m if sample_step_m is None else sample_step_m
        ),
        max_samples=(
            config.los_dense_max_samples if max_samples is None else max_samples
        ),
    )
    clearance, d1, d2 = _min_clearance_over_samples(
        samples=dense_samples,
        src_height_asl=src_height,
        dst_height_asl=dst_height,
        total_distance_m=total_distance,
        wavelength_m=config.wavelength_m,
        effective_radius_m=config.effective_earth_radius_m,
    )
    return (clearance, float(total_distance), d1, d2)


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
