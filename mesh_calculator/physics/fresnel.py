"""
Fresnel zone clearance calculation with earth curvature.

Based on h3_visibility_clearance.sql from the original implementation.
"""
from dataclasses import dataclass
import math
from typing import Dict, Tuple
import h3

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..core.geometry import h3_distance, great_circle_distance

logger = structlog.get_logger(__name__)
_MAX_FRESNEL_OBSTRUCTION_RATIO = 0.4


@dataclass(frozen=True)
class ProfilePoint:
    frac: float
    terrain_elev: float


@dataclass(frozen=True)
class ProfileMetric:
    frac: float
    terrain_elev: float
    d1: float
    d2: float
    line_alt: float
    earth_curv: float
    fresnel_r: float
    geometric_clearance: float
    clearance: float
    obstruction_ratio: float


@dataclass(frozen=True)
class FresnelProfileSummary:
    clearance_m: float
    distance_m: float
    d1_m: float
    d2_m: float
    max_obstruction_ratio: float
    worst_obstruction_frac: float


def _cell_los_coords(cell: H3Cell) -> tuple[float, float]:
    """Return fixed LOS anchor coordinates, falling back to centroid."""
    return (
        float(getattr(cell, "los_lat", cell.lat)),
        float(getattr(cell, "los_lon", cell.lon)),
    )


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
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    cells: Dict[str, H3Cell],
    elevation_provider=None,
) -> list[tuple[float, float]]:
    """
    Build coarse terrain samples along the RF line.

    Endpoints come from already-built cell elevations; intermediate points use
    H3 grid-path centers and provider elevation if available.
    """
    samples: list[ProfilePoint] = [
        ProfilePoint(0.0, float(src_cell.elevation)),
        ProfilePoint(1.0, float(dst_cell.elevation)),
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
                src_lat, src_lon, dst_lat, dst_lon, c.lat, c.lon
            )
            samples.append(ProfilePoint(frac, float(c.elevation)))
            continue
        if not use_provider:
            continue
        try:
            lat, lon = h3.cell_to_latlng(cell_h3)
            elev = float(get_elevation(lat, lon))
        except Exception:
            continue
        frac = _fraction_along_line(
            src_lat, src_lon, dst_lat, dst_lon, lat, lon
        )
        samples.append(ProfilePoint(frac, elev))

    # Merge near-identical fractions conservatively by keeping higher terrain.
    merged: dict[float, float] = {}
    for sample in samples:
        key = round(float(sample.frac), 6)
        prev = merged.get(key)
        if prev is None or sample.terrain_elev > prev:
            merged[key] = float(sample.terrain_elev)
    return [ProfilePoint(frac, elev) for frac, elev in sorted(merged.items(), key=lambda x: x[0])]


def _dense_profile_samples(
    src_cell: H3Cell,
    dst_cell: H3Cell,
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    total_distance_m: float,
    elevation_provider,
    sample_step_m: float,
    max_samples: int,
) -> list[ProfilePoint]:
    """Build dense terrain samples over straight RF line using DEM interpolation."""
    get_elevation = getattr(elevation_provider, "get_elevation_bilinear", None)
    if not callable(get_elevation):
        get_elevation = getattr(elevation_provider, "get_elevation", None)
    if not callable(get_elevation) or total_distance_m <= 0.0:
        return [
            ProfilePoint(0.0, float(src_cell.elevation)),
            ProfilePoint(1.0, float(dst_cell.elevation)),
        ]

    n_samples = max(2, int(total_distance_m / max(sample_step_m, 1.0)))
    n_samples = min(n_samples, max(max_samples, 2))

    samples: list[ProfilePoint] = [
        ProfilePoint(0.0, float(src_cell.elevation)),
        ProfilePoint(1.0, float(dst_cell.elevation)),
    ]
    for i in range(1, n_samples):
        frac = i / n_samples
        lat = src_lat + (dst_lat - src_lat) * frac
        lon = src_lon + (dst_lon - src_lon) * frac
        try:
            elev = float(get_elevation(lat, lon))
        except Exception:
            continue
        samples.append(ProfilePoint(frac, elev))
    return samples


def _merge_profile_points(samples: list[ProfilePoint]) -> list[ProfilePoint]:
    """Merge duplicate fractions conservatively by keeping higher terrain."""
    merged: dict[float, float] = {}
    for sample in samples:
        key = round(float(sample.frac), 6)
        prev = merged.get(key)
        if prev is None or sample.terrain_elev > prev:
            merged[key] = float(sample.terrain_elev)
    return [ProfilePoint(frac, elev) for frac, elev in sorted(merged.items(), key=lambda x: x[0])]


def _evaluate_profile_samples(
    samples: list[ProfilePoint],
    src_height_asl: float,
    dst_height_asl: float,
    total_distance_m: float,
    wavelength_m: float,
    effective_radius_m: float,
) -> list[ProfileMetric]:
    """Evaluate LOS/Fresnel metrics at each terrain sample."""
    metrics: list[ProfileMetric] = []
    for sample in samples:
        d1 = total_distance_m * sample.frac
        d2 = total_distance_m - d1
        line_alt = src_height_asl + (dst_height_asl - src_height_asl) * sample.frac
        earth_curv = (d1 * d2) / (2 * effective_radius_m)
        fresnel_r = 0.0
        if d1 > 0.0 and d2 > 0.0:
            fresnel_r = math.sqrt(max(wavelength_m * d1 * d2 / total_distance_m, 0.0))
        geometric_clearance = line_alt - (sample.terrain_elev + earth_curv)
        clearance = geometric_clearance - fresnel_r
        if fresnel_r > 0.0:
            obstruction_ratio = max(0.0, min(1.0, (fresnel_r - geometric_clearance) / fresnel_r))
        else:
            obstruction_ratio = 0.0 if geometric_clearance >= 0.0 else 1.0
        metrics.append(ProfileMetric(
            frac=float(sample.frac),
            terrain_elev=float(sample.terrain_elev),
            d1=float(d1),
            d2=float(d2),
            line_alt=float(line_alt),
            earth_curv=float(earth_curv),
            fresnel_r=float(fresnel_r),
            geometric_clearance=float(geometric_clearance),
            clearance=float(clearance),
            obstruction_ratio=float(obstruction_ratio),
        ))
    return metrics


def _summarize_profile_metrics(
    metrics: list[ProfileMetric],
    total_distance_m: float,
) -> FresnelProfileSummary:
    """Reduce per-sample metrics to the LOS/path-loss summary."""
    if not metrics:
        return FresnelProfileSummary(0.0, float(total_distance_m), 0.0, float(total_distance_m), 0.0, 0.0)

    worst_clear = min(metrics, key=lambda m: m.clearance)
    interior = [m for m in metrics if 0.0 < m.frac < 1.0]
    worst_ratio_metric = max(interior or metrics, key=lambda m: m.obstruction_ratio)
    return FresnelProfileSummary(
        clearance_m=float(worst_clear.clearance),
        distance_m=float(total_distance_m),
        d1_m=float(worst_clear.d1),
        d2_m=float(worst_clear.d2),
        max_obstruction_ratio=float(worst_ratio_metric.obstruction_ratio),
        worst_obstruction_frac=float(worst_ratio_metric.frac),
    )


def _refine_dense_profile(
    samples: list[ProfilePoint],
    metrics: list[ProfileMetric],
    src_cell: H3Cell,
    dst_cell: H3Cell,
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    total_distance_m: float,
    elevation_provider,
    max_samples: int,
    local_step_m: float = 10.0,
    top_k: int = 3,
) -> list[ProfilePoint]:
    """Refine the worst local dense intervals with finer DEM sampling."""
    if len(samples) < 3 or len(samples) >= max_samples:
        return samples

    get_elevation = getattr(elevation_provider, "get_elevation_bilinear", None)
    if not callable(get_elevation):
        get_elevation = getattr(elevation_provider, "get_elevation", None)
    if not callable(get_elevation):
        return samples

    interior = [(idx, metric) for idx, metric in enumerate(metrics) if 0 < idx < len(metrics) - 1]
    if not interior:
        return samples

    ranked = sorted(
        interior,
        key=lambda item: (item[1].obstruction_ratio, -item[1].clearance),
        reverse=True,
    )[:top_k]

    refined: list[ProfilePoint] = list(samples)
    for idx, _metric in ranked:
        start_frac = samples[max(0, idx - 1)].frac
        end_frac = samples[min(len(samples) - 1, idx + 1)].frac
        if end_frac <= start_frac:
            continue
        interval_m = total_distance_m * (end_frac - start_frac)
        n_steps = max(1, int(interval_m / max(local_step_m, 1.0)))
        for step in range(1, n_steps):
            if len(refined) >= max_samples:
                break
            frac = start_frac + (end_frac - start_frac) * (step / n_steps)
            lat = src_lat + (dst_lat - src_lat) * frac
            lon = src_lon + (dst_lon - src_lon) * frac
            try:
                elev = float(get_elevation(lat, lon))
            except Exception:
                continue
            refined.append(ProfilePoint(float(frac), elev))
        if len(refined) >= max_samples:
            break
    return _merge_profile_points(refined[:max_samples])


def fresnel_obstruction_ratio_accepts(obstruction_ratio: float) -> bool:
    """Return whether the 40% Fresnel obstruction rule accepts this link."""
    return float(obstruction_ratio) <= _MAX_FRESNEL_OBSTRUCTION_RATIO


def max_allowed_fresnel_obstruction_ratio() -> float:
    """Return the hardcoded maximum accepted Fresnel obstruction ratio."""
    return _MAX_FRESNEL_OBSTRUCTION_RATIO


def compute_fresnel_profile_summary(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider=None,
    mast_height_src_m: float | None = None,
    mast_height_dst_m: float | None = None,
) -> FresnelProfileSummary:
    """Calculate coarse-profile clearance and obstruction summary."""
    if h3_src not in cells or h3_dst not in cells:
        raise ValueError(f"Cells not found: {h3_src}, {h3_dst}")

    src_cell = cells[h3_src]
    dst_cell = cells[h3_dst]
    src_mast = config.mast_height_m if mast_height_src_m is None else mast_height_src_m
    dst_mast = config.mast_height_m if mast_height_dst_m is None else mast_height_dst_m
    src_lat, src_lon = _cell_los_coords(src_cell)
    dst_lat, dst_lon = _cell_los_coords(dst_cell)

    if h3_src == h3_dst:
        return FresnelProfileSummary(
            clearance_m=min(src_mast, dst_mast),
            distance_m=0.0,
            d1_m=0.0,
            d2_m=0.0,
            max_obstruction_ratio=0.0,
            worst_obstruction_frac=0.0,
        )

    src_height = src_cell.elevation + src_mast
    dst_height = dst_cell.elevation + dst_mast
    total_distance = great_circle_distance(
        src_lat, src_lon, dst_lat, dst_lon
    )
    if total_distance <= 0:
        return FresnelProfileSummary(
            clearance_m=min(src_mast, dst_mast),
            distance_m=0.0,
            d1_m=0.0,
            d2_m=0.0,
            max_obstruction_ratio=0.0,
            worst_obstruction_frac=0.0,
        )

    samples = _coarse_profile_samples(
        h3_src=h3_src,
        h3_dst=h3_dst,
        src_cell=src_cell,
        dst_cell=dst_cell,
        src_lat=src_lat,
        src_lon=src_lon,
        dst_lat=dst_lat,
        dst_lon=dst_lon,
        cells=cells,
        elevation_provider=elevation_provider,
    )
    metrics = _evaluate_profile_samples(
        samples=samples,
        src_height_asl=src_height,
        dst_height_asl=dst_height,
        total_distance_m=total_distance,
        wavelength_m=config.wavelength_m,
        effective_radius_m=config.effective_earth_radius_m,
    )
    return _summarize_profile_metrics(metrics, total_distance)


def compute_fresnel_profile_summary_dense(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider=None,
    sample_step_m: float | None = None,
    max_samples: int | None = None,
    mast_height_src_m: float | None = None,
    mast_height_dst_m: float | None = None,
) -> FresnelProfileSummary:
    """Calculate dense-profile clearance and obstruction summary with local refinement."""
    if h3_src not in cells or h3_dst not in cells:
        raise ValueError(f"Cells not found: {h3_src}, {h3_dst}")

    src_cell = cells[h3_src]
    dst_cell = cells[h3_dst]
    src_mast = config.mast_height_m if mast_height_src_m is None else mast_height_src_m
    dst_mast = config.mast_height_m if mast_height_dst_m is None else mast_height_dst_m
    src_lat, src_lon = _cell_los_coords(src_cell)
    dst_lat, dst_lon = _cell_los_coords(dst_cell)

    if h3_src == h3_dst:
        return FresnelProfileSummary(
            clearance_m=min(src_mast, dst_mast),
            distance_m=0.0,
            d1_m=0.0,
            d2_m=0.0,
            max_obstruction_ratio=0.0,
            worst_obstruction_frac=0.0,
        )

    src_height = src_cell.elevation + src_mast
    dst_height = dst_cell.elevation + dst_mast
    total_distance = great_circle_distance(
        src_lat, src_lon, dst_lat, dst_lon
    )
    if total_distance <= 0:
        return FresnelProfileSummary(
            clearance_m=min(src_mast, dst_mast),
            distance_m=0.0,
            d1_m=0.0,
            d2_m=0.0,
            max_obstruction_ratio=0.0,
            worst_obstruction_frac=0.0,
        )

    dense_samples = _dense_profile_samples(
        src_cell=src_cell,
        dst_cell=dst_cell,
        src_lat=src_lat,
        src_lon=src_lon,
        dst_lat=dst_lat,
        dst_lon=dst_lon,
        total_distance_m=total_distance,
        elevation_provider=elevation_provider,
        sample_step_m=(
            config.los_dense_sample_step_m if sample_step_m is None else sample_step_m
        ),
        max_samples=(
            config.los_dense_max_samples if max_samples is None else max_samples
        ),
    )
    metrics = _evaluate_profile_samples(
        samples=dense_samples,
        src_height_asl=src_height,
        dst_height_asl=dst_height,
        total_distance_m=total_distance,
        wavelength_m=config.wavelength_m,
        effective_radius_m=config.effective_earth_radius_m,
    )
    summary = _summarize_profile_metrics(metrics, total_distance)
    if not fresnel_obstruction_ratio_accepts(summary.max_obstruction_ratio):
        return summary

    refined_samples = _refine_dense_profile(
        samples=dense_samples,
        metrics=metrics,
        src_cell=src_cell,
        dst_cell=dst_cell,
        src_lat=src_lat,
        src_lon=src_lon,
        dst_lat=dst_lat,
        dst_lon=dst_lon,
        total_distance_m=total_distance,
        elevation_provider=elevation_provider,
        max_samples=(
            config.los_dense_max_samples if max_samples is None else max_samples
        ),
    )
    refined_metrics = _evaluate_profile_samples(
        samples=refined_samples,
        src_height_asl=src_height,
        dst_height_asl=dst_height,
        total_distance_m=total_distance,
        wavelength_m=config.wavelength_m,
        effective_radius_m=config.effective_earth_radius_m,
    )
    return _summarize_profile_metrics(refined_metrics, total_distance)


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
    summary = compute_fresnel_profile_summary(
        h3_src, h3_dst, cells, config,
        elevation_provider=elevation_provider,
        mast_height_src_m=mast_height_src_m,
        mast_height_dst_m=mast_height_dst_m,
    )
    return (summary.clearance_m, summary.distance_m, summary.d1_m, summary.d2_m)


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
    summary = compute_fresnel_profile_summary_dense(
        h3_src, h3_dst, cells, config,
        elevation_provider=elevation_provider,
        sample_step_m=sample_step_m,
        max_samples=max_samples,
        mast_height_src_m=mast_height_src_m,
        mast_height_dst_m=mast_height_dst_m,
    )
    return (summary.clearance_m, summary.distance_m, summary.d1_m, summary.d2_m)


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
