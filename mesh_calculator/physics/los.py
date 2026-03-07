"""
Line-of-sight (LOS) calculation combining Fresnel clearance and path loss.
"""
from typing import Dict

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..core.geometry import great_circle_distance, h3_distance
from ..data.cache import LOSCache, LOSResult
from .fresnel import (
    compute_fresnel_profile_summary,
    compute_fresnel_profile_summary_dense,
    fresnel_obstruction_ratio_accepts,
    max_allowed_fresnel_obstruction_ratio,
)
from .path_loss import compute_path_loss
from .path_loss import fspl_only

_LOS_VERIFICATION_MODE = "hybrid_accept_verify"


def _endpoint_mast_height_m(
    h3_idx: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
) -> float:
    """Resolve endpoint mast height: global mast + optional cell-specific offset."""
    cell = cells.get(h3_idx)
    if cell is None:
        return config.mast_height_m
    offset = float(getattr(cell, "antenna_height_offset_m", 0.0) or 0.0)
    return config.mast_height_m + offset


def _endpoint_los_coords(
    h3_idx: str,
    cells: Dict[str, H3Cell],
) -> tuple[float, float]:
    """Resolve fixed LOS anchor coordinates for an endpoint cell."""
    cell = cells.get(h3_idx)
    if cell is None:
        raise ValueError(f"Cell not found: {h3_idx}")
    return (
        float(getattr(cell, "los_lat", cell.lat)),
        float(getattr(cell, "los_lon", cell.lon)),
    )


def compute_los(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None,
    elevation_provider=None,
) -> LOSResult:
    """
    Compute complete LOS result including clearance and path loss.

    Args:
        h3_src: Source H3 cell index
        h3_dst: Destination H3 cell index
        cells: Dictionary of all H3 cells
        config: Mesh configuration
        cache: Optional LOS cache

    Returns:
        LOSResult with clearance, path loss, distance, and visibility
    """
    src_mast_height_m = _endpoint_mast_height_m(h3_src, cells, config)
    dst_mast_height_m = _endpoint_mast_height_m(h3_dst, cells, config)
    src_lat, src_lon = _endpoint_los_coords(h3_src, cells)
    dst_lat, dst_lon = _endpoint_los_coords(h3_dst, cells)
    max_allowed_ratio = max_allowed_fresnel_obstruction_ratio()

    # Same-cell: trivially visible at zero distance, skip path loss calculation
    if h3_src == h3_dst:
        result = LOSResult(
            clearance_m=min(src_mast_height_m, dst_mast_height_m),
            path_loss_db=0.0,
            distance_m=0.0,
            is_visible=True,
        )
        setattr(result, "fresnel_obstruction_ratio", 0.0)
        setattr(result, "max_allowed_fresnel_obstruction_ratio", max_allowed_ratio)
        setattr(result, "fresnel_obstruction_margin_ratio", max_allowed_ratio)
        if cache is not None:
            cache.put(
                h3_src, h3_dst,
                src_mast_height_m, dst_mast_height_m,
                config.frequency_hz,
                result,
                tx_power_mw=config.tx_power_mw,
                antenna_gain_dbi=config.antenna_gain_dbi,
                receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
                min_fresnel_clearance_m=config.min_fresnel_clearance_m,
                los_dense_sample_step_m=config.los_dense_sample_step_m,
                los_dense_max_samples=config.los_dense_max_samples,
                los_verification_mode=_LOS_VERIFICATION_MODE,
                src_lat=src_lat,
                src_lon=src_lon,
                dst_lat=dst_lat,
                dst_lon=dst_lon,
            )
        return result

    # All LOS checks use the straight-line RF path — always cacheable
    use_cache = cache is not None

    # Check cache first
    if use_cache:
        cached = cache.get(
            h3_src, h3_dst,
            src_mast_height_m, dst_mast_height_m,
            config.frequency_hz,
            tx_power_mw=config.tx_power_mw,
            antenna_gain_dbi=config.antenna_gain_dbi,
            receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
            min_fresnel_clearance_m=config.min_fresnel_clearance_m,
            los_dense_sample_step_m=config.los_dense_sample_step_m,
            los_dense_max_samples=config.los_dense_max_samples,
            los_verification_mode=_LOS_VERIFICATION_MODE,
            src_lat=src_lat,
            src_lon=src_lon,
            dst_lat=dst_lat,
            dst_lon=dst_lon,
        )
        if cached is not None:
            return cached

    # Check distance constraint
    distance = great_circle_distance(src_lat, src_lon, dst_lat, dst_lon)
    if distance > config.max_visibility_m:
        result = LOSResult(
            clearance_m=-999.0,
            path_loss_db=999.0,
            distance_m=distance,
            is_visible=False
        )
        if use_cache:
            cache.put(
                h3_src, h3_dst,
                src_mast_height_m, dst_mast_height_m,
                config.frequency_hz,
                result,
                tx_power_mw=config.tx_power_mw,
                antenna_gain_dbi=config.antenna_gain_dbi,
                receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
                min_fresnel_clearance_m=config.min_fresnel_clearance_m,
                los_dense_sample_step_m=config.los_dense_sample_step_m,
                los_dense_max_samples=config.los_dense_max_samples,
                los_verification_mode=_LOS_VERIFICATION_MODE,
                src_lat=src_lat,
                src_lon=src_lon,
                dst_lat=dst_lat,
                dst_lon=dst_lon,
            )
        return result

    # Fast budget prefilter: if ideal free-space loss already exceeds budget,
    # skip Fresnel/profile sampling entirely.
    fspl_db = fspl_only(distance, config.frequency_hz)
    if fspl_db > config.link_budget_db:
        result = LOSResult(
            clearance_m=-999.0,
            path_loss_db=fspl_db,
            distance_m=distance,
            is_visible=False,
        )
        setattr(result, "fresnel_obstruction_ratio", 1.0)
        setattr(result, "max_allowed_fresnel_obstruction_ratio", float(max_allowed_ratio))
        setattr(
            result,
            "fresnel_obstruction_margin_ratio",
            float(max_allowed_ratio - 1.0),
        )
        if use_cache:
            cache.put(
                h3_src, h3_dst,
                src_mast_height_m, dst_mast_height_m,
                config.frequency_hz,
                result,
                tx_power_mw=config.tx_power_mw,
                antenna_gain_dbi=config.antenna_gain_dbi,
                receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
                min_fresnel_clearance_m=config.min_fresnel_clearance_m,
                los_dense_sample_step_m=config.los_dense_sample_step_m,
                los_dense_max_samples=config.los_dense_max_samples,
                los_verification_mode=_LOS_VERIFICATION_MODE,
                src_lat=src_lat,
                src_lon=src_lon,
                dst_lat=dst_lat,
                dst_lon=dst_lon,
            )
        return result

    # Compute coarse-profile Fresnel clearance + obstruction ratio
    coarse_summary = compute_fresnel_profile_summary(
        h3_src, h3_dst, cells, config,
        elevation_provider=elevation_provider,
        mast_height_src_m=src_mast_height_m,
        mast_height_dst_m=dst_mast_height_m,
    )
    clearance = coarse_summary.clearance_m
    distance_m = coarse_summary.distance_m
    d1 = coarse_summary.d1_m
    d2 = coarse_summary.d2_m
    obstruction_ratio = coarse_summary.max_obstruction_ratio

    if distance_m <= 0.0:
        result = LOSResult(
            clearance_m=clearance,
            path_loss_db=0.0,
            distance_m=0.0,
            is_visible=fresnel_obstruction_ratio_accepts(obstruction_ratio),
        )
        setattr(result, "fresnel_obstruction_ratio", float(obstruction_ratio))
        setattr(result, "max_allowed_fresnel_obstruction_ratio", float(max_allowed_ratio))
        setattr(
            result,
            "fresnel_obstruction_margin_ratio",
            float(max_allowed_ratio - obstruction_ratio),
        )
        if use_cache:
            cache.put(
                h3_src, h3_dst,
                src_mast_height_m, dst_mast_height_m,
                config.frequency_hz,
                result,
                tx_power_mw=config.tx_power_mw,
                antenna_gain_dbi=config.antenna_gain_dbi,
                receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
                min_fresnel_clearance_m=config.min_fresnel_clearance_m,
                los_dense_sample_step_m=config.los_dense_sample_step_m,
                los_dense_max_samples=config.los_dense_max_samples,
                los_verification_mode=_LOS_VERIFICATION_MODE,
                src_lat=src_lat,
                src_lon=src_lon,
                dst_lat=dst_lat,
                dst_lon=dst_lon,
            )
        return result

    # Compute path loss
    path_loss = compute_path_loss(
        distance_m, config.frequency_hz, clearance, d1, d2
    )

    is_link_budget_ok = (path_loss <= config.link_budget_db)
    is_fresnel_ok = fresnel_obstruction_ratio_accepts(obstruction_ratio)

    # Hybrid verification: only dense-sample links that pass coarse acceptance.
    # This removes coarse H3-center false positives while keeping fast rejects.
    if is_link_budget_ok and is_fresnel_ok and elevation_provider is not None:
        dense_summary = compute_fresnel_profile_summary_dense(
            h3_src, h3_dst, cells, config,
            elevation_provider=elevation_provider,
            sample_step_m=config.los_dense_sample_step_m,
            max_samples=config.los_dense_max_samples,
            mast_height_src_m=src_mast_height_m,
            mast_height_dst_m=dst_mast_height_m,
        )
        dense_clearance = dense_summary.clearance_m
        dense_distance_m = dense_summary.distance_m
        dense_d1 = dense_summary.d1_m
        dense_d2 = dense_summary.d2_m
        dense_path_loss = compute_path_loss(
            dense_distance_m, config.frequency_hz, dense_clearance, dense_d1, dense_d2
        )
        clearance = dense_clearance
        distance_m = dense_distance_m
        path_loss = dense_path_loss
        obstruction_ratio = dense_summary.max_obstruction_ratio
        is_link_budget_ok = (path_loss <= config.link_budget_db)
        is_fresnel_ok = fresnel_obstruction_ratio_accepts(obstruction_ratio)

    # Create result
    result = LOSResult(
        clearance_m=clearance,
        path_loss_db=path_loss,
        distance_m=distance_m,
        is_visible=(is_link_budget_ok and is_fresnel_ok)
    )
    setattr(result, "fresnel_obstruction_ratio", float(obstruction_ratio))
    setattr(result, "max_allowed_fresnel_obstruction_ratio", float(max_allowed_ratio))
    setattr(
        result,
        "fresnel_obstruction_margin_ratio",
        float(max_allowed_ratio - obstruction_ratio),
    )

    if use_cache:
        cache.put(
            h3_src, h3_dst,
            src_mast_height_m, dst_mast_height_m,
            config.frequency_hz,
            result,
            tx_power_mw=config.tx_power_mw,
            antenna_gain_dbi=config.antenna_gain_dbi,
            receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
            min_fresnel_clearance_m=config.min_fresnel_clearance_m,
            los_dense_sample_step_m=config.los_dense_sample_step_m,
            los_dense_max_samples=config.los_dense_max_samples,
            los_verification_mode=_LOS_VERIFICATION_MODE,
            src_lat=src_lat,
            src_lon=src_lon,
            dst_lat=dst_lat,
            dst_lon=dst_lon,
        )

    return result


def has_los(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None,
    elevation_provider=None,
) -> bool:
    """
    Quick check if two cells have line-of-sight.

    Args:
        h3_src: Source H3 cell index
        h3_dst: Destination H3 cell index
        cells: Dictionary of all H3 cells
        config: Mesh configuration
        cache: Optional LOS cache

    Returns:
        True if LOS exists, False otherwise
    """
    result = compute_los(
        h3_src, h3_dst, cells, config, cache,
        elevation_provider=elevation_provider,
    )
    return result.is_visible
