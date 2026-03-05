"""
Line-of-sight (LOS) calculation combining Fresnel clearance and path loss.
"""
from typing import Dict

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..core.geometry import h3_distance
from ..data.cache import LOSCache, LOSResult
from .fresnel import compute_fresnel_clearance, compute_fresnel_clearance_dense
from .path_loss import compute_path_loss

_LOS_VERIFICATION_MODE = "hybrid_accept_verify"


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
    # Same-cell: trivially visible at zero distance, skip path loss calculation
    if h3_src == h3_dst:
        result = LOSResult(
            clearance_m=config.mast_height_m,
            path_loss_db=0.0,
            distance_m=0.0,
            is_visible=True,
        )
        if cache is not None:
            cache.put(
                h3_src, h3_dst,
                config.mast_height_m, config.mast_height_m,
                config.frequency_hz,
                result,
                tx_power_mw=config.tx_power_mw,
                antenna_gain_dbi=config.antenna_gain_dbi,
                receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
                min_fresnel_clearance_m=config.min_fresnel_clearance_m,
                los_dense_sample_step_m=config.los_dense_sample_step_m,
                los_dense_max_samples=config.los_dense_max_samples,
                los_verification_mode=_LOS_VERIFICATION_MODE,
            )
        return result

    # All LOS checks use the straight-line RF path — always cacheable
    use_cache = cache is not None

    # Check cache first
    if use_cache:
        cached = cache.get(
            h3_src, h3_dst,
            config.mast_height_m, config.mast_height_m,
            config.frequency_hz,
            tx_power_mw=config.tx_power_mw,
            antenna_gain_dbi=config.antenna_gain_dbi,
            receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
            min_fresnel_clearance_m=config.min_fresnel_clearance_m,
            los_dense_sample_step_m=config.los_dense_sample_step_m,
            los_dense_max_samples=config.los_dense_max_samples,
            los_verification_mode=_LOS_VERIFICATION_MODE,
        )
        if cached is not None:
            return cached

    # Check distance constraint
    distance = h3_distance(h3_src, h3_dst)
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
                config.mast_height_m, config.mast_height_m,
                config.frequency_hz,
                result,
                tx_power_mw=config.tx_power_mw,
                antenna_gain_dbi=config.antenna_gain_dbi,
                receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
                min_fresnel_clearance_m=config.min_fresnel_clearance_m,
                los_dense_sample_step_m=config.los_dense_sample_step_m,
                los_dense_max_samples=config.los_dense_max_samples,
                los_verification_mode=_LOS_VERIFICATION_MODE,
            )
        return result

    # Compute Fresnel clearance
    clearance, distance_m, d1, d2 = compute_fresnel_clearance(
        h3_src, h3_dst, cells, config,
        elevation_provider=elevation_provider,
    )

    # Compute path loss
    path_loss = compute_path_loss(
        distance_m, config.frequency_hz, clearance, d1, d2
    )

    # Link feasibility is determined by end-to-end link budget.
    # Fresnel clearance still contributes via diffraction loss inside path_loss.
    is_link_budget_ok = (path_loss <= config.link_budget_db)
    if config.min_fresnel_clearance_m is None:
        is_clearance_ok = True
    else:
        is_clearance_ok = (clearance >= config.min_fresnel_clearance_m)

    # Hybrid verification: only dense-sample links that pass coarse acceptance.
    # This removes coarse H3-center false positives while keeping fast rejects.
    if is_link_budget_ok and is_clearance_ok and elevation_provider is not None:
        dense_clearance, dense_distance_m, dense_d1, dense_d2 = compute_fresnel_clearance_dense(
            h3_src, h3_dst, cells, config,
            elevation_provider=elevation_provider,
            sample_step_m=config.los_dense_sample_step_m,
            max_samples=config.los_dense_max_samples,
        )
        dense_path_loss = compute_path_loss(
            dense_distance_m, config.frequency_hz, dense_clearance, dense_d1, dense_d2
        )
        clearance = dense_clearance
        distance_m = dense_distance_m
        path_loss = dense_path_loss
        is_link_budget_ok = (path_loss <= config.link_budget_db)
        if config.min_fresnel_clearance_m is None:
            is_clearance_ok = True
        else:
            is_clearance_ok = (clearance >= config.min_fresnel_clearance_m)

    # Create result
    result = LOSResult(
        clearance_m=clearance,
        path_loss_db=path_loss,
        distance_m=distance_m,
        is_visible=(is_link_budget_ok and is_clearance_ok)
    )

    if use_cache:
        cache.put(
            h3_src, h3_dst,
            config.mast_height_m, config.mast_height_m,
            config.frequency_hz,
            result,
            tx_power_mw=config.tx_power_mw,
            antenna_gain_dbi=config.antenna_gain_dbi,
            receiver_sensitivity_dbm=config.receiver_sensitivity_dbm,
            min_fresnel_clearance_m=config.min_fresnel_clearance_m,
            los_dense_sample_step_m=config.los_dense_sample_step_m,
            los_dense_max_samples=config.los_dense_max_samples,
            los_verification_mode=_LOS_VERIFICATION_MODE,
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
