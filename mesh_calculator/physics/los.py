"""
Line-of-sight (LOS) calculation combining Fresnel clearance and path loss.
"""
from typing import Dict

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..core.geometry import h3_distance
from ..data.cache import LOSCache, LOSResult
from .fresnel import compute_fresnel_clearance
from .path_loss import compute_path_loss


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
            )
        return result

    # All LOS checks use the straight-line RF path — always cacheable
    use_cache = cache is not None

    # Check cache first
    if use_cache:
        cached = cache.get(
            h3_src, h3_dst,
            config.mast_height_m, config.mast_height_m,
            config.frequency_hz
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
                result
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

    # Create result
    result = LOSResult(
        clearance_m=clearance,
        path_loss_db=path_loss,
        distance_m=distance_m,
        is_visible=is_link_budget_ok
    )

    if use_cache:
        cache.put(
            h3_src, h3_dst,
            config.mast_height_m, config.mast_height_m,
            config.frequency_hz,
            result
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
