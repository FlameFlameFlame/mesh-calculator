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
    cache: LOSCache = None
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
    # Check cache first
    if cache is not None:
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
        if cache is not None:
            cache.put(
                h3_src, h3_dst,
                config.mast_height_m, config.mast_height_m,
                config.frequency_hz,
                result
            )
        return result

    # Compute Fresnel clearance
    clearance, distance_m, d1, d2 = compute_fresnel_clearance(
        h3_src, h3_dst, cells, config
    )

    # Compute path loss
    path_loss = compute_path_loss(
        distance_m, config.frequency_hz, clearance, d1, d2
    )

    # Create result
    result = LOSResult(
        clearance_m=clearance,
        path_loss_db=path_loss,
        distance_m=distance_m,
        is_visible=(clearance > 0)
    )

    # Cache result
    if cache is not None:
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
    cache: LOSCache = None
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
    result = compute_los(h3_src, h3_dst, cells, config, cache)
    return result.is_visible
