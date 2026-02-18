"""
Parallel LOS computation using multithreading.
"""
import os
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache, LOSResult
from ..physics.los import compute_los

logger = structlog.get_logger(__name__)


def compute_los_batch(
    pairs: List[Tuple[str, str]],
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None,
    max_workers: int = None,
    elevation_provider=None,
) -> Dict[Tuple[str, str], LOSResult]:
    """
    Compute LOS for multiple cell pairs in parallel.

    Args:
        pairs: List of (h3_src, h3_dst) tuples
        cells: Dictionary of all H3 cells
        config: Mesh configuration
        cache: Optional LOS cache
        max_workers: Number of worker threads (default: CPU count)
        elevation_provider: Optional elevation provider for off-grid cells

    Returns:
        Dictionary mapping (h3_src, h3_dst) to LOSResult
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    results = {}

    # Worker function
    def compute_pair(pair: Tuple[str, str]) -> Tuple[Tuple[str, str], LOSResult]:
        h3_src, h3_dst = pair
        result = compute_los(h3_src, h3_dst, cells, config, cache,
                             elevation_provider=elevation_provider)
        return (pair, result)

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_pair, pair) for pair in pairs]

        for future in as_completed(futures):
            try:
                pair, result = future.result()
                results[pair] = result
            except Exception as e:
                logger.warning("LOS computation failed", error=str(e))

    return results


def compute_los_batch_progress(
    pairs: List[Tuple[str, str]],
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None,
    max_workers: int = None,
    progress_interval: int = 100,
    elevation_provider=None,
) -> Dict[Tuple[str, str], LOSResult]:
    """
    Compute LOS for multiple cell pairs with progress reporting.

    Args:
        pairs: List of (h3_src, h3_dst) tuples
        cells: Dictionary of all H3 cells
        config: Mesh configuration
        cache: Optional LOS cache
        max_workers: Number of worker threads (default: CPU count)
        progress_interval: Report progress every N completions
        elevation_provider: Optional elevation provider for off-grid cells

    Returns:
        Dictionary mapping (h3_src, h3_dst) to LOSResult
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    logger.info("Computing LOS batch", pairs=len(pairs), workers=max_workers)

    results = {}
    completed = 0

    # Worker function
    def compute_pair(pair: Tuple[str, str]) -> Tuple[Tuple[str, str], LOSResult]:
        h3_src, h3_dst = pair
        result = compute_los(h3_src, h3_dst, cells, config, cache,
                             elevation_provider=elevation_provider)
        return (pair, result)

    # Process in parallel with progress
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_pair, pair) for pair in pairs]

        for future in as_completed(futures):
            try:
                pair, result = future.result()
                results[pair] = result

                completed += 1
                if completed % progress_interval == 0:
                    logger.debug("LOS progress",
                                 completed=completed, total=len(pairs),
                                 pct=round(100 * completed / len(pairs), 1))

            except Exception as e:
                logger.warning("LOS computation failed", error=str(e))
                completed += 1

    logger.info("LOS computation complete",
                successful=len(results), total=len(pairs))

    # Log cache stats if available
    if cache is not None:
        stats = cache.stats()
        logger.debug("LOS cache stats",
                     hits=stats['hits'], misses=stats['misses'],
                     hit_rate=f"{stats['hit_rate']:.1%}")

    return results
