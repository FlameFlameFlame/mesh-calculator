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
    compute_fn=None,
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
    if compute_fn is None:
        compute_fn = compute_los

    if not pairs:
        return {}

    # Preserve input orientation in the result map while avoiding duplicate work.
    pair_aliases: Dict[Tuple[str, str], list[Tuple[str, str]]] = {}
    for pair in pairs:
        key = pair if pair[0] <= pair[1] else (pair[1], pair[0])
        pair_aliases.setdefault(key, []).append(pair)

    unique_pairs = list(pair_aliases.keys())
    batch_size = max(32, len(unique_pairs) // max(max_workers * 8, 1))
    pair_batches = [
        unique_pairs[i:i + batch_size]
        for i in range(0, len(unique_pairs), batch_size)
    ]

    def compute_pair_batch(batch: List[Tuple[str, str]]) -> Dict[Tuple[str, str], LOSResult]:
        out: Dict[Tuple[str, str], LOSResult] = {}
        for h3_src, h3_dst in batch:
            out[(h3_src, h3_dst)] = compute_fn(
                h3_src, h3_dst, cells, config, cache,
                elevation_provider=elevation_provider,
            )
        return out

    canonical_results: Dict[Tuple[str, str], LOSResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_pair_batch, batch) for batch in pair_batches]
        for future in as_completed(futures):
            try:
                canonical_results.update(future.result())
            except Exception as e:
                logger.warning("LOS computation failed: %s", str(e))

    results: Dict[Tuple[str, str], LOSResult] = {}
    for key, aliases in pair_aliases.items():
        result = canonical_results.get(key)
        if result is None:
            continue
        for alias in aliases:
            results[alias] = result
    return results


def compute_los_batch_progress(
    pairs: List[Tuple[str, str]],
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None,
    max_workers: int = None,
    progress_interval: int = 100,
    elevation_provider=None,
    compute_fn=None,
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
    if compute_fn is None:
        compute_fn = compute_los

    logger.info(
        "Computing LOS batch: pairs=%d workers=%d",
        len(pairs),
        max_workers,
    )

    if not pairs:
        return {}

    pair_aliases: Dict[Tuple[str, str], list[Tuple[str, str]]] = {}
    for pair in pairs:
        key = pair if pair[0] <= pair[1] else (pair[1], pair[0])
        pair_aliases.setdefault(key, []).append(pair)

    unique_pairs = list(pair_aliases.keys())
    batch_size = max(32, len(unique_pairs) // max(max_workers * 8, 1))
    pair_batches = [
        unique_pairs[i:i + batch_size]
        for i in range(0, len(unique_pairs), batch_size)
    ]

    completed = 0
    canonical_results: Dict[Tuple[str, str], LOSResult] = {}

    def compute_pair_batch(batch: List[Tuple[str, str]]) -> Dict[Tuple[str, str], LOSResult]:
        out: Dict[Tuple[str, str], LOSResult] = {}
        for h3_src, h3_dst in batch:
            out[(h3_src, h3_dst)] = compute_fn(
                h3_src, h3_dst, cells, config, cache,
                elevation_provider=elevation_provider,
            )
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_pair_batch, batch) for batch in pair_batches]
        for future in as_completed(futures):
            try:
                batch_result = future.result()
                canonical_results.update(batch_result)
                completed += len(batch_result)
                if completed % progress_interval == 0:
                    logger.debug(
                        "LOS progress: completed=%d total=%d pct=%.1f",
                        completed,
                        len(unique_pairs),
                        round(100 * completed / len(unique_pairs), 1),
                    )

            except Exception as e:
                logger.warning("LOS computation failed: %s", str(e))
                completed += batch_size

    logger.info(
        "LOS computation complete: successful=%d total=%d",
        len(canonical_results),
        len(unique_pairs),
    )

    # Log cache stats if available
    if cache is not None:
        stats = cache.stats()
        logger.debug(
            "LOS cache stats: hits=%d misses=%d hit_rate=%.1f%%",
            stats['hits'],
            stats['misses'],
            stats['hit_rate'] * 100.0,
        )

    results: Dict[Tuple[str, str], LOSResult] = {}
    for key, aliases in pair_aliases.items():
        result = canonical_results.get(key)
        if result is None:
            continue
        for alias in aliases:
            results[alias] = result
    return results
