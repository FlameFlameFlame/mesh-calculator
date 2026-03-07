"""
Parallel LOS computation using multithreading.
"""
import os
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache, LOSResult
from ..physics.los import compute_los

logger = structlog.get_logger(__name__)


def _resolve_max_workers(config: MeshConfig, max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    configured = getattr(config, "los_parallel_workers", None)
    if configured is not None:
        return max(1, int(configured))
    return os.cpu_count() or 4


def _compute_pair_batch(
    batch: List[Tuple[str, str]],
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache | None,
    elevation_provider,
    compute_fn,
) -> Dict[Tuple[str, str], LOSResult]:
    out: Dict[Tuple[str, str], LOSResult] = {}
    for h3_src, h3_dst in batch:
        out[(h3_src, h3_dst)] = compute_fn(
            h3_src, h3_dst, cells, config, cache,
            elevation_provider=elevation_provider,
        )
    return out


def _pair_aliases(
    pairs: List[Tuple[str, str]]
) -> Dict[Tuple[str, str], list[Tuple[str, str]]]:
    pair_aliases: Dict[Tuple[str, str], list[Tuple[str, str]]] = {}
    for pair in pairs:
        key = pair if pair[0] <= pair[1] else (pair[1], pair[0])
        pair_aliases.setdefault(key, []).append(pair)
    return pair_aliases


def compute_los_batch(
    pairs: List[Tuple[str, str]],
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None,
    max_workers: int = None,
    min_pairs_for_parallel: int = 32,
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
    max_workers = _resolve_max_workers(config, max_workers)
    if compute_fn is None:
        compute_fn = compute_los

    if not pairs:
        return {}

    pair_aliases = _pair_aliases(pairs)
    unique_pairs = list(pair_aliases.keys())
    started_at = time.perf_counter()
    canonical_results: Dict[Tuple[str, str], LOSResult] = {}
    if len(unique_pairs) < max(1, int(min_pairs_for_parallel)) or max_workers <= 1:
        for h3_src, h3_dst in unique_pairs:
            canonical_results[(h3_src, h3_dst)] = compute_fn(
                h3_src, h3_dst, cells, config, cache,
                elevation_provider=elevation_provider,
            )
        logger.debug(
            "LOS batch executed serially",
            pairs=len(pairs),
            unique_pairs=len(unique_pairs),
            workers=max_workers,
            elapsed_s=round(time.perf_counter() - started_at, 4),
        )
    else:
        batch_size = max(32, len(unique_pairs) // max(max_workers * 8, 1))
        pair_batches = [
            unique_pairs[i:i + batch_size]
            for i in range(0, len(unique_pairs), batch_size)
        ]
        ordered_batches: list[Dict[Tuple[str, str], LOSResult] | None] = [None] * len(pair_batches)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _compute_pair_batch,
                    batch,
                    cells,
                    config,
                    cache,
                    elevation_provider,
                    compute_fn,
                ): idx
                for idx, batch in enumerate(pair_batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_result = future.result()
                    ordered_batches[idx] = batch_result
                    logger.debug(
                        "LOS batch chunk complete",
                        chunk_index=idx,
                        chunk_pairs=len(pair_batches[idx]),
                        workers=max_workers,
                    )
                except Exception as e:
                    logger.warning("LOS computation failed: %s", str(e))
        for batch_result in ordered_batches:
            if batch_result:
                canonical_results.update(batch_result)
        logger.debug(
            "LOS batch executed in parallel",
            pairs=len(pairs),
            unique_pairs=len(unique_pairs),
            workers=max_workers,
            chunks=len(pair_batches),
            batch_size=batch_size,
            elapsed_s=round(time.perf_counter() - started_at, 4),
        )

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
    min_pairs_for_parallel: int = 32,
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
    max_workers = _resolve_max_workers(config, max_workers)
    if compute_fn is None:
        compute_fn = compute_los

    logger.info(
        "Computing LOS batch: pairs=%d workers=%d",
        len(pairs),
        max_workers,
    )

    if not pairs:
        return {}

    pair_aliases = _pair_aliases(pairs)
    unique_pairs = list(pair_aliases.keys())
    canonical_results: Dict[Tuple[str, str], LOSResult] = {}
    started_at = time.perf_counter()
    completed = 0

    if len(unique_pairs) < max(1, int(min_pairs_for_parallel)) or max_workers <= 1:
        for h3_src, h3_dst in unique_pairs:
            canonical_results[(h3_src, h3_dst)] = compute_fn(
                h3_src, h3_dst, cells, config, cache,
                elevation_provider=elevation_provider,
            )
            completed += 1
            if completed % progress_interval == 0:
                logger.debug(
                    "LOS progress: completed=%d total=%d pct=%.1f",
                    completed,
                    len(unique_pairs),
                    round(100 * completed / len(unique_pairs), 1),
                )
    else:
        batch_size = max(32, len(unique_pairs) // max(max_workers * 8, 1))
        pair_batches = [
            unique_pairs[i:i + batch_size]
            for i in range(0, len(unique_pairs), batch_size)
        ]
        ordered_batches: list[Dict[Tuple[str, str], LOSResult] | None] = [None] * len(pair_batches)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _compute_pair_batch,
                    batch,
                    cells,
                    config,
                    cache,
                    elevation_provider,
                    compute_fn,
                ): idx
                for idx, batch in enumerate(pair_batches)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_result = future.result()
                    ordered_batches[idx] = batch_result
                    completed += len(batch_result)
                    logger.debug(
                        "LOS batch chunk complete",
                        chunk_index=idx,
                        chunk_pairs=len(pair_batches[idx]),
                        workers=max_workers,
                    )
                except Exception as e:
                    logger.warning("LOS computation failed: %s", str(e))
                if completed % progress_interval == 0:
                    logger.debug(
                        "LOS progress: completed=%d total=%d pct=%.1f",
                        completed,
                        len(unique_pairs),
                        round(100 * completed / len(unique_pairs), 1),
                    )
        for batch_result in ordered_batches:
            if batch_result:
                canonical_results.update(batch_result)

    logger.info(
        "LOS computation complete: successful=%d total=%d workers=%d elapsed_s=%.3f",
        len(canonical_results),
        len(unique_pairs),
        max_workers,
        (time.perf_counter() - started_at),
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
