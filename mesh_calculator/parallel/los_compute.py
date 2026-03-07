"""
Parallel LOS computation using multithreading.
"""
import os
import time
from typing import Any, Dict, List, Tuple
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
) -> tuple[Dict[Tuple[str, str], LOSResult], List[tuple[Tuple[str, str], str]]]:
    out: Dict[Tuple[str, str], LOSResult] = {}
    failures: List[tuple[Tuple[str, str], str]] = []
    for h3_src, h3_dst in batch:
        try:
            out[(h3_src, h3_dst)] = compute_fn(
                h3_src, h3_dst, cells, config, cache,
                elevation_provider=elevation_provider,
            )
        except Exception as exc:
            failures.append(((h3_src, h3_dst), str(exc)))
    return out, failures


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
    diagnostics: Dict[str, Any] | None = None,
    strict_failures: bool = False,
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
    failure_details: List[tuple[Tuple[str, str], str]] = []
    chunk_failures = 0
    chunk_count = 1
    batch_size = len(unique_pairs)
    if len(unique_pairs) < max(1, int(min_pairs_for_parallel)) or max_workers <= 1:
        for h3_src, h3_dst in unique_pairs:
            try:
                canonical_results[(h3_src, h3_dst)] = compute_fn(
                    h3_src, h3_dst, cells, config, cache,
                    elevation_provider=elevation_provider,
                )
            except Exception as exc:
                failure_details.append(((h3_src, h3_dst), str(exc)))
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
        chunk_count = len(pair_batches)
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
                    batch_result, batch_failures = future.result()
                    ordered_batches[idx] = batch_result
                    if batch_failures:
                        chunk_failures += 1
                        failure_details.extend(batch_failures)
                        logger.warning(
                            "LOS chunk had per-pair failures: chunk=%d failed_pairs=%d sample_pair=%s sample_error=%s",
                            idx,
                            len(batch_failures),
                            batch_failures[0][0],
                            batch_failures[0][1],
                        )
                    logger.debug(
                        "LOS batch chunk complete",
                        chunk_index=idx,
                        chunk_pairs=len(pair_batches[idx]),
                        workers=max_workers,
                    )
                except Exception as e:
                    chunk_failures += 1
                    logger.warning("LOS chunk failed: chunk=%d error=%s", idx, str(e))
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

    if strict_failures and failure_details:
        sample_pair, sample_error = failure_details[0]
        raise RuntimeError(
            "LOS batch encountered failures "
            f"(failed_pairs={len(failure_details)}, sample_pair={sample_pair}, error={sample_error})"
        )

    if len(canonical_results) + len(failure_details) != len(unique_pairs):
        logger.warning(
            "LOS batch invariant mismatch: unique=%d computed=%d failed=%d",
            len(unique_pairs),
            len(canonical_results),
            len(failure_details),
        )

    results: Dict[Tuple[str, str], LOSResult] = {}
    for key, aliases in pair_aliases.items():
        result = canonical_results.get(key)
        if result is None:
            continue
        for alias in aliases:
            results[alias] = result

    elapsed_s = round(time.perf_counter() - started_at, 4)
    if diagnostics is not None:
        diagnostics.update({
            "pairs_requested": len(pairs),
            "unique_pairs": len(unique_pairs),
            "pairs_computed": len(canonical_results),
            "pairs_failed": len(failure_details),
            "chunk_failures": chunk_failures,
            "workers": max_workers,
            "chunks": chunk_count,
            "batch_size": batch_size,
            "elapsed_s": elapsed_s,
        })
        if failure_details:
            diagnostics["failure_samples"] = [
                {
                    "pair": [pair[0], pair[1]],
                    "error": error,
                }
                for pair, error in failure_details[:5]
            ]
    logger.debug(
        "LOS batch diagnostics: requested=%d unique=%d computed=%d failed=%d chunks=%d chunk_failures=%d workers=%d elapsed_s=%.4f",
        len(pairs),
        len(unique_pairs),
        len(canonical_results),
        len(failure_details),
        chunk_count,
        chunk_failures,
        max_workers,
        elapsed_s,
    )
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
    diagnostics: Dict[str, Any] | None = None,
    strict_failures: bool = False,
    progress_callback=None,
    chunk_progress_callback=None,
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
        progress_callback: Optional callback(completed, total) for live progress
        chunk_progress_callback: Optional callback(completed_chunks, total_chunks)

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
    failure_details: List[tuple[Tuple[str, str], str]] = []
    chunk_failures = 0
    chunk_count = 1
    batch_size = len(unique_pairs)
    completed_chunks = 0

    def _emit_batch_progress() -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(int(completed), int(len(unique_pairs)))
        except Exception:
            logger.debug("LOS progress callback failed", exc_info=True)

    def _emit_chunk_progress() -> None:
        if chunk_progress_callback is None:
            return
        try:
            chunk_progress_callback(int(completed_chunks), int(chunk_count))
        except Exception:
            logger.debug("LOS chunk progress callback failed", exc_info=True)

    _emit_batch_progress()
    _emit_chunk_progress()

    if len(unique_pairs) < max(1, int(min_pairs_for_parallel)) or max_workers <= 1:
        completed_chunks = 1
        for h3_src, h3_dst in unique_pairs:
            completed += 1
            try:
                canonical_results[(h3_src, h3_dst)] = compute_fn(
                    h3_src, h3_dst, cells, config, cache,
                    elevation_provider=elevation_provider,
                )
            except Exception as exc:
                failure_details.append(((h3_src, h3_dst), str(exc)))
            if completed % progress_interval == 0:
                logger.debug(
                    "LOS progress: completed=%d total=%d pct=%.1f",
                    completed,
                    len(unique_pairs),
                    round(100 * completed / len(unique_pairs), 1),
                )
                _emit_batch_progress()
        _emit_chunk_progress()
    else:
        batch_size = max(32, len(unique_pairs) // max(max_workers * 8, 1))
        pair_batches = [
            unique_pairs[i:i + batch_size]
            for i in range(0, len(unique_pairs), batch_size)
        ]
        chunk_count = len(pair_batches)
        completed_chunks = 0
        _emit_chunk_progress()
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
                    batch_result, batch_failures = future.result()
                    ordered_batches[idx] = batch_result
                    completed += len(batch_result) + len(batch_failures)
                    if batch_failures:
                        chunk_failures += 1
                        failure_details.extend(batch_failures)
                        logger.warning(
                            "LOS chunk had per-pair failures: chunk=%d failed_pairs=%d sample_pair=%s sample_error=%s",
                            idx,
                            len(batch_failures),
                            batch_failures[0][0],
                            batch_failures[0][1],
                        )
                    logger.debug(
                        "LOS batch chunk complete",
                        chunk_index=idx,
                        chunk_pairs=len(pair_batches[idx]),
                        workers=max_workers,
                    )
                except Exception as e:
                    chunk_failures += 1
                    logger.warning("LOS chunk failed: chunk=%d error=%s", idx, str(e))
                completed_chunks += 1
                _emit_chunk_progress()
                if completed % progress_interval == 0:
                    logger.debug(
                        "LOS progress: completed=%d total=%d pct=%.1f",
                        completed,
                        len(unique_pairs),
                        round(100 * completed / len(unique_pairs), 1),
                    )
                    _emit_batch_progress()
        for batch_result in ordered_batches:
            if batch_result:
                canonical_results.update(batch_result)

    completed = len(unique_pairs)
    _emit_batch_progress()
    _emit_chunk_progress()

    if strict_failures and failure_details:
        sample_pair, sample_error = failure_details[0]
        raise RuntimeError(
            "LOS batch encountered failures "
            f"(failed_pairs={len(failure_details)}, sample_pair={sample_pair}, error={sample_error})"
        )

    if len(canonical_results) + len(failure_details) != len(unique_pairs):
        logger.warning(
            "LOS batch invariant mismatch: unique=%d computed=%d failed=%d",
            len(unique_pairs),
            len(canonical_results),
            len(failure_details),
        )

    elapsed_s = time.perf_counter() - started_at
    logger.info(
        "LOS computation complete: successful=%d total=%d failed=%d workers=%d elapsed_s=%.3f",
        len(canonical_results),
        len(unique_pairs),
        len(failure_details),
        max_workers,
        elapsed_s,
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
    if diagnostics is not None:
        diagnostics.update({
            "pairs_requested": len(pairs),
            "unique_pairs": len(unique_pairs),
            "pairs_computed": len(canonical_results),
            "pairs_failed": len(failure_details),
            "chunk_failures": chunk_failures,
            "workers": max_workers,
            "chunks": chunk_count,
            "batch_size": batch_size,
            "elapsed_s": round(elapsed_s, 4),
        })
        if failure_details:
            diagnostics["failure_samples"] = [
                {
                    "pair": [pair[0], pair[1]],
                    "error": error,
                }
                for pair, error in failure_details[:5]
            ]
    return results
