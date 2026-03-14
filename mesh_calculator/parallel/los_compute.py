"""
LOS batch computation.

This module intentionally runs serially. Historical parallel-worker hints are
accepted for backward compatibility and ignored.
"""
import time
from typing import Any, Dict, List, Tuple

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache, LOSResult
from ..physics.los import compute_los

logger = structlog.get_logger(__name__)
_PARALLEL_HINT_WARNED = False


def _warn_if_parallel_hints_requested(
    max_workers: int | None,
    min_pairs_for_parallel: int | None,
) -> None:
    global _PARALLEL_HINT_WARNED
    if _PARALLEL_HINT_WARNED:
        return
    requested_parallel = False
    if max_workers is not None:
        try:
            requested_parallel = int(max_workers) > 1
        except Exception:
            requested_parallel = True
    if min_pairs_for_parallel is not None:
        try:
            requested_parallel = requested_parallel or int(min_pairs_for_parallel) != 32
        except Exception:
            requested_parallel = True
    if requested_parallel:
        logger.warning(
            "Parallel LOS worker hints are deprecated and ignored; running serial LOS batch"
        )
        _PARALLEL_HINT_WARNED = True


def _pair_aliases(
    pairs: List[Tuple[str, str]]
) -> Dict[Tuple[str, str], list[Tuple[str, str]]]:
    pair_aliases: Dict[Tuple[str, str], list[Tuple[str, str]]] = {}
    for pair in pairs:
        key = pair if pair[0] <= pair[1] else (pair[1], pair[0])
        pair_aliases.setdefault(key, []).append(pair)
    return pair_aliases


def _emit_progress(progress_callback, completed: int, total: int) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(int(completed), int(total))
    except Exception:
        logger.debug("LOS progress callback failed", exc_info=True)


def _emit_chunk_progress(chunk_progress_callback, completed_chunks: int, total_chunks: int) -> None:
    if chunk_progress_callback is None:
        return
    try:
        chunk_progress_callback(int(completed_chunks), int(total_chunks))
    except Exception:
        logger.debug("LOS chunk progress callback failed", exc_info=True)


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
    stage: str = "unknown",
) -> Dict[Tuple[str, str], LOSResult]:
    """Compute LOS for multiple cell pairs deterministically (serial)."""
    _warn_if_parallel_hints_requested(max_workers, min_pairs_for_parallel)
    if compute_fn is None:
        compute_fn = compute_los

    if not pairs:
        return {}

    pair_aliases = _pair_aliases(pairs)
    unique_pairs = list(pair_aliases.keys())
    started_at = time.perf_counter()
    canonical_results: Dict[Tuple[str, str], LOSResult] = {}
    failure_details: List[tuple[Tuple[str, str], str]] = []

    for h3_src, h3_dst in unique_pairs:
        try:
            canonical_results[(h3_src, h3_dst)] = compute_fn(
                h3_src,
                h3_dst,
                cells,
                config,
                cache,
                elevation_provider=elevation_provider,
            )
        except Exception as exc:
            failure_details.append(((h3_src, h3_dst), str(exc)))

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
            "stage": stage,
            "pairs_requested": len(pairs),
            "unique_pairs": len(unique_pairs),
            "pairs_computed": len(canonical_results),
            "pairs_failed": len(failure_details),
            "chunk_failures": 0,
            "workers": 1,
            "chunks": 1,
            "batch_size": len(unique_pairs),
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
        "LOS batch diagnostics: stage=%s requested=%d unique=%d computed=%d failed=%d chunks=%d chunk_failures=%d workers=%d elapsed_s=%.4f",
        stage,
        len(pairs),
        len(unique_pairs),
        len(canonical_results),
        len(failure_details),
        1,
        0,
        1,
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
    stage: str = "unknown",
) -> Dict[Tuple[str, str], LOSResult]:
    """Compute LOS serially with optional batch/chunk progress callbacks."""
    _warn_if_parallel_hints_requested(max_workers, min_pairs_for_parallel)
    if compute_fn is None:
        compute_fn = compute_los

    if not pairs:
        return {}

    pair_aliases = _pair_aliases(pairs)
    unique_pairs = list(pair_aliases.keys())
    canonical_results: Dict[Tuple[str, str], LOSResult] = {}
    started_at = time.perf_counter()
    failure_details: List[tuple[Tuple[str, str], str]] = []

    _emit_progress(progress_callback, 0, len(unique_pairs))
    _emit_chunk_progress(chunk_progress_callback, 0, 1)

    completed = 0
    progress_step = max(1, int(progress_interval))
    for h3_src, h3_dst in unique_pairs:
        try:
            canonical_results[(h3_src, h3_dst)] = compute_fn(
                h3_src,
                h3_dst,
                cells,
                config,
                cache,
                elevation_provider=elevation_provider,
            )
        except Exception as exc:
            failure_details.append(((h3_src, h3_dst), str(exc)))
        completed += 1
        if (completed % progress_step) == 0:
            _emit_progress(progress_callback, completed, len(unique_pairs))

    _emit_progress(progress_callback, len(unique_pairs), len(unique_pairs))
    _emit_chunk_progress(chunk_progress_callback, 1, 1)

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
        "LOS computation complete: stage=%s successful=%d total=%d failed=%d workers=%d elapsed_s=%.3f",
        stage,
        len(canonical_results),
        len(unique_pairs),
        len(failure_details),
        1,
        elapsed_s,
    )

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
            "stage": stage,
            "pairs_requested": len(pairs),
            "unique_pairs": len(unique_pairs),
            "pairs_computed": len(canonical_results),
            "pairs_failed": len(failure_details),
            "chunk_failures": 0,
            "workers": 1,
            "chunks": 1,
            "batch_size": len(unique_pairs),
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
