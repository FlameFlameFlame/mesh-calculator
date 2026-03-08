"""
Node placement along corridors with LOS constraints.
"""
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import h3

import structlog

from ..core.grid import H3Cell, resolve_cell_profile
from ..core.config import MeshConfig
from ..core.geometry import great_circle_distance
from ..data.cache import LOSCache, LOSResult
from ..physics.los import has_los, compute_los
from ..physics.fresnel import max_allowed_fresnel_obstruction_ratio
from ..physics.path_loss import fspl_only
from ..parallel.los_compute import compute_los_batch, compute_los_batch_progress
from ..network.graph import MeshSurface, _los_decision_debug

logger = structlog.get_logger(__name__)
_GAP_REPAIR_WIGGLE_ROUNDS = 3


def _record_los_diag(surface: MeshSurface, diagnostics: Optional[dict]) -> None:
    """Best-effort forwarding of LOS batch diagnostics to surface counters."""
    try:
        surface.record_los_batch_diagnostics(diagnostics)
    except Exception:
        logger.debug("Failed to record LOS diagnostics", exc_info=True)


def _cell_elevation(elevation_provider, h3_idx: str, lat: float, lon: float) -> float:
    """Get elevation with backward-compatible provider fallback."""
    if elevation_provider is None:
        return 0.0
    if callable(getattr(type(elevation_provider), "get_h3_cell_max_elevation", None)):
        try:
            return float(elevation_provider.get_h3_cell_max_elevation(h3_idx))
        except Exception:
            pass
    return float(elevation_provider.get_elevation(lat, lon))


def _cell_profile(
    config: MeshConfig,
    elevation_provider,
    h3_idx: str,
    lat: float,
    lon: float,
) -> tuple[float, float, float]:
    """Resolve (elevation, los_lat, los_lon) for dynamically-added cells."""
    return resolve_cell_profile(
        elevation_provider,
        h3_idx,
        lat,
        lon,
        anchor_margin_m=config.cell_anchor_margin_m,
    )


def _terrain_shadow_prefilter_rejects(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    elevation_provider,
) -> bool:
    """
    Fast reject for obviously terrain-blocked links using line-peak elevation.

    Returns True only when the peak terrain point along the line is guaranteed
    to violate the current Fresnel obstruction policy at that exact point.
    """
    if elevation_provider is None or not callable(
        getattr(type(elevation_provider), "get_line_peak_elevation", None)
    ):
        return False
    src = cells.get(h3_src)
    dst = cells.get(h3_dst)
    if src is None or dst is None:
        return False
    src_lat = float(getattr(src, "los_lat", src.lat))
    src_lon = float(getattr(src, "los_lon", src.lon))
    dst_lat = float(getattr(dst, "los_lat", dst.lat))
    dst_lon = float(getattr(dst, "los_lon", dst.lon))
    try:
        peak_elev_m, _peak_lat, _peak_lon, frac = elevation_provider.get_line_peak_elevation(
            src_lat, src_lon, dst_lat, dst_lon
        )
    except Exception:
        return False

    frac = max(0.0, min(1.0, float(frac)))
    distance_m = great_circle_distance(src_lat, src_lon, dst_lat, dst_lon)
    if distance_m <= 0.0:
        return False

    src_asl = float(src.elevation) + float(config.mast_height_m) + float(
        getattr(src, "antenna_height_offset_m", 0.0) or 0.0
    )
    dst_asl = float(dst.elevation) + float(config.mast_height_m) + float(
        getattr(dst, "antenna_height_offset_m", 0.0) or 0.0
    )
    d1 = distance_m * frac
    d2 = distance_m - d1
    line_alt = src_asl + (dst_asl - src_asl) * frac
    earth_curv = (d1 * d2) / (2.0 * float(config.effective_earth_radius_m))
    fresnel_r = 0.0
    if d1 > 0.0 and d2 > 0.0:
        fresnel_r = (max(float(config.wavelength_m) * d1 * d2 / distance_m, 0.0)) ** 0.5
    required_geometric_clearance = fresnel_r * (
        1.0 - float(max_allowed_fresnel_obstruction_ratio())
    )
    geometric_clearance = float(line_alt) - (float(peak_elev_m) + float(earth_curv))
    return geometric_clearance < required_geometric_clearance


def _effective_initial_search_radius_m(config: MeshConfig) -> float:
    """Resolve planner search radius, preserving backward compatibility."""
    if config.optimizer_search_radius_m is not None:
        return max(0.0, float(config.optimizer_search_radius_m))
    return max(0.0, float(config.road_buffer_m))


def _resolve_prefilter_workers(
    config: MeshConfig,
    los_max_workers: Optional[int],
) -> int:
    """Resolve worker count for DP pair prefiltering."""
    if los_max_workers is not None:
        return max(1, int(los_max_workers))
    configured = getattr(config, "los_parallel_workers", None)
    if configured is not None:
        return max(1, int(configured))
    return os.cpu_count() or 4


def _radius_to_ring_m(config: MeshConfig, radius_m: float, minimum: int = 0) -> int:
    """Convert a metric radius to H3 disk ring count."""
    if radius_m <= 0:
        return minimum
    import math
    edge_m = h3.average_hexagon_edge_length(config.h3_resolution, unit='m')
    return max(minimum, math.ceil(radius_m / edge_m))


def _adaptive_cells_within_radius(
    surface: MeshSurface,
    center_h3: str,
    radius_m: float,
    *,
    candidate_cells: Optional[set[str]] = None,
) -> set[str]:
    """Query nearby cells in meters, using GridProvider adaptive mesh when available."""
    provider = surface.elevation_provider
    if provider is not None and callable(
        getattr(type(provider), "adaptive_cells_within_radius", None)
    ):
        try:
            return set(provider.adaptive_cells_within_radius(
                center_h3,
                float(radius_m),
                surface.config.h3_resolution,
                surface.config,
                candidate_cells=candidate_cells,
            ))
        except Exception:
            logger.debug("adaptive_cells_within_radius failed; falling back to grid_disk", exc_info=True)
    ring = _radius_to_ring_m(surface.config, radius_m, minimum=1 if radius_m > 0 else 0)
    try:
        return set(h3.grid_disk(center_h3, ring))
    except Exception:
        # Unit tests sometimes use synthetic non-H3 ids ("cell_1", ...).
        if candidate_cells is not None:
            return set(candidate_cells)
        return {center_h3}


def _append_search_debug_records(
    surface: MeshSurface,
    h3_indices: List[str],
    algorithm: str,
    phase: str,
    attempt_id: int,
    segment_idx: int,
    repair_round: Optional[int],
    search_radius_m: float,
    search_ring: int,
    search_scope: Optional[str] = None,
    step_idx: Optional[int] = None,
) -> None:
    for h3_idx in h3_indices:
        surface.gap_repair_debug.append({
            'h3_index': h3_idx,
            'algorithm': algorithm,
            'phase': phase,
            'attempt_id': attempt_id,
            'segment_idx': segment_idx,
            'repair_round': repair_round,
            'search_radius_m': search_radius_m,
            'search_ring': search_ring,
            'search_scope': search_scope,
            'step_idx': step_idx,
            # Legacy keys kept for compatibility with existing layer tooltips.
            'gap_idx': segment_idx,
            'buffer_ring': search_ring,
        })


def _dp_place_towers_with_meta(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache,
    k: int,
    los_max_workers: Optional[int] = None,
    los_progress_callback=None,
    los_pair_memo: Optional[Dict[tuple[str, str], LOSResult]] = None,
) -> Optional[tuple]:
    """
    Optimal tower placement using MaxMin Bottleneck Path DP.

    Finds a chain of <=k towers from corridor[0] to corridor[-1] that
    maximises the minimum Fresnel clearance across all consecutive links.
    Naturally prefers high-elevation cells (peaks) as relay positions.

    State:  dp[t][i]  = best min-clearance reaching corridor[i] using
                        exactly t towers placed so far
    Transition: dp[t+1][j] = max(dp[t+1][j], min(dp[t][i], clearance(i→j)))
                for all j > i where has_los(i, j) is True

    Args:
        corridor: Ordered list of H3 indices from start to end.
        surface:  MeshSurface (provides cells, config, elevation_provider).
        cache:    LOSCache (may be None).
        k:        Maximum number of towers allowed (including endpoints).

    Returns:
        (chain, best_t) — chain is a list of H3 indices forming the optimal
        placement, best_t is the winning tower count.  Returns None if no
        feasible connected chain exists within k towers.
    """
    if len(corridor) < 2:
        return (list(corridor), len(corridor))

    config = surface.config
    cells = surface.cells
    elevation_provider = surface.elevation_provider
    from ..core.geometry import h3_distance

    n = len(corridor)
    NEG_INF = float('-inf')

    # dp[t][i]: best min-clearance using t towers, with the last tower at position i
    dp = [[NEG_INF] * n for _ in range(k + 1)]
    parent = [[-1] * n for _ in range(k + 1)]

    # Tower 1 placed at start (position 0); no links yet → infinite clearance
    dp[1][0] = float('inf')

    los_results_by_pair = los_pair_memo if los_pair_memo is not None else {}
    feasible_pairs_all: list[tuple[int, int, tuple[str, str]]] = []
    feasible_pairs_by_i: dict[int, list[tuple[int, tuple[str, str]]]] = {}
    total_forward_pairs = 0
    filtered_by_distance = 0
    filtered_by_fspl = 0
    filtered_by_unfit = 0
    filtered_by_shadow = 0
    reused_from_memo = 0
    memo_keys = set(los_results_by_pair.keys())
    total_sources = max(1, n - 1)
    next_progress_pct = 10

    def _scan_source_row(source_i: int) -> dict:
        src_h3 = corridor[source_i]
        row_pairs: list[tuple[int, int, tuple[str, str]]] = []
        row_forward_pairs = 0
        row_filtered_distance = 0
        row_filtered_fspl = 0
        row_filtered_unfit = 0
        row_filtered_shadow = 0
        row_reused = 0
        for j in range(source_i + 1, n):
            dst_h3 = corridor[j]
            row_forward_pairs += 1
            # Early exit: corridor positions are roughly ordered by distance;
            # once we exceed max_visibility_m we can stop.
            dist = h3_distance(src_h3, dst_h3)
            if dist > config.max_visibility_m:
                row_filtered_distance += (n - j)
                break
            pair = (src_h3, dst_h3)
            if dist <= 0.0:
                if pair in memo_keys:
                    row_reused += 1
                row_pairs.append((source_i, j, pair))
                continue

            # Conservative prefilter: if ideal FSPL already fails budget,
            # the full LOS policy cannot pass this pair.
            if fspl_only(dist, config.frequency_hz) > config.link_budget_db:
                row_filtered_fspl += 1
                continue

            # Skip non-endpoint cells that violate constraints
            is_endpoint_j = (j == n - 1)
            if not is_endpoint_j:
                cell_j = cells.get(dst_h3)
                if cell_j and getattr(cell_j, 'is_in_unfit_area', False):
                    row_filtered_unfit += 1
                    continue

            if pair in memo_keys:
                row_reused += 1
                row_pairs.append((source_i, j, pair))
                continue

            if _terrain_shadow_prefilter_rejects(
                src_h3, dst_h3, cells, config, elevation_provider
            ):
                row_filtered_shadow += 1
                continue

            row_pairs.append((source_i, j, pair))
        return {
            "i": source_i,
            "pairs": row_pairs,
            "forward_pairs": row_forward_pairs,
            "filtered_distance": row_filtered_distance,
            "filtered_fspl": row_filtered_fspl,
            "filtered_unfit": row_filtered_unfit,
            "filtered_shadow": row_filtered_shadow,
            "reused": row_reused,
        }

    def _merge_row(row: dict) -> None:
        nonlocal total_forward_pairs
        nonlocal filtered_by_distance
        nonlocal filtered_by_fspl
        nonlocal filtered_by_unfit
        nonlocal filtered_by_shadow
        nonlocal reused_from_memo
        total_forward_pairs += int(row["forward_pairs"])
        filtered_by_distance += int(row["filtered_distance"])
        filtered_by_fspl += int(row["filtered_fspl"])
        filtered_by_unfit += int(row["filtered_unfit"])
        filtered_by_shadow += int(row["filtered_shadow"])
        reused_from_memo += int(row["reused"])
        for i, j, pair in row["pairs"]:
            feasible_pairs_all.append((i, j, pair))
            feasible_pairs_by_i.setdefault(i, []).append((j, pair))

    prefilter_workers = _resolve_prefilter_workers(config, los_max_workers)
    source_indices = list(range(n - 1))
    if prefilter_workers <= 1 or len(source_indices) < 16:
        for i in source_indices:
            _merge_row(_scan_source_row(i))
            progress_pct = int(((i + 1) * 100) / total_sources)
            if progress_pct >= next_progress_pct:
                logger.debug(
                    "DP prefilter progress",
                    progress_pct=progress_pct,
                    source_cells_scanned=i + 1,
                    source_cells_total=total_sources,
                    feasible_pairs=len(feasible_pairs_all),
                    filtered_by_shadow=filtered_by_shadow,
                    filtered_by_fspl=filtered_by_fspl,
                    filtered_by_unfit=filtered_by_unfit,
                    filtered_by_distance=filtered_by_distance,
                    reused_from_memo=reused_from_memo,
                )
                next_progress_pct += 10
    else:
        try:
            rows_by_i: dict[int, dict] = {}
            completed = 0
            with ThreadPoolExecutor(max_workers=prefilter_workers) as executor:
                futures = {executor.submit(_scan_source_row, i): i for i in source_indices}
                for future in as_completed(futures):
                    row = future.result()
                    rows_by_i[row["i"]] = row
                    completed += 1
                    progress_pct = int((completed * 100) / total_sources)
                    if progress_pct >= next_progress_pct:
                        logger.debug(
                            "DP prefilter progress",
                            progress_pct=progress_pct,
                            source_cells_scanned=completed,
                            source_cells_total=total_sources,
                            feasible_pairs=len(feasible_pairs_all),
                            filtered_by_shadow=filtered_by_shadow,
                            filtered_by_fspl=filtered_by_fspl,
                            filtered_by_unfit=filtered_by_unfit,
                            filtered_by_distance=filtered_by_distance,
                            reused_from_memo=reused_from_memo,
                            workers=prefilter_workers,
                        )
                        next_progress_pct += 10
            for i in source_indices:
                row = rows_by_i.get(i)
                if row is not None:
                    _merge_row(row)
        except Exception:
            logger.warning(
                "DP prefilter parallel execution failed; falling back to serial prefilter",
                exc_info=True,
            )
            feasible_pairs_all.clear()
            feasible_pairs_by_i.clear()
            total_forward_pairs = 0
            filtered_by_distance = 0
            filtered_by_fspl = 0
            filtered_by_unfit = 0
            filtered_by_shadow = 0
            reused_from_memo = 0
            next_progress_pct = 10
            for i in source_indices:
                _merge_row(_scan_source_row(i))
                progress_pct = int(((i + 1) * 100) / total_sources)
                if progress_pct >= next_progress_pct:
                    logger.debug(
                        "DP prefilter progress",
                        progress_pct=progress_pct,
                        source_cells_scanned=i + 1,
                        source_cells_total=total_sources,
                        feasible_pairs=len(feasible_pairs_all),
                        filtered_by_shadow=filtered_by_shadow,
                        filtered_by_fspl=filtered_by_fspl,
                        filtered_by_unfit=filtered_by_unfit,
                        filtered_by_distance=filtered_by_distance,
                        reused_from_memo=reused_from_memo,
                    )
                    next_progress_pct += 10

    logger.debug(
        "DP pair prefilter summary",
        corridor_cells=n,
        total_forward_pairs=total_forward_pairs,
        feasible_pairs=len(feasible_pairs_all),
        filtered_by_distance=filtered_by_distance,
        filtered_by_fspl=filtered_by_fspl,
        filtered_by_unfit=filtered_by_unfit,
        filtered_by_shadow=filtered_by_shadow,
        reused_from_memo=reused_from_memo,
    )
    visible_next_by_i: dict[int, list[tuple[int, LOSResult]]] = {}
    precomputed_edges = False
    if feasible_pairs_all:
        precompute_pairs = [pair for _i, _j, pair in feasible_pairs_all]
        missing_precompute_pairs = [pair for pair in precompute_pairs if pair not in los_results_by_pair]
        los_diag: dict = {}
        try:
            if missing_precompute_pairs:
                if los_progress_callback is None:
                    precomputed_results = compute_los_batch(
                        missing_precompute_pairs,
                        cells,
                        config,
                        cache=cache,
                        max_workers=los_max_workers,
                        elevation_provider=elevation_provider,
                        compute_fn=compute_los,
                        diagnostics=los_diag,
                        stage="dp_search",
                    )
                else:
                    precomputed_results = compute_los_batch_progress(
                        missing_precompute_pairs,
                        cells,
                        config,
                        cache=cache,
                        max_workers=los_max_workers,
                        elevation_provider=elevation_provider,
                        compute_fn=compute_los,
                        progress_callback=los_progress_callback,
                        diagnostics=los_diag,
                        stage="dp_search",
                    )
                los_results_by_pair.update(precomputed_results)
                _record_los_diag(surface, los_diag)

            for i, j, pair in feasible_pairs_all:
                los = los_results_by_pair.get(pair)
                if los is None or not los.is_visible:
                    continue
                visible_next_by_i.setdefault(i, []).append((j, los))
            precomputed_edges = True
        except Exception:
            logger.warning(
                "DP LOS precompute failed; falling back to per-layer missing-only batches",
                exc_info=True,
            )

    for t in range(1, k):
        reachable_i = [i for i in range(n - 1) if dp[t][i] != NEG_INF]
        if not reachable_i:
            continue

        if precomputed_edges:
            # Preserve deterministic DP tie behavior by applying transitions in
            # the exact i/j order from feasible_pairs_by_i.
            for i in reachable_i:
                for j, los in visible_next_by_i.get(i, []):
                    link_quality = min(dp[t][i], los.clearance_m)
                    if link_quality > dp[t + 1][j]:
                        dp[t + 1][j] = link_quality
                        parent[t + 1][j] = i
            continue

        # Safety fallback: compute only unseen pairs per layer.
        layer_pairs: list[tuple[str, str]] = []
        for i in reachable_i:
            for _j, pair in feasible_pairs_by_i.get(i, []):
                layer_pairs.append(pair)
        if not layer_pairs:
            continue
        missing_pairs = [pair for pair in layer_pairs if pair not in los_results_by_pair]
        if missing_pairs:
            los_diag: dict = {}
            if los_progress_callback is None:
                missing_results = compute_los_batch(
                    missing_pairs,
                    cells,
                    config,
                    cache=cache,
                    max_workers=los_max_workers,
                    elevation_provider=elevation_provider,
                    compute_fn=compute_los,
                    diagnostics=los_diag,
                    stage="dp_search",
                )
            else:
                missing_results = compute_los_batch_progress(
                    missing_pairs,
                    cells,
                    config,
                    cache=cache,
                    max_workers=los_max_workers,
                    elevation_provider=elevation_provider,
                    compute_fn=compute_los,
                    progress_callback=los_progress_callback,
                    diagnostics=los_diag,
                    stage="dp_search",
                )
            los_results_by_pair.update(missing_results)
            _record_los_diag(surface, los_diag)
        for i in reachable_i:
            for j, pair in feasible_pairs_by_i.get(i, []):
                los = los_results_by_pair.get(pair)
                if los is None or not los.is_visible:
                    continue
                link_quality = min(dp[t][i], los.clearance_m)
                if link_quality > dp[t + 1][j]:
                    dp[t + 1][j] = link_quality
                    parent[t + 1][j] = i

    # Find the best feasible chain that reaches the last position.
    # Prefer fewer towers: pick the smallest t that reaches the endpoint
    # with positive clearance.  Only fall back to the highest-clearance
    # chain (any t) if no chain has positive clearance.
    best_quality = NEG_INF
    best_t = -1
    for t in range(2, k + 1):
        if dp[t][n - 1] > 0:
            # First feasible chain with positive clearance wins (fewest towers)
            best_quality = dp[t][n - 1]
            best_t = t
            break
    if best_t < 0:
        # No positive-clearance chain; fall back to best of any quality
        for t in range(2, k + 1):
            if dp[t][n - 1] > best_quality:
                best_quality = dp[t][n - 1]
                best_t = t

    if best_t < 0:
        return None  # No feasible chain found

    # Reconstruct path via parent pointers
    path_indices = []
    j = n - 1
    for t in range(best_t, 0, -1):
        path_indices.append(j)
        j = parent[t][j]
    path_indices.reverse()

    return ([corridor[i] for i in path_indices], best_t)


def _dp_place_towers(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache,
    k: int,
    los_max_workers: Optional[int] = None,
) -> Optional[List[str]]:
    """
    Thin wrapper around _dp_place_towers_with_meta that discards best_t.

    Returns:
        List of H3 indices (corridor order) forming the optimal chain, or
        None if no feasible connected chain exists within k towers.
    """
    result = _dp_place_towers_with_meta(
        corridor,
        surface,
        cache,
        k,
        los_max_workers=los_max_workers,
    )
    if result is None:
        return None
    return result[0]


def _peak_fallback(
    corridor: List[str],
    cells: Dict[str, H3Cell],
    k: int,
) -> List[str]:
    """
    Fallback: place k towers at highest-elevation corridor cells.

    Always includes both endpoints; fills remaining budget with the
    highest-elevation cells in the corridor.

    Args:
        corridor: Ordered list of H3 indices.
        cells:    Cell lookup dict.
        k:        Maximum number of towers.

    Returns:
        Ordered list of selected H3 indices.
    """
    n = len(corridor)
    sorted_by_elev = sorted(
        range(n),
        key=lambda i: (cells.get(corridor[i]) or H3Cell(
            h3_index=corridor[i], lat=0, lon=0, elevation=0,
        )).elevation,
        reverse=True,
    )
    selected = {0, n - 1}
    for i in sorted_by_elev:
        if len(selected) >= k:
            break
        selected.add(i)
    return [corridor[i] for i in sorted(selected)]


def _prune_redundant(
    chain: List[str],
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache,
) -> List[str]:
    """
    Remove interior towers whose neighbors have LOS to each other.

    The DP maximises minimum clearance and may fill its budget with adjacent
    towers that add no connectivity benefit.  This pass removes them so the
    budget is not wasted on clusters.

    Uses corridor-path LOS (not straight-line) so that towers placed as
    relays around terrain obstacles are not incorrectly pruned because the
    straight-line between their neighbours happens to be clear.

    Args:
        chain:   Ordered list of H3 indices returned by the DP / fallback.
        corridor: The corridor slice used for this segment (to extract sub-paths).
        surface: MeshSurface (cells, config, elevation_provider).
        cache:   LOSCache (may be None).

    Returns:
        Pruned chain (same list object, modified in place).
    """
    corridor_pos: Dict[str, int] = {}
    for idx, h3_idx in enumerate(corridor):
        if h3_idx not in corridor_pos:
            corridor_pos[h3_idx] = idx

    changed = True
    while changed:
        changed = False
        for i in range(len(chain) - 2, 0, -1):
            prev_h3 = chain[i - 1]
            next_h3 = chain[i + 1]
            los = compute_los(
                prev_h3, next_h3,
                surface.cells, surface.config, cache,
                elevation_provider=surface.elevation_provider,
            )
            if los.is_visible:
                removed = chain.pop(i)
                logger.debug("Pruned redundant tower", node=removed)
                changed = True
    return chain


def _expand_segment_buffer(
    segment: List[str],
    segment_set: set,
    ring: int,
    surface: MeshSurface,
) -> int:
    """
    Inject higher-elevation buffer neighbors into segment for a given ring size.

    For each road cell in segment, looks ring-neighbors for cells already in
    surface.cells (pre-loaded buffer).  Keeps only the highest-elevation
    neighbor per road cell and inserts it after that road cell in segment.

    Returns the number of newly injected cells.
    """
    from ..core.geometry import h3_distance as _dist
    cells = surface.cells
    elevation_provider = surface.elevation_provider
    injected = 0
    best_by_road: dict = {}
    edge_m = h3.average_hexagon_edge_length(surface.config.h3_resolution, unit='m')
    radius_m = max(float(ring) * edge_m, 0.0)
    for road_h3 in list(segment_set):
        neighbors = _adaptive_cells_within_radius(
            surface,
            road_h3,
            radius_m,
        )
        for nb in neighbors:
            if nb in segment_set:
                continue
            if nb in cells:
                elev = cells[nb].elevation
            elif elevation_provider is not None:
                lat, lon = h3.cell_to_latlng(nb)
                elev, _, _ = _cell_profile(surface.config, elevation_provider, nb, lat, lon)
            else:
                continue
            closest = min(
                segment_set,
                key=lambda r: _dist(nb, r),
            )
            if (closest not in best_by_road
                    or elev > best_by_road[closest][1]):
                best_by_road[closest] = (nb, elev)

    for road_h3, (nb, elev) in best_by_road.items():
        if nb in segment_set:
            continue
        if nb not in cells and elevation_provider is not None:
            lat, lon = h3.cell_to_latlng(nb)
            from ..core.grid import H3Cell
            elev2, los_lat, los_lon = _cell_profile(
                surface.config, elevation_provider, nb, lat, lon
            )
            cells[nb] = H3Cell(
                h3_index=nb, lat=lat, lon=lon,
                elevation=elev2,
                has_road=False, is_in_boundary=False,
                los_lat=los_lat, los_lon=los_lon,
            )
        try:
            pos = segment.index(road_h3)
        except ValueError:
            continue
        segment.insert(pos + 1, nb)
        segment_set.add(nb)
        injected += 1

    return injected


def _fill_visibility_gaps(
    selected: List[str],
    candidates: List[str],
    max_vis_m: float,
) -> List[str]:
    """
    Insert relay nodes from ``candidates`` to fill any gap between consecutive
    selected nodes that exceeds ``max_vis_m``.

    Uses great-circle distance only (no LOS check) as a lightweight
    connectivity guarantee — the goal is to ensure no two consecutive towers
    are further apart than the radio horizon, so the downstream LOS step has
    a chance to build edges.

    Args:
        selected: Ordered list of H3 indices to check.
        candidates: Pool of H3 indices eligible for insertion.
        max_vis_m: Maximum allowed distance between consecutive nodes (metres).

    Returns:
        New ordered list with gap-filling nodes inserted where needed.
    """
    from ..core.geometry import h3_distance as _dist

    result = list(selected)
    changed = True
    while changed:
        changed = False
        for i in range(len(result) - 1):
            gap = _dist(result[i], result[i + 1])
            if gap <= max_vis_m:
                continue
            # Find the candidate that minimises the larger of the two sub-gaps
            pool = [n for n in candidates if n not in result]
            if not pool:
                break
            best = min(
                pool,
                key=lambda n: max(_dist(result[i], n), _dist(n, result[i + 1])),
            )
            sub_gap = max(_dist(result[i], best), _dist(best, result[i + 1]))
            if sub_gap <= max_vis_m:
                result.insert(i + 1, best)
                logger.warning(
                    "Inserted gap-fill relay node",
                    gap_m=round(gap),
                    sub_gap_m=round(sub_gap),
                    node=best,
                )
                changed = True
                break  # restart scan after insertion
    return result


def _find_broken_gaps(
    chain: List[str],
    surface: MeshSurface,
    cache: LOSCache,
    los_max_workers: Optional[int] = None,
    los_progress_callback=None,
) -> List[int]:
    """
    Return list of chain indices i where chain[i]↔chain[i+1] has no LOS.

    Args:
        chain:    Ordered list of placed tower H3 indices.
        surface:  MeshSurface.
        cache:    LOSCache (may be None).

    Returns:
        List of indices i (into chain) where the pair (chain[i], chain[i+1]) is broken.
    """
    broken = []
    if len(chain) < 2:
        return broken
    pairs = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    los_diag: dict = {}
    if los_progress_callback is None:
        los_results = compute_los_batch(
            pairs,
            surface.cells,
            surface.config,
            cache=cache,
            max_workers=los_max_workers,
            elevation_provider=surface.elevation_provider,
            compute_fn=compute_los,
            diagnostics=los_diag,
            stage="gap_check",
        )
    else:
        los_results = compute_los_batch_progress(
            pairs,
            surface.cells,
            surface.config,
            cache=cache,
            max_workers=los_max_workers,
            elevation_provider=surface.elevation_provider,
            compute_fn=compute_los,
            progress_callback=los_progress_callback,
            diagnostics=los_diag,
            stage="gap_check",
        )
    _record_los_diag(surface, los_diag)
    for i, pair in enumerate(pairs):
        los = los_results.get(pair)
        if los is None or not los.is_visible:
            broken.append(i)
    return broken


def _repair_broken_gaps(
    chain: List[str],
    corridor: List[str],
    corridor_pos: Dict[str, int],
    search_radius_m: float,
    repair_round: int,
    attempt_id: int,
    surface: MeshSurface,
    cache: LOSCache,
    user_budget: int,
    node_meta: Optional[Dict[str, dict]] = None,
    los_max_workers: Optional[int] = None,
    progress_step_callback=None,
    progress_prefix: str = "",
) -> List[str]:
    """
    For each broken gap in chain, locally wiggle the two endpoint towers
    around their original placement and try to restore LOS.

    Args:
        chain:           Current tower chain (modified in-place and returned).
        corridor:        Full corridor (used to register any newly created cells).
        corridor_pos:    Position lookup {h3_idx: position_in_corridor}.
        search_radius_m: Base wiggle diameter step (meters). Local repairs run
                         three rounds and increase diameter by this amount each
                         round; search radius is diameter/2.
        repair_round:    Debug field for metadata compatibility.
        attempt_id:      Placement attempt id.
        surface:         MeshSurface.
        cache:           LOSCache (may be None).
        user_budget:     Unused by local wiggle repair; kept for compatibility.
        node_meta:       Optional dict populated with placement metadata for newly
                         introduced/moved nodes (algorithm='dp_repair').
    Returns:
        Updated chain (same list object).
    """
    from ..core.geometry import h3_distance as _dist

    broken = _find_broken_gaps(
        chain, surface, cache, los_max_workers=los_max_workers
    )
    if not broken:
        return chain

    cells = surface.cells
    elevation_provider = surface.elevation_provider
    max_candidates_per_side = 24

    def _ensure_cell(h3_idx: str) -> bool:
        if h3_idx in cells:
            return True
        if elevation_provider is None:
            return False
        try:
            lat, lon = h3.cell_to_latlng(h3_idx)
            elev, los_lat, los_lon = _cell_profile(
                surface.config, elevation_provider, h3_idx, lat, lon
            )
            cells[h3_idx] = H3Cell(
                h3_index=h3_idx,
                lat=lat,
                lon=lon,
                elevation=elev,
                has_road=False,
                is_in_boundary=True,
                los_lat=los_lat,
                los_lon=los_lon,
            )
        except Exception:
            return False
        if h3_idx not in corridor_pos:
            corridor_pos[h3_idx] = len(corridor)
            corridor.append(h3_idx)
        return True

    def _candidate_pool(center_h3: str, radius_m: float, movable: bool) -> list[str]:
        if not movable:
            return [center_h3]
        raw = set(_adaptive_cells_within_radius(surface, center_h3, radius_m))
        raw.add(center_h3)
        out: list[str] = []
        for h3_idx in raw:
            if not _ensure_cell(h3_idx):
                continue
            c = cells.get(h3_idx)
            if c is None:
                continue
            if getattr(c, 'is_in_unfit_area', False):
                continue
            out.append(h3_idx)
        if center_h3 not in out and _ensure_cell(center_h3):
            out.append(center_h3)
        ranked = sorted(
            set(out),
            key=lambda h: (-cells[h].elevation, _dist(center_h3, h), h),
        )
        return ranked[:max_candidates_per_side]

    for i in broken:
        anchor_a0 = chain[i]
        anchor_b0 = chain[i + 1]
        prev_h3 = chain[i - 1] if i > 0 else None
        next_h3 = chain[i + 2] if (i + 2) < len(chain) else None
        fixed_a = (i == 0) or (anchor_a0 in surface.tower_by_h3)
        fixed_b = ((i + 1) == (len(chain) - 1)) or (anchor_b0 in surface.tower_by_h3)
        repaired = False

        base_diameter_m = float(max(search_radius_m, 0.0))
        for local_round in range(1, _GAP_REPAIR_WIGGLE_ROUNDS + 1):
            diameter_m = base_diameter_m * local_round
            radius_m = diameter_m / 2.0
            if radius_m <= 0.0:
                break
            if progress_step_callback is not None:
                try:
                    progress_step_callback(
                        f"{progress_prefix} Gap repair {i + 1}/{len(broken)} round {local_round}/{_GAP_REPAIR_WIGGLE_ROUNDS} (diameter {int(round(diameter_m))} m)"
                    )
                except Exception:
                    logger.debug("Gap-repair progress callback failed", exc_info=True)
            search_ring = _radius_to_ring_m(surface.config, radius_m, minimum=1)
            cand_a = _candidate_pool(anchor_a0, radius_m, not fixed_a)
            cand_b = _candidate_pool(anchor_b0, radius_m, not fixed_b)
            if not cand_a or not cand_b:
                continue

            _append_search_debug_records(
                surface=surface,
                h3_indices=sorted(set(cand_a + cand_b)),
                algorithm='dp',
                phase='gap_repair',
                attempt_id=attempt_id,
                segment_idx=i,
                repair_round=local_round,
                search_radius_m=radius_m,
                search_ring=search_ring,
                search_scope='gap_repair_subcorridor',
            )

            valid_a = set(cand_a)
            valid_b = set(cand_b)

            if prev_h3 is not None:
                prev_pairs = [(prev_h3, a) for a in cand_a]
                prev_diag: dict = {}
                prev_results = compute_los_batch(
                    prev_pairs,
                    cells,
                    surface.config,
                    cache=cache,
                    max_workers=los_max_workers,
                    elevation_provider=surface.elevation_provider,
                    compute_fn=compute_los,
                    diagnostics=prev_diag,
                    stage="gap_repair_wiggle_prev",
                )
                _record_los_diag(surface, prev_diag)
                valid_a = {
                    a for a in cand_a
                    if (prev_results.get((prev_h3, a)) is not None)
                    and prev_results[(prev_h3, a)].is_visible
                }
            if next_h3 is not None:
                next_pairs = [(b, next_h3) for b in cand_b]
                next_diag: dict = {}
                next_results = compute_los_batch(
                    next_pairs,
                    cells,
                    surface.config,
                    cache=cache,
                    max_workers=los_max_workers,
                    elevation_provider=surface.elevation_provider,
                    compute_fn=compute_los,
                    diagnostics=next_diag,
                    stage="gap_repair_wiggle_next",
                )
                _record_los_diag(surface, next_diag)
                valid_b = {
                    b for b in cand_b
                    if (next_results.get((b, next_h3)) is not None)
                    and next_results[(b, next_h3)].is_visible
                }

            cand_a = [a for a in cand_a if a in valid_a]
            cand_b = [b for b in cand_b if b in valid_b]
            if not cand_a or not cand_b:
                continue

            pair_candidates = [(a, b) for a in cand_a for b in cand_b]
            pair_diag: dict = {}
            pair_results = compute_los_batch(
                pair_candidates,
                cells,
                surface.config,
                cache=cache,
                max_workers=los_max_workers,
                elevation_provider=surface.elevation_provider,
                compute_fn=compute_los,
                diagnostics=pair_diag,
                stage="gap_repair_wiggle_pair",
            )
            _record_los_diag(surface, pair_diag)

            best = None
            best_score = None
            for a, b in pair_candidates:
                los = pair_results.get((a, b))
                if los is None or not los.is_visible:
                    continue
                move_cost = _dist(anchor_a0, a) + _dist(anchor_b0, b)
                score = (
                    float(los.clearance_m),
                    float(-move_cost),
                    float(cells[a].elevation + cells[b].elevation),
                )
                tie = (a, b)
                if best is None or score > best_score or (score == best_score and tie < best):
                    best = tie
                    best_score = score

            if best is None:
                continue

            new_a, new_b = best
            chain[i] = new_a
            chain[i + 1] = new_b
            if node_meta is not None:
                if new_a != anchor_a0 and new_a not in surface.tower_by_h3:
                    node_meta[new_a] = {
                        'algorithm': 'dp_repair',
                        'dp_steps': None,
                        'repair_round': local_round,
                    }
                if new_b != anchor_b0 and new_b not in surface.tower_by_h3:
                    node_meta[new_b] = {
                        'algorithm': 'dp_repair',
                        'dp_steps': None,
                        'repair_round': local_round,
                    }
            logger.info(
                "Gap repair wiggle successful",
                gap_idx=i,
                local_round=local_round,
                radius_m=radius_m,
                moved_a=(new_a != anchor_a0),
                moved_b=(new_b != anchor_b0),
            )
            if progress_step_callback is not None:
                try:
                    progress_step_callback(
                        f"{progress_prefix} Gap repair {i + 1}/{len(broken)} succeeded on round {local_round}/{_GAP_REPAIR_WIGGLE_ROUNDS}"
                    )
                except Exception:
                    logger.debug("Gap-repair progress callback failed", exc_info=True)
            repaired = True
            break

        if not repaired:
            logger.warning(
                "Gap repair wiggle failed",
                gap_idx=i,
                base_radius_m=search_radius_m,
                anchor_a=anchor_a0,
                anchor_b=anchor_b0,
            )
            if progress_step_callback is not None:
                try:
                    progress_step_callback(
                        f"{progress_prefix} Gap repair {i + 1}/{len(broken)} failed after 3 rounds"
                    )
                except Exception:
                    logger.debug("Gap-repair progress callback failed", exc_info=True)
    return chain


def place_nodes_along_corridor(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache = None,
    out_meta: Optional[Dict[str, dict]] = None,
    los_max_workers: Optional[int] = None,
    progress_callback=None,
) -> List[str]:
    """
    Place nodes along a corridor ensuring LOS connectivity.

    Uses MaxMin Bottleneck Path DP to find the globally optimal tower
    placement that maximises the minimum Fresnel clearance across all
    consecutive links while respecting the per-road tower budget
    (config.max_nodes_per_road).

    Existing towers that fall on the corridor are reused as free relay
    points — the corridor is split at each such tower and each segment
    is processed independently.

    Args:
        corridor:  List of H3 cell indices forming the corridor
        surface:   Mesh surface
        cache:     Optional LOS cache
        out_meta:  Optional dict; if provided, populated with
                   {h3_index: {'algorithm': ..., 'dp_steps': ..., 'repair_round': ...}}
                   for every node returned.  Endpoints reused from existing towers
                   are not written (they were placed by an earlier call).

    Returns:
        List of H3 indices where nodes were placed
    """
    if not corridor or len(corridor) < 2:
        return []

    logger.debug("Placing nodes along corridor", corridor_cells=len(corridor))

    config = surface.config
    cells = surface.cells
    from ..core.geometry import h3_distance
    effective_los_workers = (
        config.los_parallel_workers
        if los_max_workers is None
        else los_max_workers
    )
    placement_progress = 0.0

    def _emit_progress(step: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(max(0.0, min(1.0, placement_progress)), step)
        except Exception:
            logger.debug("Placement progress callback failed", exc_info=True)

    def _make_los_progress(step: str):
        state = {'completed': 0}

        def _callback(completed: int, total: int) -> None:
            nonlocal placement_progress
            total_safe = max(int(total), 1)
            completed_i = max(0, min(int(completed), total_safe))
            delta = max(0, completed_i - state['completed'])
            state['completed'] = completed_i
            if delta > 0:
                # Allocate up to ~60% of placement progress to LOS batch advances.
                placement_progress = min(
                    0.98,
                    placement_progress + 0.60 * (float(delta) / float(total_safe)),
                )
                _emit_progress(step)

        return _callback

    def _normalize_radii(values: Optional[List[float]], default_value: float) -> List[float]:
        radii = []
        for value in values or []:
            try:
                radius = float(value)
            except (TypeError, ValueError):
                continue
            if radius < 0:
                continue
            radii.append(radius)
        if not radii:
            radii = [default_value]
        return radii

    def _inject_buffer_candidates(
        working_corridor: List[str],
        search_radius_m: float,
    ) -> tuple[int, List[str], List[str]]:
        corridor_set = set(working_corridor)
        search_ring = _radius_to_ring_m(config, search_radius_m, minimum=0)
        if search_ring <= 0:
            return 0, [], []
        candidate_buffer_cells = []
        pending_cells: dict[str, tuple[float, float, float, float, float]] = {}
        for road_h3 in list(corridor_set):
            neighbors = _adaptive_cells_within_radius(
                surface,
                road_h3,
                search_radius_m,
            )
            for nb in neighbors:
                if nb in corridor_set:
                    continue
                if nb in cells:
                    if not cells[nb].has_road:
                        candidate_buffer_cells.append(nb)
                    continue
                if surface.elevation_provider is None:
                    continue
                try:
                    lat, lon = h3.cell_to_latlng(nb)
                    elev, los_lat, los_lon = _cell_profile(
                        config,
                        surface.elevation_provider,
                        nb,
                        lat,
                        lon,
                    )
                    pending_cells[nb] = (lat, lon, elev, los_lat, los_lon)
                    candidate_buffer_cells.append(nb)
                except Exception:
                    continue
        all_candidate_cells = sorted(set(candidate_buffer_cells))
        best_by_road: dict = {}
        for nb in candidate_buffer_cells:
            pending = pending_cells.get(nb)
            if nb in cells:
                elev = cells[nb].elevation
            elif pending is not None:
                elev = pending[2]
            else:
                continue
            closest = min(corridor_set, key=lambda r: h3_distance(nb, r))
            if closest not in best_by_road or elev > best_by_road[closest][1]:
                best_by_road[closest] = (nb, elev)

        max_candidates = getattr(config, "dp_buffer_candidates_max_per_segment", None)
        if max_candidates is not None:
            try:
                cap = max(0, int(max_candidates))
            except (TypeError, ValueError):
                cap = 0
            if cap > 0 and len(best_by_road) > cap:
                top_entries = sorted(
                    (
                        (road_h3, nb, elev)
                        for road_h3, (nb, elev) in best_by_road.items()
                    ),
                    key=lambda row: (-row[2], row[1], row[0]),
                )[:cap]
                best_by_road = {road_h3: (nb, elev) for road_h3, nb, elev in top_entries}

        injected = 0
        selected_buffer_cells: List[str] = []
        for road_h3, (nb, _elev) in best_by_road.items():
            if nb in corridor_set:
                continue
            if nb not in cells:
                pending = pending_cells.get(nb)
                if pending is not None:
                    lat, lon, elev, los_lat, los_lon = pending
                    cells[nb] = H3Cell(
                        h3_index=nb,
                        lat=lat,
                        lon=lon,
                        elevation=elev,
                        has_road=False,
                        is_in_boundary=True,
                        los_lat=los_lat,
                        los_lon=los_lon,
                    )
                else:
                    continue
            try:
                pos = working_corridor.index(road_h3)
            except ValueError:
                continue
            working_corridor.insert(pos + 1, nb)
            corridor_set.add(nb)
            selected_buffer_cells.append(nb)
            injected += 1
        return injected, all_candidate_cells, sorted(set(selected_buffer_cells))

    def _run_attempt(
        search_radius_m: float,
        attempt_id: int,
        phase_label: str,
        run_gap_repair: bool = False,
    ) -> tuple[List[str], Dict[str, dict], List[str], Dict[str, int], int, int]:
        working_corridor = list(corridor)
        injected_buffer, all_candidate_cells, selected_buffer_cells = (
            _inject_buffer_candidates(working_corridor, search_radius_m)
        )
        if injected_buffer:
            logger.info(
                "Injected %d buffer cells as corridor candidates", injected_buffer
            )
        scope_prefix = 'fallback' if phase_label == 'fallback_initial' else 'initial'
        initial_ring = _radius_to_ring_m(config, search_radius_m, minimum=0)
        if all_candidate_cells:
            _append_search_debug_records(
                surface=surface,
                h3_indices=all_candidate_cells,
                algorithm='dp',
                phase=phase_label,
                attempt_id=attempt_id,
                segment_idx=-1,
                repair_round=None,
                search_radius_m=search_radius_m,
                search_ring=initial_ring,
                search_scope=f'{scope_prefix}_buffer_candidate',
            )
        if selected_buffer_cells:
            _append_search_debug_records(
                surface=surface,
                h3_indices=selected_buffer_cells,
                algorithm='dp',
                phase=phase_label,
                attempt_id=attempt_id,
                segment_idx=-1,
                repair_round=None,
                search_radius_m=search_radius_m,
                search_ring=initial_ring,
                search_scope=f'{scope_prefix}_buffer_selected',
            )

        corridor_pos: Dict[str, int] = {}
        for _idx, _h3 in enumerate(working_corridor):
            if _h3 not in corridor_pos:
                corridor_pos[_h3] = _idx

        existing_tower_indices = [
            i for i, h3_idx in enumerate(working_corridor)
            if h3_idx in surface.tower_by_h3 and i != 0
        ]
        if existing_tower_indices:
            logger.debug("Reusing existing towers as waypoints",
                         count=len(existing_tower_indices))

        last_idx = len(working_corridor) - 1
        boundaries = [0] + existing_tower_indices
        if boundaries[-1] != last_idx:
            boundaries.append(last_idx)

        endpoint_anchors = sum(
            1 for idx in (0, len(working_corridor) - 1)
            if working_corridor[idx] in surface.tower_by_h3
        )
        effective_budget = config.max_towers_per_route + endpoint_anchors

        total_corridor_len = len(working_corridor)
        all_nodes: List[str] = []
        seen: set = set()
        node_meta: Dict[str, dict] = {}

        phase_text = "Initial" if phase_label == "initial" else f"Fallback #{attempt_id}"
        for bi in range(len(boundaries) - 1):
            seg_start = boundaries[bi]
            seg_end = boundaries[bi + 1]
            segment = working_corridor[seg_start:seg_end + 1]

            _append_search_debug_records(
                surface=surface,
                h3_indices=segment,
                algorithm='dp',
                phase=phase_label,
                attempt_id=attempt_id,
                segment_idx=bi,
                repair_round=None,
                search_radius_m=search_radius_m,
                search_ring=initial_ring,
                search_scope=f'{scope_prefix}_corridor',
            )

            seg_len = seg_end - seg_start
            seg_k = max(2, round(
                effective_budget * seg_len / max(total_corridor_len, 1)
            ))

            seg_label = f"{phase_text} segment {bi + 1}/{max(1, len(boundaries) - 1)}"
            seg_result = _dp_place_towers_with_meta(
                segment,
                surface,
                cache,
                seg_k,
                los_max_workers=effective_los_workers,
                los_progress_callback=_make_los_progress(
                    f'{seg_label} • Evaluating LOS chain candidates'
                ),
                los_pair_memo=shared_dp_los_pairs,
            )
            seg_nodes = seg_result[0] if seg_result is not None else None
            seg_best_t = seg_result[1] if seg_result is not None else None

            if seg_nodes is None:
                logger.warning(
                    "DP found no feasible LOS chain — gap repair will handle it",
                    seg_len=len(segment), k=seg_k,
                    start=segment[0], end=segment[-1],
                )
                seg_nodes = [segment[0], segment[-1]]
                seg_best_t = None

            for node_h3 in seg_nodes:
                if node_h3 in surface.tower_by_h3:
                    continue
                if seg_best_t is not None:
                    node_meta[node_h3] = {
                        'algorithm': 'dp',
                        'dp_steps': seg_best_t,
                        'repair_round': None,
                    }
                else:
                    node_meta[node_h3] = {
                        'algorithm': 'endpoint_fallback',
                        'dp_steps': None,
                        'repair_round': None,
                    }

            for seg_h3 in segment:
                if seg_h3 not in corridor_pos:
                    corridor_pos[seg_h3] = len(working_corridor)
                    working_corridor.append(seg_h3)

            for n in seg_nodes:
                if n not in seen:
                    seen.add(n)
                    all_nodes.append(n)

        if run_gap_repair and config.gap_repair_rounds > 0:
            broken = _find_broken_gaps(
                all_nodes,
                surface,
                cache,
                los_max_workers=effective_los_workers,
                los_progress_callback=_make_los_progress(
                    f'{phase_text} • Checking broken LOS gaps'
                ),
            )
            if broken:
                base_buffer_radius_m = max(0.0, float(config.road_buffer_m))
                if base_buffer_radius_m <= 0.0:
                    base_buffer_radius_m = max(0.0, float(search_radius_m))
                logger.info(
                    "Gap repair wiggle: %d broken pair(s), base_radius_m=%.0f",
                    len(broken),
                    base_buffer_radius_m,
                )
                repair_kwargs = {
                    'search_radius_m': base_buffer_radius_m,
                    'repair_round': 1,
                    'attempt_id': attempt_id,
                    'surface': surface,
                    'cache': cache,
                    'user_budget': effective_budget - 2,
                    'node_meta': node_meta,
                    'los_max_workers': effective_los_workers,
                }
                if progress_callback is not None:
                    repair_kwargs['progress_step_callback'] = _emit_progress
                    repair_kwargs['progress_prefix'] = f'{phase_text} •'
                all_nodes = _repair_broken_gaps(
                    all_nodes,
                    working_corridor,
                    corridor_pos,
                    **repair_kwargs,
                )
        pre_fill_count = len(all_nodes)
        all_nodes = _fill_visibility_gaps(all_nodes, working_corridor, config.max_visibility_m)
        if len(all_nodes) > pre_fill_count:
            logger.debug("After gap-fill", count=len(all_nodes))

        broken_after = len(
            _find_broken_gaps(
                all_nodes,
                surface,
                cache,
                los_max_workers=effective_los_workers,
                los_progress_callback=_make_los_progress(
                    f'{phase_text} • Final LOS connectivity check'
                ),
            )
        )
        return all_nodes, node_meta, working_corridor, corridor_pos, broken_after, effective_budget

    def _prune_unreachable_endpoint_fallback_nodes(
        nodes: List[str],
        meta: Dict[str, dict],
    ) -> tuple[List[str], Dict[str, dict], int]:
        """
        Remove unreachable endpoint_fallback artifacts from DP chain.

        Only prunes nodes tagged as endpoint_fallback, excluding any pre-existing
        placed towers/site anchors. Recomputes broken links until stable.
        """
        pruned_total = 0
        while True:
            broken = _find_broken_gaps(
                nodes, surface, cache, los_max_workers=effective_los_workers
            )
            if not broken:
                if pruned_total > 0:
                    logger.info(
                        "Pruned unreachable endpoint_fallback nodes",
                        count=pruned_total,
                    )
                return nodes, meta, 0

            removable: List[str] = []
            for i in broken:
                start_node = nodes[0]
                end_node = nodes[-1]
                for h3_idx in (nodes[i], nodes[i + 1]):
                    if h3_idx in (start_node, end_node):
                        continue
                    node_info = meta.get(h3_idx) or {}
                    if node_info.get('algorithm') != 'endpoint_fallback':
                        continue
                    if h3_idx in surface.tower_by_h3:
                        continue
                    if h3_idx not in removable:
                        removable.append(h3_idx)

            if not removable:
                if pruned_total > 0:
                    logger.info(
                        "Pruned unreachable endpoint_fallback nodes",
                        count=pruned_total,
                    )
                return nodes, meta, len(broken)

            nodes = [n for n in nodes if n not in set(removable)]
            for h3_idx in removable:
                meta.pop(h3_idx, None)
            pruned_total += len(removable)
            if len(nodes) < 2:
                logger.warning(
                    "DP chain reduced below 2 nodes after endpoint fallback pruning"
                )
                return nodes, meta, max(len(
                    _find_broken_gaps(
                        nodes, surface, cache, los_max_workers=effective_los_workers
                    )
                ), 0)

    initial_search_radius_m = _effective_initial_search_radius_m(config)
    shared_dp_los_pairs: Dict[tuple[str, str], LOSResult] = {}
    selected_nodes, selected_meta, selected_working_corridor, selected_corridor_pos, broken_count, selected_budget = _run_attempt(
        search_radius_m=initial_search_radius_m,
        attempt_id=0,
        phase_label='initial',
        run_gap_repair=True,
    )
    selected_search_radius_m = initial_search_radius_m
    selected_attempt_id = 0

    if broken_count > 0:
        fallback_ladder = _normalize_radii(
            config.fallback_initial_search_radius_ladder_m,
            default_value=initial_search_radius_m,
        )
        attempt_id = 1
        for fallback_radius_m in fallback_ladder:
            if fallback_radius_m <= selected_search_radius_m:
                continue
            (
                nodes,
                meta,
                selected_working_corridor,
                selected_corridor_pos,
                broken,
                selected_budget,
            ) = _run_attempt(
                search_radius_m=fallback_radius_m,
                attempt_id=attempt_id,
                phase_label='fallback_initial',
                run_gap_repair=True,
            )
            selected_nodes, selected_meta = nodes, meta
            selected_search_radius_m = fallback_radius_m
            selected_attempt_id = attempt_id
            broken_count = broken
            if broken_count <= 0:
                break
            attempt_id += 1

        selected_nodes, selected_meta, broken_count = _prune_unreachable_endpoint_fallback_nodes(
            selected_nodes,
            selected_meta,
        )

        if broken_count > 0:
            for i in _find_broken_gaps(
                selected_nodes, surface, cache, los_max_workers=effective_los_workers
            ):
                logger.error(
                    "No LOS after fallback attempts",
                    h3_a=selected_nodes[i], h3_b=selected_nodes[i + 1],
                )

    logger.debug("Nodes placed along corridor", count=len(selected_nodes))
    if out_meta is not None:
        out_meta.update(selected_meta)
    return selected_nodes


def optimize_node_selection(
    nodes: List[str],
    max_nodes: int,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None,
    elevation_provider=None,
) -> List[str]:
    """
    Select best subset of nodes respecting limit while maintaining connectivity.

    Iteratively removes the lowest-scored node whose neighbors can still see
    each other (Fix #5: preserves corridor order, Fix #6: maintains chain
    connectivity).

    NOTE: This function is no longer called by place_nodes_along_corridor
    (replaced by the MaxMin DP algorithm), but is retained for external callers
    and tests.

    Args:
        nodes: List of H3 cell indices (in corridor order)
        max_nodes: Maximum number of nodes allowed
        cells: Dictionary of H3 cells
        config: Mesh configuration
        cache: Optional LOS cache
        elevation_provider: Optional elevation provider for terrain lookups

    Returns:
        Optimized list of H3 indices (in corridor order, chain-connected)
    """
    if len(nodes) <= max_nodes:
        return nodes

    if max_nodes <= 2:
        return [nodes[0], nodes[-1]]

    # Pre-compute scores for intermediate nodes
    intermediates = nodes[1:-1]
    node_scores = {}
    for node in intermediates:
        score = 0.0

        cell = cells.get(node)
        if cell:
            score += cell.elevation / 100.0

        visible_count = 0
        for other in intermediates:
            if other != node and has_los(node, other, cells, config, cache,
                                        elevation_provider=elevation_provider):
                visible_count += 1

        score += visible_count
        node_scores[node] = score

    # Iteratively remove lowest-scored node that doesn't break connectivity
    result = list(nodes)
    while len(result) > max_nodes:
        best_to_remove_idx = None
        best_remove_score = float('inf')

        for i in range(1, len(result) - 1):
            node = result[i]
            prev_node = result[i - 1]
            next_node = result[i + 1]

            # Can remove only if prev and next have LOS
            if has_los(prev_node, next_node, cells, config, cache,
                       elevation_provider=elevation_provider):
                if node_scores.get(node, 0) < best_remove_score:
                    best_remove_score = node_scores.get(node, 0)
                    best_to_remove_idx = i

        if best_to_remove_idx is not None:
            removed = result.pop(best_to_remove_idx)
            logger.debug("Removed node during optimization",
                         node=removed, score=best_remove_score)
        else:
            # No LOS-preserving removal possible; force-remove the
            # lowest-scored intermediate node to respect the hard limit.
            best_force_idx = None
            best_force_score = float('inf')
            for i in range(1, len(result) - 1):
                node = result[i]
                s = node_scores.get(node, 0.0)
                if s < best_force_score:
                    best_force_score = s
                    best_force_idx = i

            if best_force_idx is not None:
                removed = result.pop(best_force_idx)
                logger.warning(
                    "Force-removed node to reach tower limit (no LOS-safe removal available)",
                    node=removed, score=best_force_score,
                    current=len(result), target=max_nodes,
                )
            else:
                # Only endpoints remain — can't reduce further
                break

    return result


def install_nodes(
    node_h3_list: List[str],
    surface: MeshSurface,
    source: str = 'corridor',
    placement_meta: Optional[Dict[str, dict]] = None,
):
    """
    Install nodes (towers) at specified H3 cells.

    Args:
        node_h3_list:    List of H3 cell indices
        surface:         Mesh surface
        source:          Source label for towers
        placement_meta:  Optional {h3_index: meta_dict} from place_nodes_along_corridor.
                         When provided, each tower is created with its debug metadata
                         (algorithm, dp_steps, repair_round).
    """
    for h3_idx in node_h3_list:
        if h3_idx in surface.cells:
            meta = placement_meta.get(h3_idx) if placement_meta else None
            surface.place_tower(h3_idx, source=source, placement_meta=meta)


def wire_corridor_edges(
    placed: List[str],
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache,
    los_max_workers: Optional[int] = None,
) -> None:
    """
    Add visibility edges between consecutive towers in the placed chain,
    using corridor-path LOS (not straight-line).

    The DP places towers that are LOS-connected along the road path
    (corridor_cells).  The global update_visibility_edges step uses
    straight-line LOS which may be blocked by terrain.  This function
    registers the corridor-path edges immediately after placement so the
    visibility graph is correct regardless of terrain obstruction on the
    straight-line path.

    Args:
        placed:   Ordered list of H3 indices returned by place_nodes_along_corridor.
        corridor: The trimmed corridor used during placement (to recover sub-paths).
        surface:  MeshSurface (cells, config, elevation_provider, visibility_graph).
        cache:    LOSCache.
    """
    if len(placed) < 2:
        return

    edges_added = 0
    pending_edges = []
    for i in range(len(placed) - 1):
        h3_a = placed[i]
        h3_b = placed[i + 1]

        tower_a = surface.tower_by_h3.get(h3_a)
        tower_b = surface.tower_by_h3.get(h3_b)
        if tower_a is None or tower_b is None:
            continue

        # Skip if edge already exists in the visibility graph
        if surface.visibility_graph.has_edge(tower_a.tower_id, tower_b.tower_id):
            continue

        pending_edges.append((h3_a, h3_b, tower_a, tower_b))

    if not pending_edges:
        return

    pairs = [(h3_a, h3_b) for h3_a, h3_b, _tower_a, _tower_b in pending_edges]
    los_diag: dict = {}
    los_results = compute_los_batch(
        pairs,
        surface.cells,
        surface.config,
        cache=cache,
        max_workers=los_max_workers,
        elevation_provider=surface.elevation_provider,
        compute_fn=compute_los,
        diagnostics=los_diag,
        stage="corridor_wire",
    )
    _record_los_diag(surface, los_diag)
    for h3_a, h3_b, tower_a, tower_b in pending_edges:
        los = los_results.get((h3_a, h3_b))
        if los is None:
            continue
        if los.is_visible:
            surface.visibility_graph.add_visibility_edge(
                tower_a.tower_id, tower_b.tower_id,
                distance_m=los.distance_m,
                clearance_m=los.clearance_m,
                path_loss_db=los.path_loss_db,
                **_los_decision_debug(
                    los,
                    surface.config,
                    edge_origin='corridor_chain',
                ),
            )
            edges_added += 1
        else:
            logger.warning(
                "Corridor-path LOS failed between consecutive placed towers",
                h3_a=h3_a, h3_b=h3_b,
                clearance_m=getattr(los, 'clearance_m', None),
            )

    if edges_added:
        logger.info(
            "Wired %d corridor-path visibility edge(s)", edges_added
        )
