"""
Node placement along corridors with LOS constraints.
"""
from typing import List, Dict, Optional
import h3

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache
from ..physics.los import has_los, compute_los
from ..network.graph import MeshSurface

logger = structlog.get_logger(__name__)


def _dp_place_towers_with_meta(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache,
    k: int,
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

    for t in range(1, k):
        for i in range(n - 1):
            if dp[t][i] == NEG_INF:
                continue
            for j in range(i + 1, n):
                # Early exit: corridor positions are roughly ordered by distance;
                # once we exceed max_visibility_m we can stop.
                dist = h3_distance(corridor[i], corridor[j])
                if dist > config.max_visibility_m:
                    break

                # Skip non-endpoint cells that violate constraints
                is_endpoint_j = (j == n - 1)
                if not is_endpoint_j:
                    cell_j = cells.get(corridor[j])
                    if cell_j and getattr(cell_j, 'is_in_unfit_area', False):
                        continue

                los = compute_los(
                    corridor[i], corridor[j],
                    cells, config, cache,
                    elevation_provider=elevation_provider,
                    corridor_cells=corridor[i:j+1],
                )
                if los.is_visible:
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
) -> Optional[List[str]]:
    """
    Thin wrapper around _dp_place_towers_with_meta that discards best_t.

    Returns:
        List of H3 indices (corridor order) forming the optimal chain, or
        None if no feasible connected chain exists within k towers.
    """
    result = _dp_place_towers_with_meta(corridor, surface, cache, k)
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
            pos_prev = corridor_pos.get(prev_h3)
            pos_next = corridor_pos.get(next_h3)
            if pos_prev is not None and pos_next is not None:
                lo, hi = (pos_prev, pos_next) if pos_prev <= pos_next else (pos_next, pos_prev)
                corridor_cells_seg = corridor[lo:hi + 1]
            else:
                corridor_cells_seg = None
            los = compute_los(
                prev_h3, next_h3,
                surface.cells, surface.config, cache,
                elevation_provider=surface.elevation_provider,
                corridor_cells=corridor_cells_seg,
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
    for road_h3 in list(segment_set):
        for nb in h3.grid_disk(road_h3, ring):
            if nb in segment_set:
                continue
            if nb in cells:
                elev = cells[nb].elevation
            elif elevation_provider is not None:
                lat, lon = h3.cell_to_latlng(nb)
                elev = elevation_provider.get_elevation(lat, lon)
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
            cells[nb] = H3Cell(
                h3_index=nb, lat=lat, lon=lon,
                elevation=elev,
                has_road=False, is_in_boundary=False,
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
    corridor: List[str],
    corridor_pos: Dict[str, int],
    surface: MeshSurface,
    cache: LOSCache,
) -> List[int]:
    """
    Return list of chain indices i where chain[i]↔chain[i+1] has no corridor-path LOS.

    Args:
        chain:        Ordered list of placed tower H3 indices.
        corridor:     Full corridor used during placement.
        corridor_pos: Position lookup {h3_idx: position_in_corridor}.
        surface:      MeshSurface.
        cache:        LOSCache (may be None).

    Returns:
        List of indices i (into chain) where the pair (chain[i], chain[i+1]) is broken.
    """
    broken = []
    for i in range(len(chain) - 1):
        h3_a, h3_b = chain[i], chain[i + 1]
        pos_a = corridor_pos.get(h3_a)
        pos_b = corridor_pos.get(h3_b)
        if pos_a is not None and pos_b is not None:
            lo, hi = (pos_a, pos_b) if pos_a <= pos_b else (pos_b, pos_a)
            corridor_cells_seg = corridor[lo:hi + 1]
        else:
            corridor_cells_seg = None
        los = compute_los(
            h3_a, h3_b,
            surface.cells, surface.config, cache,
            elevation_provider=surface.elevation_provider,
            corridor_cells=corridor_cells_seg,
        )
        if not los.is_visible:
            broken.append(i)
    return broken


def _repair_broken_gaps(
    chain: List[str],
    corridor: List[str],
    corridor_pos: Dict[str, int],
    base_ring: int,
    repair_round: int,
    surface: MeshSurface,
    cache: LOSCache,
    k: int,
    node_meta: Optional[Dict[str, dict]] = None,
) -> List[str]:
    """
    For each broken gap in chain, expand the sub-corridor between surrounding
    anchors and re-run DP to find a relay path around the obstacle.

    Args:
        chain:         Current tower chain (modified in place and returned).
        corridor:      Full corridor (with buffer cells already injected).
        corridor_pos:  Position lookup {h3_idx: position_in_corridor}.
        base_ring:     Initial buffer ring size from config.
        repair_round:  1-based round number; ring expands to base_ring*(round+1).
        surface:       MeshSurface.
        cache:         LOSCache (may be None).
        k:             Max towers for DP (uses effective_budget).
        node_meta:     Optional dict to populate with placement metadata for
                       newly introduced nodes (algorithm='dp_repair', repair_round=r).

    Returns:
        Updated chain (same list object).
    """
    broken = _find_broken_gaps(chain, corridor, corridor_pos, surface, cache)
    if not broken:
        return chain

    new_ring = base_ring * (repair_round + 1)  # 2x, 3x, 4x on rounds 1/2/3
    # Process in reverse so splices don't shift subsequent broken indices
    for i in reversed(broken):
        anchor_a = chain[i]
        anchor_b = chain[i + 1]
        pos_a = corridor_pos.get(anchor_a)
        pos_b = corridor_pos.get(anchor_b)
        if pos_a is None or pos_b is None:
            logger.warning(
                "Cannot find anchor positions for gap repair",
                repair_round=repair_round, gap_idx=i,
            )
            continue
        lo, hi = (pos_a, pos_b) if pos_a <= pos_b else (pos_b, pos_a)
        sub_corridor = list(corridor[lo:hi + 1])
        sub_set = set(sub_corridor)
        try:
            _expand_segment_buffer(sub_corridor, sub_set, new_ring, surface)
        except Exception:
            pass
        result = _dp_place_towers_with_meta(sub_corridor, surface, cache, k)
        if result is None:
            logger.warning(
                "Gap repair DP failed",
                repair_round=repair_round, gap_idx=i,
                anchor_a=anchor_a, anchor_b=anchor_b,
            )
            continue
        new_seg, best_t = result
        # Splice new_seg into chain replacing only the broken pair.
        # anchor_a == new_seg[0], anchor_b == new_seg[-1].
        chain[i:i + 2] = new_seg
        # Register any newly introduced cells into corridor_pos so subsequent
        # repair rounds can locate them as anchors.
        for h3_cell in new_seg:
            if h3_cell not in corridor_pos:
                corridor_pos[h3_cell] = len(corridor)
                corridor.append(h3_cell)
            # Tag interior nodes introduced by this repair (not anchors)
            if node_meta is not None and h3_cell not in (anchor_a, anchor_b):
                node_meta[h3_cell] = {
                    'algorithm': 'dp_repair',
                    'dp_steps': best_t,
                    'repair_round': repair_round,
                }
        logger.info(
            "Gap repair successful",
            repair_round=repair_round, gap_idx=i, new_nodes=len(new_seg),
        )
    return chain


def place_nodes_along_corridor(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache = None,
    out_meta: Optional[Dict[str, dict]] = None,
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
    corridor_set = set(corridor)

    # Inject buffer cells: for each road corridor cell, add the highest-elevation
    # neighbor from the road buffer (cells already in surface.cells with has_road=False)
    # at the corridor position closest to that neighbor.
    # This lets the DP route through elevated terrain adjacent to the road.
    buffer_ring = config.road_buffer_rings if hasattr(config, 'road_buffer_rings') else (
        max(1, round(config.road_buffer_m / h3.average_hexagon_edge_length(
            config.h3_resolution, unit='m'))) if config.road_buffer_m > 0 else 0
    )
    if buffer_ring > 0:
        candidate_buffer_cells = []
        for road_h3 in list(corridor_set):
            neighbors = h3.grid_disk(road_h3, buffer_ring)
            for nb in neighbors:
                if nb not in corridor_set and nb in cells and not cells[nb].has_road:
                    candidate_buffer_cells.append(nb)
        # For each road cell, keep only the highest-elevation buffer neighbor
        # to avoid inflating corridor length with many flat buffer cells.
        best_by_road: dict = {}
        for nb in candidate_buffer_cells:
            # Find the road cell this buffer cell is closest to
            closest = min(corridor_set, key=lambda r: h3_distance(nb, r))
            elev = cells[nb].elevation
            if closest not in best_by_road or elev > best_by_road[closest][1]:
                best_by_road[closest] = (nb, elev)

        injected_buffer = 0
        for road_h3, (nb, _elev) in best_by_road.items():
            if nb in corridor_set:
                continue
            # Insert after the road cell it belongs to
            try:
                pos = corridor.index(road_h3)
            except ValueError:
                continue
            corridor.insert(pos + 1, nb)
            corridor_set.add(nb)
            injected_buffer += 1

        if injected_buffer:
            logger.info(
                "Injected %d buffer cells as corridor candidates", injected_buffer
            )

    # Build position lookup once (after buffer injection) for Phase 2 repair.
    corridor_pos: Dict[str, int] = {}
    for _idx, _h3 in enumerate(corridor):
        if _h3 not in corridor_pos:
            corridor_pos[_h3] = _idx

    # Find existing towers on this corridor and use them as free waypoints.
    # Split the corridor into segments separated by existing towers.
    # Exclude position 0: it is always the start boundary, so including it
    # would create a duplicate [0, 0, …] producing a useless 1-cell segment.
    existing_tower_indices = [
        i for i, h3_idx in enumerate(corridor)
        if h3_idx in surface.tower_by_h3 and i != 0
    ]

    if existing_tower_indices:
        logger.debug("Reusing existing towers as waypoints",
                     count=len(existing_tower_indices))

    # Build strictly-increasing segment boundaries: [0, t1, t2, ..., last]
    last_idx = len(corridor) - 1
    boundaries = [0] + existing_tower_indices
    if boundaries[-1] != last_idx:
        boundaries.append(last_idx)

    # Pre-placed anchor towers at corridor endpoints are "free" — don't count
    # against the user's max_towers_per_route budget.
    endpoint_anchors = sum(
        1 for idx in (0, len(corridor) - 1)
        if corridor[idx] in surface.tower_by_h3
    )
    effective_budget = config.max_towers_per_route + endpoint_anchors

    total_corridor_len = len(corridor)
    all_nodes: List[str] = []
    seen: set = set()
    # Tracks placement metadata for each node placed in this call.
    node_meta: Dict[str, dict] = {}

    for bi in range(len(boundaries) - 1):
        seg_start = boundaries[bi]
        seg_end = boundaries[bi + 1]
        segment = corridor[seg_start:seg_end + 1]

        # Allocate tower budget proportionally to segment length.
        # This prevents a short early segment from consuming the full budget
        # when the corridor is split at existing tower waypoints.
        seg_len = seg_end - seg_start
        seg_k = max(2, round(
            effective_budget * seg_len / max(total_corridor_len, 1)
        ))

        seg_result = _dp_place_towers_with_meta(segment, surface, cache, seg_k)
        seg_nodes = seg_result[0] if seg_result is not None else None
        seg_best_t = seg_result[1] if seg_result is not None else None

        # If DP fails, retry up to 3 times with progressively wider buffer
        # rings injected into the segment — this finds elevated relay cells
        # that give LOS across terrain the road itself cannot clear.
        if seg_nodes is None:
            segment_set = set(segment)
            for attempt in range(1, 4):
                try:
                    added = _expand_segment_buffer(
                        segment, segment_set,
                        buffer_ring + attempt, surface,
                    )
                except Exception:
                    added = 0
                logger.warning(
                    "DP failed; retrying with wider buffer",
                    attempt=attempt,
                    added_cells=added,
                    seg_len=len(segment),
                )
                seg_result = _dp_place_towers_with_meta(
                    segment, surface, cache, seg_k,
                )
                if seg_result is not None:
                    seg_nodes, seg_best_t = seg_result
                    break

        if seg_nodes is None:
            logger.error(
                "DP found no feasible LOS chain after 3 buffer expansions"
                " — skipping segment",
                seg_len=len(segment),
                k=seg_k,
                start=segment[0],
                end=segment[-1],
            )
            # Include only endpoints so the corridor is not completely broken
            seg_nodes = [segment[0], segment[-1]]
            seg_best_t = None

        # Tag placement metadata for nodes in this segment.
        # Endpoints are existing anchors (already placed) — skip them.
        for node_h3 in seg_nodes:
            if node_h3 in surface.tower_by_h3:
                continue  # reused existing tower, already has its own meta
            if seg_best_t is not None:
                node_meta[node_h3] = {'algorithm': 'dp', 'dp_steps': seg_best_t,
                                      'repair_round': None}
            else:
                node_meta[node_h3] = {'algorithm': 'endpoint_fallback',
                                      'dp_steps': None, 'repair_round': None}

        # Sync corridor_pos with any buffer cells injected into segment
        # during the retry loop — they may have been chosen by DP and will
        # appear in all_nodes, so they must be findable in corridor_pos.
        for seg_h3 in segment:
            if seg_h3 not in corridor_pos:
                corridor_pos[seg_h3] = len(corridor)
                corridor.append(seg_h3)

        for n in seg_nodes:
            if n not in seen:
                seen.add(n)
                all_nodes.append(n)

    logger.debug("Nodes placed along corridor (DP)", count=len(all_nodes))

    # Phase 2: targeted gap repair.
    # Find consecutive pairs with no corridor-path LOS and re-run DP on the
    # sub-corridor between their surrounding anchors with a wider buffer ring.
    for repair_round in range(1, 4):
        broken = _find_broken_gaps(
            all_nodes, corridor, corridor_pos, surface, cache,
        )
        if not broken:
            break
        logger.info(
            "Gap repair round %d: %d broken pair(s)",
            repair_round, len(broken),
        )
        all_nodes = _repair_broken_gaps(
            all_nodes, corridor, corridor_pos,
            buffer_ring, repair_round, surface, cache, effective_budget,
            node_meta=node_meta,
        )
    else:
        still_broken = _find_broken_gaps(
            all_nodes, corridor, corridor_pos, surface, cache,
        )
        for i in still_broken:
            logger.error(
                "No LOS after 3 gap repair rounds",
                h3_a=all_nodes[i], h3_b=all_nodes[i + 1],
            )

    # Gap-fill: ensure no consecutive pair exceeds max_visibility_m
    pre_fill_count = len(all_nodes)
    all_nodes = _fill_visibility_gaps(all_nodes, corridor, config.max_visibility_m)
    if len(all_nodes) > pre_fill_count:
        logger.debug("After gap-fill", count=len(all_nodes))

    if out_meta is not None:
        out_meta.update(node_meta)

    return all_nodes


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

    # Build a position lookup in the corridor for fast sub-path extraction.
    corridor_pos: Dict[str, int] = {}
    for idx, h3_idx in enumerate(corridor):
        if h3_idx not in corridor_pos:
            corridor_pos[h3_idx] = idx

    edges_added = 0
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

        pos_a = corridor_pos.get(h3_a)
        pos_b = corridor_pos.get(h3_b)
        if pos_a is None or pos_b is None:
            corridor_cells_seg = None
        else:
            lo, hi = (pos_a, pos_b) if pos_a <= pos_b else (pos_b, pos_a)
            corridor_cells_seg = corridor[lo:hi + 1]

        los = compute_los(
            h3_a, h3_b,
            surface.cells, surface.config, cache,
            elevation_provider=surface.elevation_provider,
            corridor_cells=corridor_cells_seg,
        )
        if los.is_visible:
            surface.visibility_graph.add_visibility_edge(
                tower_a.tower_id, tower_b.tower_id,
                distance_m=los.distance_m,
                clearance_m=los.clearance_m,
                path_loss_db=los.path_loss_db,
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
