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


def _dp_place_towers(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache,
    k: int,
) -> Optional[List[str]]:
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
        List of H3 indices (corridor order) forming the optimal chain, or
        None if no feasible connected chain exists within k towers.
    """
    if len(corridor) < 2:
        return list(corridor)

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

    return [corridor[i] for i in path_indices]


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
    surface: MeshSurface,
    cache: LOSCache,
) -> List[str]:
    """
    Remove interior towers whose neighbors have LOS to each other.

    The DP maximises minimum clearance and may fill its budget with adjacent
    towers that add no connectivity benefit.  This pass removes them so the
    budget is not wasted on clusters.

    Args:
        chain:   Ordered list of H3 indices returned by the DP / fallback.
        surface: MeshSurface (cells, config, elevation_provider).
        cache:   LOSCache (may be None).

    Returns:
        Pruned chain (same list object, modified in place).
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(chain) - 2, 0, -1):
            los = compute_los(
                chain[i - 1], chain[i + 1],
                surface.cells, surface.config, cache,
                elevation_provider=surface.elevation_provider,
                corridor_cells=None,
            )
            if los.is_visible:
                removed = chain.pop(i)
                logger.debug("Pruned redundant tower", node=removed)
                changed = True
    return chain


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


def place_nodes_along_corridor(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache = None
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
        corridor: List of H3 cell indices forming the corridor
        surface: Mesh surface
        cache: Optional LOS cache

    Returns:
        List of H3 indices where nodes were placed
    """
    if not corridor or len(corridor) < 2:
        return []

    logger.debug("Placing nodes along corridor", corridor_cells=len(corridor))

    config = surface.config
    cells = surface.cells

    from ..core.geometry import h3_distance, h3_to_lat_lon
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

        seg_nodes = _dp_place_towers(segment, surface, cache, seg_k)

        if seg_nodes is None:
            logger.warning(
                "DP found no feasible chain for corridor segment; "
                "falling back to peak-based placement",
                seg_len=len(segment),
                k=seg_k,
                start=segment[0],
                end=segment[-1],
            )
            seg_nodes = _peak_fallback(segment, cells, seg_k)

        # Prune redundant towers: remove interior nodes whose neighbors
        # already have LOS to each other (prevents clustering).
        pre_prune = len(seg_nodes)
        seg_nodes = _prune_redundant(seg_nodes, surface, cache)
        if len(seg_nodes) < pre_prune:
            logger.info(
                "Pruned %d redundant tower(s) from segment",
                pre_prune - len(seg_nodes),
            )

        for n in seg_nodes:
            if n not in seen:
                seen.add(n)
                all_nodes.append(n)

    logger.debug("Nodes placed along corridor (DP)", count=len(all_nodes))

    # Gap-fill: ensure no consecutive pair exceeds max_visibility_m
    pre_fill_count = len(all_nodes)
    all_nodes = _fill_visibility_gaps(all_nodes, corridor, config.max_visibility_m)
    if len(all_nodes) > pre_fill_count:
        logger.debug("After gap-fill", count=len(all_nodes))

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
    source: str = 'corridor'
):
    """
    Install nodes (towers) at specified H3 cells.

    Args:
        node_h3_list: List of H3 cell indices
        surface: Mesh surface
        source: Source label for towers
    """
    for h3_idx in node_h3_list:
        if h3_idx in surface.cells:
            surface.place_tower(h3_idx, source=source)
