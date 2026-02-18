"""
Node placement along corridors with LOS constraints.
"""
from typing import List, Dict
import h3

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache
from ..physics.los import has_los
from ..network.graph import MeshSurface

logger = structlog.get_logger(__name__)


def _walk_segment(
    segment: List[str],
    surface: MeshSurface,
    cache: LOSCache = None
) -> List[str]:
    """
    Walk a corridor segment and place towers ensuring LOS connectivity.
    Returns list of H3 indices (including both endpoints).
    """
    if not segment or len(segment) < 2:
        return list(segment)

    config = surface.config
    cells = surface.cells

    from ..core.geometry import h3_distance

    placed_nodes = [segment[0]]
    current_h3 = segment[0]
    current_idx = 0

    while current_idx < len(segment) - 1:
        furthest_visible_idx = None

        # Find the farthest reachable index within max_visibility_m.
        # Cache distances so the backward scan can reuse them without recomputing.
        distances: dict = {}
        max_reachable_idx = current_idx
        for next_idx in range(current_idx + 1, len(segment)):
            d = h3_distance(current_h3, segment[next_idx])
            distances[next_idx] = d
            if d > config.max_visibility_m:
                break
            max_reachable_idx = next_idx

        # Scan backwards from farthest reachable cell — first LOS hit is furthest
        # visible, so we stop immediately (O(1) vs O(n) forward scan)
        for next_idx in range(max_reachable_idx, current_idx, -1):
            next_h3 = segment[next_idx]
            distance = distances.get(next_idx) or h3_distance(current_h3, next_h3)

            is_endpoint = (next_idx == len(segment) - 1)
            if (not is_endpoint
                    and config.tower_separation_m > 0
                    and distance < config.tower_separation_m):
                continue

            if has_los(current_h3, next_h3, cells, config, cache,
                       elevation_provider=surface.elevation_provider):
                furthest_visible_idx = next_idx
                break  # first hit from the far end = furthest visible cell

        if furthest_visible_idx is not None:
            furthest_h3 = segment[furthest_visible_idx]
            if furthest_h3 not in placed_nodes:
                placed_nodes.append(furthest_h3)
            current_h3 = furthest_h3
            current_idx = furthest_visible_idx
        else:
            # No visible cell — forced advance by one
            current_idx += 1
            current_h3 = segment[current_idx]
            if current_h3 not in placed_nodes:
                placed_nodes.append(current_h3)
            logger.warning(
                "No LOS to any forward cell, forced advance",
                current=segment[current_idx - 1],
                next=current_h3,
            )

    if segment[-1] not in placed_nodes:
        placed_nodes.append(segment[-1])

    return placed_nodes


def place_nodes_along_corridor(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache = None
) -> List[str]:
    """
    Place nodes along a corridor ensuring LOS connectivity.

    Existing towers that fall on the corridor are reused as free relay
    points — the corridor is split at each such tower and each segment
    is processed independently, so no new tower is placed where one
    already exists.

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

    # Find existing towers on this corridor and use them as free waypoints.
    # Split the corridor into segments separated by existing towers.
    existing_tower_indices = [
        i for i, h3_idx in enumerate(corridor)
        if h3_idx in surface.tower_by_h3
    ]

    if existing_tower_indices:
        logger.debug("Reusing existing towers as waypoints",
                     count=len(existing_tower_indices))

    # Build segment boundaries: [0, t1, t2, ..., last]
    boundaries = [0] + existing_tower_indices
    if boundaries[-1] != len(corridor) - 1:
        boundaries.append(len(corridor) - 1)

    all_nodes: List[str] = []
    seen: set = set()

    for bi in range(len(boundaries) - 1):
        seg_start = boundaries[bi]
        seg_end = boundaries[bi + 1]
        segment = corridor[seg_start:seg_end + 1]
        seg_nodes = _walk_segment(segment, surface, cache)
        for n in seg_nodes:
            if n not in seen:
                seen.add(n)
                all_nodes.append(n)

    placed_nodes = all_nodes

    logger.debug("Nodes placed along corridor", count=len(placed_nodes))

    config = surface.config
    cells = surface.cells

    # Check if we need to optimize for node limit
    if len(placed_nodes) > config.max_nodes_per_road:
        logger.debug("Optimizing node count",
                     current=len(placed_nodes), limit=config.max_nodes_per_road)
        placed_nodes = optimize_node_selection(
            placed_nodes, config.max_nodes_per_road, cells, config, cache,
            elevation_provider=surface.elevation_provider,
        )
        logger.debug("After optimization", count=len(placed_nodes))

    # Validate hop limit
    hop_count = len(placed_nodes) - 1
    if hop_count > config.hop_limit:
        logger.warning(
            "Corridor exceeds hop limit",
            hops=hop_count,
            limit=config.hop_limit,
            start=corridor[0],
            end=corridor[-1],
        )

    return placed_nodes


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
            # Can't remove any more without breaking connectivity
            logger.warning(
                "Cannot reduce to target without breaking connectivity",
                current=len(result), target=max_nodes)
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
