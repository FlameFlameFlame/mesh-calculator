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


def place_nodes_along_corridor(
    corridor: List[str],
    surface: MeshSurface,
    cache: LOSCache = None
) -> List[str]:
    """
    Place nodes along a corridor ensuring LOS connectivity.

    Args:
        corridor: List of H3 cell indices forming the corridor
        surface: Mesh surface
        cache: Optional LOS cache

    Returns:
        List of H3 indices where nodes were placed
    """
    if not corridor or len(corridor) < 2:
        return []

    config = surface.config
    cells = surface.cells

    logger.debug("Placing nodes along corridor", corridor_cells=len(corridor))

    # Start with first cell
    placed_nodes = [corridor[0]]
    current_h3 = corridor[0]
    current_idx = 0

    while current_idx < len(corridor) - 1:
        # Try to reach as far as possible with LOS
        furthest_visible_idx = current_idx + 1
        furthest_visible_h3 = corridor[furthest_visible_idx]

        # Look ahead to find furthest visible cell
        for next_idx in range(current_idx + 2, len(corridor)):
            next_h3 = corridor[next_idx]

            # Check distance constraint
            from ..core.geometry import h3_distance
            distance = h3_distance(current_h3, next_h3)
            if distance > config.max_visibility_m:
                break

            # Check LOS
            if has_los(current_h3, next_h3, cells, config, cache):
                furthest_visible_idx = next_idx
                furthest_visible_h3 = next_h3
            else:
                # Can't see further, stop looking
                break

        # Place node at furthest visible point
        if furthest_visible_h3 not in placed_nodes:
            placed_nodes.append(furthest_visible_h3)

        # Move to next position
        current_h3 = furthest_visible_h3
        current_idx = furthest_visible_idx

    # Ensure end node is included
    if corridor[-1] not in placed_nodes:
        placed_nodes.append(corridor[-1])

    logger.debug("Nodes placed along corridor", count=len(placed_nodes))

    # Check if we need to optimize for node limit
    if len(placed_nodes) > config.max_nodes_per_road:
        logger.debug("Optimizing node count",
                     current=len(placed_nodes), limit=config.max_nodes_per_road)
        placed_nodes = optimize_node_selection(
            placed_nodes, config.max_nodes_per_road, cells, config, cache
        )
        logger.debug("After optimization", count=len(placed_nodes))

    return placed_nodes


def optimize_node_selection(
    nodes: List[str],
    max_nodes: int,
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None
) -> List[str]:
    """
    Select best subset of nodes respecting limit while maintaining connectivity.

    Args:
        nodes: List of H3 cell indices
        max_nodes: Maximum number of nodes allowed
        cells: Dictionary of H3 cells
        config: Mesh configuration
        cache: Optional LOS cache

    Returns:
        Optimized list of H3 indices
    """
    if len(nodes) <= max_nodes:
        return nodes

    # Always keep start and end
    essential = [nodes[0], nodes[-1]]
    candidates = nodes[1:-1]

    if max_nodes <= 2:
        return essential

    # Score each candidate node
    scores = []
    for node in candidates:
        score = 0.0

        # Prefer high elevation (better LOS)
        cell = cells.get(node)
        if cell:
            score += cell.elevation / 100.0

        # Count how many other candidates this node can see
        visible_count = 0
        for other in candidates:
            if other != node and has_los(node, other, cells, config, cache):
                visible_count += 1

        score += visible_count

        scores.append((score, node))

    # Sort by score (highest first)
    scores.sort(reverse=True, key=lambda x: x[0])

    # Select top (max_nodes - 2) candidates
    selected = [n for _, n in scores[:max_nodes-2]]

    # Return start + selected + end
    return [nodes[0]] + selected + [nodes[-1]]


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
