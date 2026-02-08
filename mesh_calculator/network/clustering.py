"""
Tower clustering using connected components analysis.

Based on mesh_tower_clusters.sql from the original implementation.
"""
from typing import Dict, Set
import networkx as nx

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache
from ..physics.los import has_los
from ..core.geometry import h3_distance
from .graph import Tower

logger = structlog.get_logger(__name__)


def compute_tower_clusters(
    towers: Dict[int, Tower],
    cells: Dict[str, H3Cell],
    config: MeshConfig,
    cache: LOSCache = None
) -> Dict[int, int]:
    """
    Compute tower clusters using connected components.

    Towers are in the same cluster if they have line-of-sight visibility
    within the maximum visibility distance.

    Implements the algorithm from mesh_tower_clusters.sql (lines 12-53).

    Args:
        towers: Dictionary of towers
        cells: Dictionary of H3 cells
        config: Mesh configuration
        cache: Optional LOS cache

    Returns:
        Dictionary mapping tower_id to cluster_id (using min tower_id as cluster_id)

    Reference:
        h3-mesh-placement/functions/mesh_tower_clusters.sql:12-53
    """
    if not towers:
        return {}

    logger.info("Computing clusters", towers=len(towers))

    # Build visibility graph using NetworkX
    G = nx.Graph()
    G.add_nodes_from(towers.keys())

    # Add edges for towers with LOS within max visibility distance
    tower_list = list(towers.values())
    edges_added = 0

    for i, t1 in enumerate(tower_list):
        for t2 in tower_list[i+1:]:
            # Check distance constraint
            distance = h3_distance(t1.h3_index, t2.h3_index)

            if distance <= config.max_visibility_m:
                # Check LOS
                if has_los(t1.h3_index, t2.h3_index, cells, config, cache):
                    G.add_edge(t1.tower_id, t2.tower_id)
                    edges_added += 1

    logger.debug("Visibility edges found", count=edges_added)

    # Find connected components
    components = list(nx.connected_components(G))
    logger.info("Clusters identified", count=len(components))

    # Assign cluster IDs (use min tower_id as cluster_id)
    clusters = {}
    for component in components:
        cluster_id = min(component)
        for tower_id in component:
            clusters[tower_id] = cluster_id

    # Report cluster sizes
    cluster_sizes = {}
    for cluster_id in clusters.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    if cluster_sizes:
        logger.debug("Cluster sizes", sizes=dict(list(cluster_sizes.items())[:5]))

    return clusters


def get_cluster_representatives(
    clusters: Dict[int, int],
    towers: Dict[int, Tower]
) -> Dict[int, Tower]:
    """
    Get representative tower for each cluster (the one with cluster_id as tower_id).

    Args:
        clusters: Dictionary mapping tower_id to cluster_id
        towers: Dictionary of towers

    Returns:
        Dictionary mapping cluster_id to representative Tower
    """
    representatives = {}

    for cluster_id in set(clusters.values()):
        if cluster_id in towers:
            representatives[cluster_id] = towers[cluster_id]

    return representatives


def find_nearest_tower_in_clusters(
    from_h3: str,
    target_clusters: Set[int],
    towers: Dict[int, Tower],
    clusters: Dict[int, int]
) -> Optional[Tower]:
    """
    Find nearest tower that belongs to any of the target clusters.

    Args:
        from_h3: Starting H3 cell index
        target_clusters: Set of cluster IDs to search in
        towers: Dictionary of towers
        clusters: Dictionary mapping tower_id to cluster_id

    Returns:
        Nearest tower in target clusters, or None if none found
    """
    from ..core.geometry import h3_distance

    nearest_tower = None
    nearest_distance = float('inf')

    for tower_id, tower in towers.items():
        cluster_id = clusters.get(tower_id)

        if cluster_id in target_clusters:
            distance = h3_distance(from_h3, tower.h3_index)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_tower = tower

    return nearest_tower
