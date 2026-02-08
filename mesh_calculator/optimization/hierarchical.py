"""
Hierarchical site connectivity based on priority levels.

Implements the core algorithm:
- Priority 1 sites: Fully interconnected (all-to-all)
- Priority 2+ sites: Connect to nearest higher-priority site
"""
import itertools
from typing import List
import networkx as nx

import structlog

from ..data.sites import Site, group_sites_by_priority, find_nearest_site
from ..network.graph import MeshSurface
from ..network.routing import build_routing_graph, find_road_corridor
from ..data.cache import LOSCache
from .corridor import place_nodes_along_corridor, install_nodes

logger = structlog.get_logger(__name__)


def connect_sites_by_priority(
    sites: List[Site],
    surface: MeshSurface,
    routing_graph: nx.DiGraph,
    cache: LOSCache = None
):
    """
    Connect sites based on hierarchical priorities.

    Algorithm:
    1. Connect all Priority 1 sites (full mesh)
    2. For each Priority 2 site, connect to nearest Priority 1 site
    3. For each Priority 3 site, connect to nearest Priority 2+ site
    4. And so on...

    Args:
        sites: List of Site objects with priorities
        surface: Mesh surface
        routing_graph: Routing graph for pathfinding
        cache: Optional LOS cache
    """
    if not sites:
        logger.info("No sites to connect")
        return

    logger.info("Connecting sites by priority hierarchy", site_count=len(sites))

    # Group sites by priority
    sites_by_priority = group_sites_by_priority(sites)
    priority_levels = sorted(sites_by_priority.keys())

    logger.info("Priority levels identified", levels=priority_levels)
    for p in priority_levels:
        logger.info("Priority level", priority=p, sites=len(sites_by_priority[p]))

    # Phase 1: Connect all Priority 1 sites (full mesh)
    if 1 in sites_by_priority:
        connect_priority1_mesh(sites_by_priority[1], surface, routing_graph, cache)

    # Track all connected sites
    connected_sites = list(sites_by_priority.get(1, []))

    # Phase 2+: Connect each priority level to higher priorities
    for priority in priority_levels[1:]:  # Skip priority 1 (already done)
        current_level_sites = sites_by_priority[priority]

        logger.info("Connecting priority level",
                    priority=priority, sites=len(current_level_sites))

        for site in current_level_sites:
            # Find nearest higher-priority site
            nearest_site = find_nearest_site(site, connected_sites)

            if nearest_site is None:
                logger.warning("No connected site found", site=site.name)
                continue

            logger.info("Connecting site",
                        site=site.name, site_priority=priority,
                        target=nearest_site.name, target_priority=nearest_site.priority)

            # Find corridor
            corridor = find_road_corridor(site.h3_index, nearest_site.h3_index, routing_graph)

            if corridor:
                # Place nodes along corridor
                nodes = place_nodes_along_corridor(corridor, surface, cache)

                # Install towers
                install_nodes(nodes, surface, source=f'priority_{priority}')

                # Mark this site as connected
                connected_sites.append(site)

                logger.info("Corridor established",
                            corridor_cells=len(corridor), nodes_placed=len(nodes))
            else:
                logger.warning("No corridor found",
                               site=site.name, target=nearest_site.name)

    logger.info("Hierarchical connectivity complete",
                towers_placed=len(surface.towers))


def connect_priority1_mesh(
    priority1_sites: List[Site],
    surface: MeshSurface,
    routing_graph: nx.DiGraph,
    cache: LOSCache = None
):
    """
    Connect all Priority 1 sites in a full mesh (all-to-all).

    Args:
        priority1_sites: List of Priority 1 sites
        surface: Mesh surface
        routing_graph: Routing graph for pathfinding
        cache: Optional LOS cache
    """
    if len(priority1_sites) < 2:
        logger.info("Priority 1: only one site, no connections needed")
        return

    logger.info("Connecting Priority 1 sites (full mesh)",
                sites=len(priority1_sites))

    # Generate all pairs
    pairs = list(itertools.combinations(priority1_sites, 2))
    logger.info("Connections to establish", total=len(pairs))

    for i, (site1, site2) in enumerate(pairs, 1):
        logger.info("Connecting pair",
                    pair=f"{i}/{len(pairs)}",
                    site1=site1.name, site2=site2.name)

        # Find corridor
        corridor = find_road_corridor(site1.h3_index, site2.h3_index, routing_graph)

        if corridor:
            # Place nodes along corridor
            nodes = place_nodes_along_corridor(corridor, surface, cache)

            # Install towers
            install_nodes(nodes, surface, source='priority_1')

            logger.info("Corridor established",
                        corridor_cells=len(corridor), nodes_placed=len(nodes))
        else:
            logger.warning("No corridor found",
                           site1=site1.name, site2=site2.name)

    logger.info("Priority 1 mesh complete",
                towers_placed=len(surface.towers))
