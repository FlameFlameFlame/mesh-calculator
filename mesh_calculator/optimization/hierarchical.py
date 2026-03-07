"""
Hierarchical site connectivity based on priority levels.

Implements the core algorithm:
- Priority 1 sites: Fully interconnected (all-to-all)
- Priority 2+ sites: Connect to nearest higher-priority site
"""
import itertools
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx

import structlog

from ..data.sites import Site, group_sites_by_priority, find_nearest_site
from ..network.graph import MeshSurface
from ..network.routing import build_routing_graph, find_road_corridor
from ..data.cache import LOSCache
from .corridor import place_nodes_along_corridor, install_nodes, wire_corridor_edges

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

    # Place a tower at every site location (guarantees infrastructure)
    for site in sites:
        if site.h3_index in surface.cells:
            cell = surface.cells[site.h3_index]
            prev = float(getattr(cell, 'antenna_height_offset_m', 0.0) or 0.0)
            site_offset = max(0.0, float(getattr(site, 'site_height_m', 0.0) or 0.0))
            cell.antenna_height_offset_m = max(prev, site_offset)
            surface.place_tower(site.h3_index, source='site')
            logger.info("Placed site tower", site=site.name, h3=site.h3_index)
        else:
            logger.warning("Site cell not in grid, cannot place tower",
                           site=site.name, h3=site.h3_index)

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

        # Step 1: Sequential — determine targets + find corridors
        # (must preserve connected_sites order for correct find_nearest_site results)
        planned = []
        for site in current_level_sites:
            nearest_site = find_nearest_site(site, connected_sites)

            if nearest_site is None:
                logger.warning("No connected site found", site=site.name)
                continue

            logger.info("Connecting site",
                        site=site.name, site_priority=priority,
                        target=nearest_site.name, target_priority=nearest_site.priority)

            corridor = find_road_corridor(site.h3_index, nearest_site.h3_index, routing_graph)
            planned.append((site, nearest_site, corridor))
            connected_sites.append(site)

        if not planned:
            continue

        # Step 2: Parallel — compute node placement along each corridor
        def _compute_nodes(item):
            site, nearest_site, corridor = item
            if not corridor:
                return None, None, None, site.name, nearest_site.name, 0
            meta: dict = {}
            nodes = place_nodes_along_corridor(
                corridor, surface, cache, out_meta=meta, los_max_workers=1,
            )
            return nodes, corridor, meta, site.name, nearest_site.name, len(corridor)

        with ThreadPoolExecutor() as executor:
            node_results = list(executor.map(_compute_nodes, planned))

        # Step 3: Serial — install towers and wire corridor-path edges
        for nodes, corridor, meta, site_name, target_name, corridor_len in node_results:
            if nodes:
                install_nodes(nodes, surface,
                              source=f'priority_{priority}',
                              placement_meta=meta)
                wire_corridor_edges(nodes, corridor, surface, cache)
                logger.info("Corridor established",
                            site=site_name, target=target_name,
                            corridor_cells=corridor_len, nodes_placed=len(nodes))
            else:
                logger.warning("No corridor found", site=site_name, target=target_name)

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

    # Parallel — find corridors and place nodes concurrently
    # (surface reads are safe; install_nodes is deferred to serial phase)
    def _process_pair(args):
        site1, site2 = args
        corridor = find_road_corridor(site1.h3_index, site2.h3_index, routing_graph)
        if not corridor:
            return None, None, None, site1.name, site2.name, 0
        meta: dict = {}
        nodes = place_nodes_along_corridor(
            corridor, surface, cache, out_meta=meta, los_max_workers=1,
        )
        return nodes, corridor, meta, site1.name, site2.name, len(corridor)

    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_process_pair, (s1, s2)): (s1, s2) for s1, s2 in pairs}
        for future in as_completed(futures):
            results.append(future.result())

    # Serial installation — fast, negligible time
    for nodes, corridor, meta, name1, name2, corridor_len in results:
        if nodes:
            install_nodes(nodes, surface, source='priority_1',
                          placement_meta=meta)
            wire_corridor_edges(nodes, corridor, surface, cache)
            logger.info("Corridor established",
                        site1=name1, site2=name2,
                        corridor_cells=corridor_len, nodes_placed=len(nodes))
        else:
            logger.warning("No corridor found", site1=name1, site2=name2)

    logger.info("Priority 1 mesh complete",
                towers_placed=len(surface.towers))
