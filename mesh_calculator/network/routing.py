"""
Dijkstra pathfinding for road corridors between sites.
"""
from typing import Dict, List, Optional
import networkx as nx
import h3
import geopandas as gpd

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig

logger = structlog.get_logger(__name__)


def build_routing_graph(
    cells: Dict[str, H3Cell],
    roads_gdf: gpd.GeoDataFrame,
    config: MeshConfig
) -> nx.DiGraph:
    """
    Build routing graph from H3 cells containing roads.

    Args:
        cells: Dictionary of H3 cells
        roads_gdf: GeoDataFrame with road network
        config: Mesh configuration

    Returns:
        Directed graph for pathfinding
    """
    logger.info("Building routing graph")

    G = nx.DiGraph()

    # Add nodes for all cells with roads
    road_cells = [h3_idx for h3_idx, cell in cells.items() if cell.has_road]
    G.add_nodes_from(road_cells)

    logger.debug("Routing graph nodes added", count=len(road_cells))

    # Add edges between neighboring cells
    edges_added = 0
    for h3_idx in road_cells:
        # Get immediate neighbors (k=1 ring)
        neighbors = h3.grid_ring(h3_idx, 1)

        for neighbor in neighbors:
            if neighbor in G.nodes():
                # Calculate edge cost
                cost = calculate_edge_cost(h3_idx, neighbor, cells, roads_gdf, config)
                G.add_edge(h3_idx, neighbor, cost=cost)
                edges_added += 1

    logger.debug("Routing graph edges added", count=edges_added)

    return G


def calculate_edge_cost(
    h3_src: str,
    h3_dst: str,
    cells: Dict[str, H3Cell],
    roads_gdf: gpd.GeoDataFrame,
    config: MeshConfig
) -> float:
    """
    Calculate cost for traversing between neighboring cells.

    Args:
        h3_src: Source H3 cell index
        h3_dst: Destination H3 cell index
        cells: Dictionary of H3 cells
        roads_gdf: GeoDataFrame with roads
        config: Mesh configuration

    Returns:
        Edge cost (lower is better)
    """
    cost = 1.0  # Base cost

    # Elevation change penalty (prefer flat terrain)
    src_cell = cells.get(h3_src)
    dst_cell = cells.get(h3_dst)

    if src_cell and dst_cell:
        elev_change = abs(dst_cell.elevation - src_cell.elevation)

        # Add penalty for significant elevation changes
        if elev_change > 50:  # meters
            cost += 0.5

        # Extra penalty for steep climbs
        if elev_change > 100:
            cost += 1.0

    # TODO: Add road quality penalty (prefer highways over secondary roads)
    # This requires road type information in roads_gdf
    # For now, all roads are treated equally

    return cost


def find_road_corridor(
    h3_start: str,
    h3_end: str,
    routing_graph: nx.DiGraph
) -> Optional[List[str]]:
    """
    Find optimal road corridor between two H3 cells using Dijkstra.

    Args:
        h3_start: Start H3 cell index
        h3_end: End H3 cell index
        routing_graph: Routing graph

    Returns:
        List of H3 cell indices forming the corridor, or None if no path exists
    """
    # Ensure both nodes are in the graph
    if h3_start not in routing_graph.nodes():
        logger.warning("Start cell not in routing graph", cell=h3_start)
        return None

    if h3_end not in routing_graph.nodes():
        logger.warning("End cell not in routing graph", cell=h3_end)
        return None

    try:
        # Find shortest path using Dijkstra with cost weights
        path = nx.dijkstra_path(routing_graph, h3_start, h3_end, weight='cost')
        return path

    except nx.NetworkXNoPath:
        logger.warning("No path found", start=h3_start, end=h3_end)
        return None

    except Exception as e:
        logger.error("Error finding path", error=str(e), exc_info=True)
        return None


def find_multiple_corridors(
    pairs: List[tuple[str, str]],
    routing_graph: nx.DiGraph
) -> Dict[tuple[str, str], Optional[List[str]]]:
    """
    Find corridors for multiple start-end pairs.

    Args:
        pairs: List of (start, end) H3 cell index tuples
        routing_graph: Routing graph

    Returns:
        Dictionary mapping pairs to corridor paths
    """
    corridors = {}

    for start, end in pairs:
        corridor = find_road_corridor(start, end, routing_graph)
        corridors[(start, end)] = corridor

    return corridors


def corridor_length(corridor: List[str], cells: Dict[str, H3Cell]) -> float:
    """
    Calculate total length of a corridor.

    Args:
        corridor: List of H3 cell indices
        cells: Dictionary of H3 cells

    Returns:
        Total length in meters
    """
    if not corridor or len(corridor) < 2:
        return 0.0

    from ..core.geometry import h3_distance

    total_length = 0.0
    for i in range(len(corridor) - 1):
        total_length += h3_distance(corridor[i], corridor[i+1])

    return total_length
