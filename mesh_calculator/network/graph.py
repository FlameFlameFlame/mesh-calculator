"""
Network graph representation for towers and visibility.
"""
from dataclasses import dataclass
from typing import Dict, Set
import networkx as nx

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache
from ..physics.los import compute_los

logger = structlog.get_logger(__name__)


@dataclass
class Tower:
    """
    Tower placement.

    Attributes:
        tower_id: Unique tower identifier
        h3_index: H3 cell index where tower is placed
        lat, lon: Tower coordinates
        source: Source of tower placement ('seed', 'route', 'bridge', 'greedy', 'corridor')
    """
    tower_id: int
    h3_index: str
    lat: float
    lon: float
    source: str


class VisibilityGraph:
    """
    Graph representing tower visibility relationships.

    Uses NetworkX to track which towers have line-of-sight to each other.
    """

    def __init__(self):
        """Initialize empty visibility graph."""
        self.graph = nx.Graph()

    def add_tower(self, tower: Tower):
        """
        Add a tower to the graph.

        Args:
            tower: Tower to add
        """
        self.graph.add_node(
            tower.tower_id,
            h3_index=tower.h3_index,
            lat=tower.lat,
            lon=tower.lon,
            source=tower.source
        )

    def add_visibility_edge(
        self,
        tower1_id: int,
        tower2_id: int,
        distance_m: float,
        clearance_m: float = None,
        path_loss_db: float = None
    ):
        """
        Add a visibility edge between two towers.

        Args:
            tower1_id: First tower ID
            tower2_id: Second tower ID
            distance_m: Distance in meters
            clearance_m: Optional Fresnel clearance
            path_loss_db: Optional path loss
        """
        self.graph.add_edge(
            tower1_id,
            tower2_id,
            distance_m=distance_m,
            clearance_m=clearance_m,
            path_loss_db=path_loss_db
        )

    def has_edge(self, tower1_id: int, tower2_id: int) -> bool:
        """Check if two towers have visibility."""
        return self.graph.has_edge(tower1_id, tower2_id)

    def neighbors(self, tower_id: int) -> Set[int]:
        """Get all towers visible from a given tower."""
        return set(self.graph.neighbors(tower_id))

    def get_towers(self) -> Dict[int, Tower]:
        """Get all towers in the graph."""
        towers = {}
        for tower_id in self.graph.nodes():
            node_data = self.graph.nodes[tower_id]
            towers[tower_id] = Tower(
                tower_id=tower_id,
                h3_index=node_data['h3_index'],
                lat=node_data['lat'],
                lon=node_data['lon'],
                source=node_data['source']
            )
        return towers

    def connected_components(self) -> list:
        """
        Get connected components (clusters) of towers.

        Returns:
            List of sets, each set contains tower IDs in a cluster
        """
        return list(nx.connected_components(self.graph))

    def tower_count(self) -> int:
        """Get number of towers in graph."""
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        """Get number of visibility edges in graph."""
        return self.graph.number_of_edges()


class MeshSurface:
    """
    Main mesh surface containing grid, towers, and visibility graph.
    """

    def __init__(self, cells: Dict[str, H3Cell], config: MeshConfig,
                 elevation_provider=None):
        """
        Initialize mesh surface.

        Args:
            cells: Dictionary of H3 cells
            config: Mesh configuration
            elevation_provider: Optional ElevationProvider for terrain lookups
                outside the road grid (used by Fresnel clearance checks)
        """
        self.cells = cells
        self.config = config
        self.elevation_provider = elevation_provider
        self.towers: Dict[int, Tower] = {}
        self.tower_by_h3: Dict[str, Tower] = {}
        self.visibility_graph = VisibilityGraph()
        self._next_tower_id = 1

    def place_tower(self, h3_index: str, source: str = 'unknown') -> Tower:
        """
        Place a tower at an H3 cell.

        Args:
            h3_index: H3 cell index
            source: Source of tower placement

        Returns:
            Created Tower object
        """
        if h3_index not in self.cells:
            raise ValueError(f"Cell {h3_index} not in grid")

        if h3_index in self.tower_by_h3:
            # Tower already exists here
            return self.tower_by_h3[h3_index]

        cell = self.cells[h3_index]

        tower = Tower(
            tower_id=self._next_tower_id,
            h3_index=h3_index,
            lat=cell.lat,
            lon=cell.lon,
            source=source
        )

        self.towers[tower.tower_id] = tower
        self.tower_by_h3[h3_index] = tower
        self.visibility_graph.add_tower(tower)

        # Mark cell as having a tower
        cell.has_tower = True

        self._next_tower_id += 1

        return tower

    def update_visibility_edges(self, cache: LOSCache = None):
        """
        Update visibility edges between all towers.

        Args:
            cache: Optional LOS cache
        """
        logger.info("Updating visibility edges", towers=len(self.towers))

        tower_list = list(self.towers.values())
        edges_added = 0

        for i, tower1 in enumerate(tower_list):
            for tower2 in tower_list[i+1:]:
                result = compute_los(
                    tower1.h3_index, tower2.h3_index,
                    self.cells, self.config, cache,
                    elevation_provider=self.elevation_provider,
                )
                if result.is_visible:
                    self.visibility_graph.add_visibility_edge(
                        tower1.tower_id,
                        tower2.tower_id,
                        distance_m=result.distance_m,
                        clearance_m=result.clearance_m,
                        path_loss_db=result.path_loss_db,
                    )
                    edges_added += 1

        logger.info("Visibility edges added", count=edges_added)

    def get_tower_clusters(self) -> Dict[int, int]:
        """
        Get tower clustering (connected components).

        Returns:
            Dictionary mapping tower_id to cluster_id
        """
        components = self.visibility_graph.connected_components()

        clusters = {}
        for component in components:
            cluster_id = min(component)  # Use min tower_id as cluster_id
            for tower_id in component:
                clusters[tower_id] = cluster_id

        return clusters
