"""
Network graph representation for towers and visibility.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

import structlog

from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.cache import LOSCache
from ..physics.los import compute_los
from .tower_coverage import CoverageSource, compute_h3_tower_coverage

logger = structlog.get_logger(__name__)

# Earth radius for approximate Cartesian conversion (meters)
_EARTH_R = 6_371_000


@dataclass
class Tower:
    """
    Tower placement.

    Attributes:
        tower_id: Unique tower identifier
        h3_index: H3 cell index where tower is placed
        lat, lon: Tower coordinates
        source: Source of tower placement ('seed', 'route', 'bridge', 'greedy', 'corridor')
        placement_meta: Debug metadata about how/why this tower was placed.
            Keys: algorithm ('dp'|'dp_repair'|'peak_fallback'|'endpoint_fallback'|'site'),
                  dp_steps (int, tower count t used in winning DP solution),
                  repair_round (int, 1-3 for dp_repair towers).
    """
    tower_id: int
    h3_index: str
    lat: float
    lon: float
    source: str
    city_link: bool = False
    coverage_radius_m: float = 0.0  # Max distance of any LOS-covered cell
    placement_meta: dict = field(default_factory=dict)


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
        self.gap_repair_debug: list = []  # debug records from gap repair rounds

    def place_tower(
        self,
        h3_index: str,
        source: str = 'unknown',
        placement_meta: Optional[dict] = None,
    ) -> Tower:
        """
        Place a tower at an H3 cell.

        Args:
            h3_index: H3 cell index
            source: Source of tower placement
            placement_meta: Optional debug metadata (algorithm, dp_steps, repair_round, …)

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
            source=source,
            placement_meta=placement_meta or {},
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

        Uses a KDTree spatial index to skip pairs beyond max_visibility_m,
        then computes LOS in parallel for candidate pairs.

        Args:
            cache: Optional LOS cache
        """
        tower_list = list(self.towers.values())
        n = len(tower_list)
        logger.info("Updating visibility edges", towers=n)

        if n < 2:
            return

        # Build KDTree from tower coords for fast proximity queries
        lats = np.array([t.lat for t in tower_list])
        lons = np.array([t.lon for t in tower_list])
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        cos_lat = np.cos(lats_rad)
        xyz = np.column_stack([
            cos_lat * np.cos(lons_rad),
            cos_lat * np.sin(lons_rad),
            np.sin(lats_rad),
        ]) * _EARTH_R
        tree = cKDTree(xyz)

        max_dist = self.config.max_visibility_m
        candidate_pairs = tree.query_pairs(r=max_dist, output_type='ndarray')
        logger.info("Visibility candidates after spatial filter",
                     total_pairs=n * (n - 1) // 2,
                     candidate_pairs=len(candidate_pairs))

        # Compute LOS in parallel for candidate pairs
        cells = self.cells
        config = self.config
        elev = self.elevation_provider

        def _check_pair(idx_pair):
            i, j = idx_pair
            t1, t2 = tower_list[i], tower_list[j]
            result = compute_los(
                t1.h3_index, t2.h3_index,
                cells, config, cache,
                elevation_provider=elev,
            )
            if result.is_visible:
                return (t1.tower_id, t2.tower_id,
                        result.distance_m, result.clearance_m,
                        result.path_loss_db)
            return None

        edges_added = 0
        max_workers = os.cpu_count() or 4

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_check_pair, pair)
                       for pair in candidate_pairs]
            for future in as_completed(futures):
                edge = future.result()
                if edge is not None:
                    tid1, tid2, dist, clearance, ploss = edge
                    self.visibility_graph.add_visibility_edge(
                        tid1, tid2,
                        distance_m=dist,
                        clearance_m=clearance,
                        path_loss_db=ploss,
                    )
                    edges_added += 1

        logger.info("Visibility edges added", count=edges_added)

    def compute_cell_coverage(self, cache: LOSCache = None):
        """Compute per-cell coverage metrics from placed towers.

        Uses a KDTree to find towers within max_visibility_m for each cell,
        then checks LOS in parallel. Updates: visible_tower_count,
        distance_to_closest_tower, clearance (best), and path_loss (best).
        """
        tower_list = list(self.towers.values())
        if not tower_list:
            return

        max_dist = self.config.max_visibility_m
        cells = self.cells
        config = self.config
        elev = self.elevation_provider

        # Build KDTree from tower positions
        t_lats = np.array([t.lat for t in tower_list])
        t_lons = np.array([t.lon for t in tower_list])
        t_lats_rad = np.radians(t_lats)
        t_lons_rad = np.radians(t_lons)
        t_cos_lat = np.cos(t_lats_rad)
        tower_xyz = np.column_stack([
            t_cos_lat * np.cos(t_lons_rad),
            t_cos_lat * np.sin(t_lons_rad),
            np.sin(t_lats_rad),
        ]) * _EARTH_R
        tower_tree = cKDTree(tower_xyz)

        # Map tower h3_index → index for fast lookup
        tower_h3_set = {t.h3_index for t in tower_list}

        # Build cell coordinates for batch query
        cell_list = list(cells.values())
        c_lats = np.array([c.lat for c in cell_list])
        c_lons = np.array([c.lon for c in cell_list])
        c_lats_rad = np.radians(c_lats)
        c_lons_rad = np.radians(c_lons)
        c_cos_lat = np.cos(c_lats_rad)
        cell_xyz = np.column_stack([
            c_cos_lat * np.cos(c_lons_rad),
            c_cos_lat * np.sin(c_lons_rad),
            np.sin(c_lats_rad),
        ]) * _EARTH_R

        # For each cell, find nearby tower indices
        nearby = tower_tree.query_ball_point(cell_xyz, r=max_dist)

        # Build (cell_idx, tower_idx) pairs that need LOS checks
        los_pairs = []
        # Track cells that have a tower on them (distance=0, always visible)
        cells_with_own_tower = set()
        for ci, cell in enumerate(cell_list):
            if cell.h3_index in tower_h3_set:
                cells_with_own_tower.add(ci)
            for ti in nearby[ci]:
                t = tower_list[ti]
                if t.h3_index == cell.h3_index:
                    continue  # handled separately
                los_pairs.append((ci, ti))

        logger.info("Cell coverage: %d cells, %d towers, %d LOS checks",
                    len(cell_list), len(tower_list), len(los_pairs))

        # Compute LOS in parallel
        def _check_cell_tower(pair):
            ci, ti = pair
            c = cell_list[ci]
            t = tower_list[ti]
            result = compute_los(
                c.h3_index, t.h3_index,
                cells, config, cache,
                elevation_provider=elev,
            )
            if result.is_visible:
                return (ci, ti, result.distance_m,
                        result.clearance_m, result.path_loss_db)
            return None

        # Accumulate per-cell results
        # ci → [visible_count, best_dist, best_clear, best_ploss, best_tower_id]
        cell_results = {}
        for ci in cells_with_own_tower:
            cell = cell_list[ci]
            own_tower = self.tower_by_h3.get(cell.h3_index)
            own_id = own_tower.tower_id if own_tower else None
            cell_results[ci] = [1, 0.0, 0.0, 0.0, own_id]

        max_workers = os.cpu_count() or 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_check_cell_tower, p) for p in los_pairs]
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                ci, ti, dist_m, clear_m, ploss_db = res
                if ci not in cell_results:
                    cell_results[ci] = [0, float('inf'), None, None, None]
                entry = cell_results[ci]
                entry[0] += 1
                if dist_m < entry[1]:
                    entry[1] = dist_m
                    entry[2] = clear_m
                    entry[3] = ploss_db
                    entry[4] = tower_list[ti].tower_id

        # Apply results to cells
        tx_dbm = config.tx_power_dbm
        gain = 2.0 * config.antenna_gain_dbi
        sens = config.receiver_sensitivity_dbm

        updated = 0
        for ci, (count, dist, clearance, ploss, closest_tid) in cell_results.items():
            cell = cell_list[ci]
            cell.visible_tower_count = count
            cell.distance_to_closest_tower = dist
            cell.clearance = clearance
            cell.path_loss = ploss
            cell.closest_tower_id = closest_tid
            if ploss is not None:
                rx = tx_dbm + gain - ploss
                cell.received_power_dbm = rx
                cell.is_covered = (count > 0 and rx >= sens)
            updated += 1

        # Per-tower coverage radius: max distance of any covered cell
        tower_radius: Dict[int, float] = {}
        for cell in self.cells.values():
            if cell.is_covered and cell.closest_tower_id is not None:
                tid = cell.closest_tower_id
                d = cell.distance_to_closest_tower
                if d > tower_radius.get(tid, 0.0):
                    tower_radius[tid] = d
        for tid, radius in tower_radius.items():
            self.towers[tid].coverage_radius_m = radius

        logger.info("Cell coverage computed",
                    cells_with_coverage=updated, total_cells=len(cell_list))

    def compute_tower_radial_coverage(self, los_cache: LOSCache = None) -> list:
        """Backward-compatible wrapper over standalone tower-coverage API."""
        if not self.towers:
            return []

        sources = [
            CoverageSource(
                source_id=t.tower_id,
                h3_index=t.h3_index,
                lat=t.lat,
                lon=t.lon,
            )
            for t in self.towers.values()
        ]
        return compute_h3_tower_coverage(
            sources=sources,
            base_cells=self.cells,
            config=self.config,
            elevation_provider=self.elevation_provider,
            los_cache=los_cache,
        )

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
