"""
H3 grid generation and management for mesh calculator.
"""
from dataclasses import dataclass
from typing import Dict, Set, Optional
import h3
import geopandas as gpd
from shapely.geometry import Polygon, shape
import fiona

import structlog

from .config import MeshConfig
from .elevation import ElevationProvider
from .geometry import h3_to_lat_lon

logger = structlog.get_logger(__name__)


def shapely_to_h3_cells(shapely_geom, resolution: int) -> set:
    """Convert a Shapely polygon to H3 cells using h3 v4 API."""
    coords = list(shapely_geom.exterior.coords)
    # h3 expects (lat, lng) tuples
    outer = [(lat, lng) for lng, lat in coords]
    holes = []
    for interior in shapely_geom.interiors:
        hole_coords = [(lat, lng) for lng, lat in interior.coords]
        holes.append(hole_coords)
    h3_poly = h3.LatLngPoly(outer, *holes)
    return set(h3.h3shape_to_cells(h3_poly, resolution))


@dataclass
class H3Cell:
    """
    Individual H3 cell with attributes.

    Attributes:
        h3_index: H3 cell index string
        lat, lon: Cell centroid coordinates
        elevation: Elevation in meters
        has_road: Whether cell contains a road
        has_tower: Whether cell has a tower placed
        is_in_boundary: Whether cell is within the boundary
        is_in_unfit_area: Whether cell is in an unfit area
        clearance: Fresnel clearance to nearest tower (meters)
        path_loss: Path loss to nearest tower (dB)
        visible_tower_count: Number of towers with LOS to this cell
        distance_to_closest_tower: Distance to nearest tower (meters)
    """
    h3_index: str
    lat: float
    lon: float
    elevation: float
    has_road: bool = False
    has_tower: bool = False
    is_in_boundary: bool = False
    is_in_unfit_area: bool = False
    clearance: Optional[float] = None
    path_loss: Optional[float] = None
    visible_tower_count: int = 0
    distance_to_closest_tower: float = float('inf')
    closest_tower_id: Optional[int] = None
    received_power_dbm: Optional[float] = None   # Computed link budget result
    is_covered: bool = False                      # received_power >= sensitivity AND LOS
    antenna_height_offset_m: float = 0.0         # Additional endpoint AGL height above global mast


def load_boundary(boundary_path: str) -> Polygon:
    """
    Load boundary polygon from GeoJSON file.

    Args:
        boundary_path: Path to GeoJSON file

    Returns:
        Shapely Polygon object
    """
    gdf = gpd.read_file(boundary_path)

    # Get the first polygon feature
    if len(gdf) == 0:
        raise ValueError("Boundary file is empty")

    # Union all geometries if multiple features
    boundary = gdf.geometry.union_all()

    if not isinstance(boundary, Polygon):
        from shapely.geometry import MultiPolygon
        if isinstance(boundary, MultiPolygon):
            # Take the largest polygon
            boundary = max(boundary.geoms, key=lambda p: p.area)
        else:
            raise ValueError(f"Boundary must be a Polygon, got {type(boundary)}")

    return boundary


def load_roads(roads_path: str) -> gpd.GeoDataFrame:
    """
    Load road network from GeoJSON file.

    Args:
        roads_path: Path to GeoJSON file

    Returns:
        GeoDataFrame with road geometries
    """
    return gpd.read_file(roads_path)


def _sample_line_to_h3(line, resolution: int) -> Set[str]:
    """Sample points along a LineString and return H3 cells."""
    cells = set()
    # Adapt sample spacing to H3 resolution so we never skip cells.
    # Use half the average hex edge length (in degrees) as spacing.
    edge_m = h3.average_hexagon_edge_length(resolution, unit='m')
    # Convert meters to approximate degrees (1 deg ~ 111320 m)
    sample_spacing = (edge_m * 0.5) / 111320.0
    length = line.length
    if length == 0:
        return cells
    num_samples = max(int(length / sample_spacing) + 1, 2)
    for i in range(num_samples):
        frac = i / (num_samples - 1)
        point = line.interpolate(frac, normalized=True)
        h3_idx = h3.latlng_to_cell(point.y, point.x, resolution)
        cells.add(h3_idx)
    return cells


def find_h3_cells_on_roads(roads_gdf: gpd.GeoDataFrame, resolution: int) -> Set[str]:
    """
    Find all H3 cells that intersect with roads.

    Args:
        roads_gdf: GeoDataFrame containing road geometries
        resolution: H3 resolution level

    Returns:
        Set of H3 cell indices
    """
    road_cells = set()

    for _, road in roads_gdf.iterrows():
        try:
            geom = road.geometry

            if geom.geom_type == 'LineString':
                road_cells.update(_sample_line_to_h3(geom, resolution))

            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    road_cells.update(_sample_line_to_h3(line, resolution))

        except Exception as e:
            logger.warning("Failed to process road geometry", error=str(e))
            continue

    return road_cells


def generate_road_grid(
    boundary: Polygon,
    roads_gdf: gpd.GeoDataFrame,
    elevation_provider: ElevationProvider,
    config: MeshConfig,
    city_polygons: list = None,
) -> Dict[str, H3Cell]:
    """
    Generate H3 grid containing only cells within boundary and on roads.

    Args:
        boundary: Boundary polygon
        roads_gdf: GeoDataFrame with road network
        elevation_provider: Elevation data provider
        config: Mesh configuration
        city_polygons: Optional list of Shapely polygons representing city
            boundaries. Cells whose centroid falls inside any of these
            polygons will have ``is_in_unfit_area`` set to True.

    Returns:
        Dictionary mapping H3 indices to H3Cell objects
    """
    logger.info("Generating H3 grid")

    # Generate all H3 cells in boundary
    all_cells = shapely_to_h3_cells(boundary, config.h3_resolution)
    logger.info("Cells in boundary", count=len(all_cells))

    # Find cells intersecting roads
    road_cells = find_h3_cells_on_roads(roads_gdf, config.h3_resolution)
    logger.info("Cells on roads", count=len(road_cells))

    # Keep the original road cell set so we can set has_road correctly.
    # Buffer cells must NOT be marked has_road=True.
    original_road_cells = set(road_cells)

    # Expand road cells by buffer: add k-ring neighbors at main resolution.
    # Using grid_disk at the working resolution is the correct approach —
    # the res-10 intermediate sampling was wrong because cell_to_parent maps
    # small offsets back to the same parent cell.
    if config.road_buffer_m > 0:
        edge_m = h3.average_hexagon_edge_length(config.h3_resolution, unit='m')
        buffer_rings = max(1, round(config.road_buffer_m / edge_m))

        new_main_cells = set()
        for road_cell in list(road_cells):
            for neighbor in h3.grid_disk(road_cell, buffer_rings):
                new_main_cells.add(neighbor)

        added = len(new_main_cells - road_cells)
        road_cells = road_cells | new_main_cells
        logger.info(
            "Road cells after buffer expansion",
            buffer_m=config.road_buffer_m,
            rings=buffer_rings,
            added=added,
            total=len(road_cells),
        )

    # Filter to cells in boundary with roads (or buffer)
    valid_cells = all_cells & road_cells
    logger.info("Valid cells (boundary + roads + buffer)", count=len(valid_cells))

    # Create H3Cell objects with elevation data
    cells_dict = {}

    logger.debug("Loading elevation data for cells")
    for i, h3_idx in enumerate(valid_cells):
        lat, lon = h3_to_lat_lon(h3_idx)
        elevation = elevation_provider.get_elevation(lat, lon)

        cell = H3Cell(
            h3_index=h3_idx,
            lat=lat,
            lon=lon,
            elevation=elevation,
            has_road=(h3_idx in original_road_cells),
            is_in_boundary=True
        )

        cells_dict[h3_idx] = cell

        # Progress indicator
        if (i + 1) % 1000 == 0:
            logger.debug("Cell processing progress",
                         processed=i + 1, total=len(valid_cells))

    # Mark cells that fall inside city polygons as unfit for site snapping
    if city_polygons:
        from shapely.geometry import Point
        unfit_count = 0
        for cell in cells_dict.values():
            pt = Point(cell.lon, cell.lat)
            if any(pt.within(poly) for poly in city_polygons):
                cell.is_in_unfit_area = True
                unfit_count += 1
        logger.info("Cells marked as unfit (inside city boundaries)",
                    count=unfit_count)

    logger.info("Grid generation complete", cells=len(cells_dict))
    return cells_dict


def generate_full_grid(
    boundary: Polygon,
    elevation_provider: ElevationProvider,
    config: MeshConfig
) -> Dict[str, H3Cell]:
    """
    Generate H3 grid for all cells in boundary (not just roads).
    Useful for full coverage analysis.

    Args:
        boundary: Boundary polygon
        elevation_provider: Elevation data provider
        config: Mesh configuration

    Returns:
        Dictionary mapping H3 indices to H3Cell objects
    """
    logger.info("Generating full H3 grid")

    # Generate all H3 cells in boundary
    all_cells = shapely_to_h3_cells(boundary, config.h3_resolution)
    logger.info("Cells in boundary", count=len(all_cells))

    # Create H3Cell objects
    cells_dict = {}

    logger.debug("Loading elevation data for cells")
    for i, h3_idx in enumerate(all_cells):
        lat, lon = h3_to_lat_lon(h3_idx)
        elevation = elevation_provider.get_elevation(lat, lon)

        cell = H3Cell(
            h3_index=h3_idx,
            lat=lat,
            lon=lon,
            elevation=elevation,
            is_in_boundary=True
        )

        cells_dict[h3_idx] = cell

        # Progress indicator
        if (i + 1) % 1000 == 0:
            logger.debug("Cell processing progress",
                         processed=i + 1, total=len(all_cells))

    logger.info("Grid generation complete", cells=len(cells_dict))
    return cells_dict
