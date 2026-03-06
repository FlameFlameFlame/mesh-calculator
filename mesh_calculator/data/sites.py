"""
Site data structures and loading.
"""
from dataclasses import dataclass
from typing import List, Dict
import geopandas as gpd
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Site:
    """
    Target site (e.g., city) with priority level.

    Attributes:
        name: Site name
        lat, lon: Site coordinates
        priority: Priority level (1 = highest priority)
        h3_index: H3 cell index (computed)
        site_height_m: Additional endpoint antenna height above mast (AGL meters)
    """
    name: str
    lat: float
    lon: float
    priority: int
    h3_index: str = None
    site_height_m: float = 0.0


def load_sites(sites_path: str, h3_resolution: int) -> List[Site]:
    """
    Load target sites from GeoJSON file.

    Args:
        sites_path: Path to sites GeoJSON file
        h3_resolution: H3 resolution for converting to cells

    Returns:
        List of Site objects
    """
    import h3

    gdf = gpd.read_file(sites_path)
    sites = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        props = row

        # Extract coordinates
        if geom.geom_type == 'Point':
            lon, lat = geom.x, geom.y
        else:
            # Use centroid for non-point geometries
            centroid = geom.centroid
            lon, lat = centroid.x, centroid.y

        # Extract properties
        name = props.get('name', f'Site_{len(sites)+1}')
        priority = int(props.get('priority', 999))
        site_height_m = float(props.get('site_height_m', 0.0) or 0.0)

        # Convert to H3
        h3_index = h3.latlng_to_cell(lat, lon, h3_resolution)

        site = Site(
            name=name,
            lat=lat,
            lon=lon,
            priority=priority,
            h3_index=h3_index,
            site_height_m=site_height_m,
        )

        sites.append(site)

    return sites


def group_sites_by_priority(sites: List[Site]) -> Dict[int, List[Site]]:
    """
    Group sites by priority level.

    Args:
        sites: List of Site objects

    Returns:
        Dictionary mapping priority level to list of sites
    """
    groups = defaultdict(list)

    for site in sites:
        groups[site.priority].append(site)

    return dict(groups)


def find_nearest_site(
    from_site: Site,
    candidate_sites: List[Site]
) -> Site:
    """
    Find nearest site from a list of candidates.

    Args:
        from_site: Starting site
        candidate_sites: List of candidate sites

    Returns:
        Nearest site
    """
    from ..core.geometry import great_circle_distance

    nearest = None
    nearest_distance = float('inf')

    for site in candidate_sites:
        distance = great_circle_distance(
            from_site.lat, from_site.lon,
            site.lat, site.lon
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest = site

    return nearest


def snap_sites_to_roads(sites: List[Site], cells: Dict) -> None:
    """
    Snap each site's h3_index to the nearest road cell.

    Mutates sites in-place. If a site's h3_index is already in cells,
    it is left unchanged. Otherwise, finds the nearest cell by
    great-circle distance and updates h3_index.

    Args:
        sites: List of Site objects (mutated in-place)
        cells: Dictionary mapping H3 index to H3Cell objects
    """
    from ..core.geometry import great_circle_distance

    road_cells = [cell for cell in cells.values() if cell.has_road]
    if not road_cells:
        logger.warning("No road cells available for site snapping")
        return

    # Prefer cells outside city boundaries; fall back to all road cells if none remain
    eligible_cells = [c for c in road_cells if not c.is_in_unfit_area] or road_cells
    if len(eligible_cells) < len(road_cells):
        logger.info("Site snapping: excluding unfit (city-boundary) cells",
                    eligible=len(eligible_cells), total=len(road_cells))

    for site in sites:
        if site.h3_index in cells:
            continue

        best_cell = None
        best_dist = float('inf')
        for cell in eligible_cells:
            dist = great_circle_distance(site.lat, site.lon, cell.lat, cell.lon)
            if dist < best_dist:
                best_dist = dist
                best_cell = cell

        if best_cell:
            old_h3 = site.h3_index
            site.h3_index = best_cell.h3_index
            logger.info("Snapped site to nearest road cell",
                        site=site.name, old_h3=old_h3,
                        new_h3=site.h3_index, distance_m=round(best_dist, 1))
