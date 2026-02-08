"""
Site data structures and loading.
"""
from dataclasses import dataclass
from typing import List, Dict
import geopandas as gpd
from collections import defaultdict


@dataclass
class Site:
    """
    Target site (e.g., city) with priority level.

    Attributes:
        name: Site name
        lat, lon: Site coordinates
        priority: Priority level (1 = highest priority)
        h3_index: H3 cell index (computed)
    """
    name: str
    lat: float
    lon: float
    priority: int
    h3_index: str = None


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

        # Convert to H3
        h3_index = h3.latlng_to_cell(lat, lon, h3_resolution)

        site = Site(
            name=name,
            lat=lat,
            lon=lon,
            priority=priority,
            h3_index=h3_index
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
