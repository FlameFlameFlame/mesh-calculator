"""
Geometric utilities for mesh calculator.
"""
import math
import functools
from typing import Tuple
import h3
from pyproj import Geod


# WGS84 geodesic calculator
_geod = Geod(ellps='WGS84')


def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using WGS84 ellipsoid.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in meters
    """
    _, _, distance = _geod.inv(lon1, lat1, lon2, lat2)
    return distance


@functools.lru_cache(maxsize=None)
def h3_to_lat_lon(h3_index: str) -> Tuple[float, float]:
    """
    Convert H3 index to latitude/longitude.

    Args:
        h3_index: H3 cell index

    Returns:
        (latitude, longitude) tuple in degrees
    """
    lat, lon = h3.cell_to_latlng(h3_index)
    return lat, lon


def lat_lon_to_h3(lat: float, lon: float, resolution: int) -> str:
    """
    Convert latitude/longitude to H3 index.

    Args:
        lat, lon: Coordinates in degrees
        resolution: H3 resolution level

    Returns:
        H3 cell index
    """
    return h3.latlng_to_cell(lat, lon, resolution)


@functools.lru_cache(maxsize=None)
def calculate_line_fraction(point_h3: str, start_h3: str, end_h3: str) -> float:
    """
    Calculate fractional position of a point along a line between start and end.

    Uses planar dot-product projection (equivalent to the previous Shapely
    implementation but avoids Shapely object creation overhead).

    Args:
        point_h3: H3 cell index of point
        start_h3: H3 cell index of line start
        end_h3: H3 cell index of line end

    Returns:
        Fraction from 0.0 (at start) to 1.0 (at end)
    """
    pl, plo = h3_to_lat_lon(point_h3)
    sl, slo = h3_to_lat_lon(start_h3)
    el, elo = h3_to_lat_lon(end_h3)
    dx = elo - slo
    dy = el - sl
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return 0.0
    frac = ((plo - slo) * dx + (pl - sl) * dy) / denom
    return max(0.0, min(1.0, frac))


def h3_distance(h3_a: str, h3_b: str) -> float:
    """
    Calculate great circle distance between two H3 cells.

    Args:
        h3_a, h3_b: H3 cell indices

    Returns:
        Distance in meters
    """
    lat_a, lon_a = h3_to_lat_lon(h3_a)
    lat_b, lon_b = h3_to_lat_lon(h3_b)
    return great_circle_distance(lat_a, lon_a, lat_b, lon_b)


def get_h3_neighbors(h3_index: str, k: int = 1) -> set:
    """
    Get neighboring H3 cells at distance k.

    Args:
        h3_index: H3 cell index
        k: Ring distance (1 = immediate neighbors)

    Returns:
        Set of H3 cell indices
    """
    return set(h3.grid_ring(h3_index, k))


def cells_within_radius(center_h3: str, radius_m: float, max_resolution: int = 8) -> set:
    """
    Get all H3 cells within a given radius of a center cell.

    Args:
        center_h3: Center H3 cell index
        radius_m: Radius in meters
        max_resolution: H3 resolution to use

    Returns:
        Set of H3 cell indices within radius
    """
    # Start with center cell
    result = {center_h3}

    # Expand rings until we exceed the radius
    k = 1
    while True:
        try:
            ring = h3.grid_ring(center_h3, k)
            if not ring:
                break

            # Check if any cell in ring is within radius
            ring_distances = [h3_distance(center_h3, cell) for cell in ring]
            if min(ring_distances) > radius_m:
                break

            # Add cells within radius
            for cell, distance in zip(ring, ring_distances):
                if distance <= radius_m:
                    result.add(cell)

            k += 1

            # Safety check to prevent infinite loops
            if k > 100:
                break

        except Exception:
            break

    return result
