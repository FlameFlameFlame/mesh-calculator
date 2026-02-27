"""
Convert GeoJSON route features to an ordered H3 cell sequence (corridor).

Unlike find_h3_cells_on_roads() which returns an unordered set, this module
preserves the spatial order along the route — required by place_nodes_along_corridor().
"""
import logging
import math
from typing import List, Optional

import h3

logger = logging.getLogger(__name__)


def _order_features(
    features: List[dict],
    s1_lat: float, s1_lon: float,
    s2_lat: float, s2_lon: float,
) -> List[dict]:
    """
    Return features reordered so their geometry forms a continuous chain
    from (s1_lat, s1_lon) toward (s2_lat, s2_lon).

    Algorithm: greedy head-to-tail stitching (same as path_profile in app.py).
    1. Pick the feature whose start/end is closest to site1 as first segment
       (flip it if its end is closer to site1 than its start).
    2. Greedily append the remaining feature whose start/end is closest to
       the current chain tail, flipping if needed.
    """
    def _hav(la1: float, lo1: float, la2: float, lo2: float) -> float:
        R = 6371000.0
        dlat = math.radians(la2 - la1)
        dlon = math.radians(lo2 - lo1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
             * math.sin(dlon / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(max(0.0, a)))

    def _endpoints(feat: dict):
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [])
        if geom.get('type') == 'MultiLineString':
            coords = coords[0] if coords else []
        if len(coords) < 2:
            return None, None
        # coords are [lon, lat]
        return (coords[0][1], coords[0][0]), (coords[-1][1], coords[-1][0])

    def _flip(feat: dict) -> dict:
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [])
        return dict(feat, geometry=dict(geom, coordinates=list(reversed(coords))))

    valid = [f for f in features if _endpoints(f)[0] is not None]
    if not valid:
        return features

    # Pick first feature: endpoint closest to site1
    best_i, best_d, best_flip = 0, float('inf'), False
    for i, f in enumerate(valid):
        p_start, p_end = _endpoints(f)
        d_s = _hav(s1_lat, s1_lon, p_start[0], p_start[1])
        d_e = _hav(s1_lat, s1_lon, p_end[0], p_end[1])
        if d_s < best_d:
            best_d, best_i, best_flip = d_s, i, False
        if d_e < best_d:
            best_d, best_i, best_flip = d_e, i, True

    first = valid.pop(best_i)
    ordered = [_flip(first) if best_flip else first]

    # Greedily chain remaining features to tail
    while valid:
        _, tail = _endpoints(ordered[-1])
        best_i, best_d, best_flip = 0, float('inf'), False
        for i, f in enumerate(valid):
            p_start, p_end = _endpoints(f)
            d_s = _hav(tail[0], tail[1], p_start[0], p_start[1])
            d_e = _hav(tail[0], tail[1], p_end[0], p_end[1])
            if d_s < best_d:
                best_d, best_i, best_flip = d_s, i, False
            if d_e < best_d:
                best_d, best_i, best_flip = d_e, i, True
        nxt = valid.pop(best_i)
        ordered.append(_flip(nxt) if best_flip else nxt)

    logger.debug("_order_features: %d features ordered site1→site2", len(ordered))
    return ordered


def road_geojson_to_h3_corridor(
    features: List[dict],
    resolution: int,
    site1: Optional[dict] = None,
    site2: Optional[dict] = None,
) -> List[str]:
    """
    Convert a list of GeoJSON LineString/MultiLineString features to an ordered
    H3 cell sequence.

    Sampling density matches the _sample_line_to_h3 logic in core/grid.py:
    half the average hex edge length converted to degrees.

    Consecutive duplicate H3 indices are removed to keep the corridor compact.

    Args:
        features: List of GeoJSON feature dicts (type: Feature with LineString
                  or MultiLineString geometry).
        resolution: H3 resolution level.
        site1: Optional dict with 'lat'/'lon' keys for route start.
        site2: Optional dict with 'lat'/'lon' keys for route end.
               When both are provided, features are spatially ordered
               from site1 toward site2 before sampling.

    Returns:
        Ordered list of H3 cell indices along the route. May be empty if
        features contain no valid coordinates.
    """
    if site1 and site2:
        features = _order_features(
            features,
            site1['lat'], site1['lon'],
            site2['lat'], site2['lon'],
        )

    edge_m = h3.average_hexagon_edge_length(resolution, unit='m')
    # Convert metres to approximate degrees (1 deg ≈ 111 320 m)
    sample_spacing = (edge_m * 0.5) / 111_320.0

    ordered: List[str] = []

    for feature in features:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type', '')
        coords_list = geom.get('coordinates', [])

        if geom_type == 'LineString':
            _sample_linestring(coords_list, sample_spacing, resolution, ordered)
        elif geom_type == 'MultiLineString':
            for line_coords in coords_list:
                _sample_linestring(line_coords, sample_spacing, resolution, ordered)
        else:
            logger.warning("Skipping unsupported geometry type: %s", geom_type)

    # Deduplicate consecutive identical indices while preserving order
    deduped: List[str] = []
    for cell in ordered:
        if not deduped or deduped[-1] != cell:
            deduped.append(cell)

    logger.info(
        "road_geojson_to_h3_corridor: %d features → %d cells (res=%d)",
        len(features), len(deduped), resolution,
    )
    return deduped


def _sample_linestring(
    coords: list,
    sample_spacing: float,
    resolution: int,
    output: List[str],
) -> None:
    """
    Sample points along a coordinate sequence and append H3 cells to output.

    Args:
        coords: List of [lon, lat] pairs.
        sample_spacing: Spacing between sample points in degrees.
        resolution: H3 resolution level.
        output: List to append H3 indices to (modified in place).
    """
    if len(coords) < 2:
        return

    for seg_start, seg_end in zip(coords, coords[1:]):
        lon0, lat0 = seg_start[0], seg_start[1]
        lon1, lat1 = seg_end[0], seg_end[1]

        seg_len = ((lon1 - lon0) ** 2 + (lat1 - lat0) ** 2) ** 0.5
        if seg_len == 0:
            continue

        num_samples = max(int(seg_len / sample_spacing) + 1, 2)
        for i in range(num_samples):
            frac = i / (num_samples - 1)
            lat = lat0 + frac * (lat1 - lat0)
            lon = lon0 + frac * (lon1 - lon0)
            cell = h3.latlng_to_cell(lat, lon, resolution)
            output.append(cell)
