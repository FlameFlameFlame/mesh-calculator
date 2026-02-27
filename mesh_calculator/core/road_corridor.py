"""
Convert GeoJSON route features to an ordered H3 cell sequence (corridor).

Unlike find_h3_cells_on_roads() which returns an unordered set, this module
preserves the spatial order along the route — required by place_nodes_along_corridor().
"""
import logging
from typing import List

import h3

logger = logging.getLogger(__name__)


def road_geojson_to_h3_corridor(features: List[dict], resolution: int) -> List[str]:
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

    Returns:
        Ordered list of H3 cell indices along the route. May be empty if
        features contain no valid coordinates.
    """
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
