"""
City link detection for tower coverage.

Tags towers as having a city link when ≥20% of their coverage cells
fall within a city boundary polygon.
"""
import logging
from typing import List

from shapely.geometry import Point, shape

from ..network.graph import MeshSurface

logger = logging.getLogger(__name__)


def tag_city_links(
    surface: MeshSurface,
    city_boundaries_geojson: dict,
    threshold: float = 0.20,
) -> None:
    """
    Tag towers with city_link=True when their coverage area overlaps a city
    boundary by at least `threshold` fraction.

    Coverage cells are those whose closest_tower_id matches the tower.
    Uses Shapely Point.within(polygon) for containment check.

    Args:
        surface: MeshSurface with placed towers and computed cell coverage.
                 compute_cell_coverage() must be called first so that
                 closest_tower_id is populated.
        city_boundaries_geojson: GeoJSON FeatureCollection with city boundary
                                 Polygon/MultiPolygon features.
        threshold: Minimum fraction of coverage cells that must fall within
                   a city polygon for the tower to receive city_link=True.
                   Default: 0.20 (20%).
    """
    features = city_boundaries_geojson.get('features', [])
    if not features:
        logger.warning("city_boundaries_geojson has no features — skipping city link tagging")
        return

    # Build Shapely polygons from city boundary features
    city_polygons = []
    for feat in features:
        try:
            geom = shape(feat['geometry'])
            city_polygons.append(geom)
        except Exception as exc:
            logger.warning("Could not parse city boundary feature: %s", exc)

    if not city_polygons:
        logger.warning("No valid city polygons parsed — skipping city link tagging")
        return

    logger.info(
        "Tagging city links: %d towers, %d city polygons, threshold=%.0f%%",
        len(surface.towers), len(city_polygons), threshold * 100,
    )

    # Group cells by closest_tower_id
    cells_by_tower: dict[int, list] = {tid: [] for tid in surface.towers}
    for cell in surface.cells.values():
        if cell.closest_tower_id is not None and cell.closest_tower_id in cells_by_tower:
            cells_by_tower[cell.closest_tower_id].append(cell)

    tagged = 0
    for tower in surface.towers.values():
        coverage_cells = cells_by_tower.get(tower.tower_id, [])
        if not coverage_cells:
            continue

        inside = 0
        for cell in coverage_cells:
            pt = Point(cell.lon, cell.lat)
            if any(pt.within(poly) for poly in city_polygons):
                inside += 1

        fraction = inside / len(coverage_cells)
        if fraction >= threshold:
            tower.city_link = True
            tagged += 1
            logger.debug(
                "Tower %d tagged city_link (%.0f%% of %d coverage cells inside city)",
                tower.tower_id, fraction * 100, len(coverage_cells),
            )

    logger.info("City link tagging complete: %d/%d towers tagged", tagged, len(surface.towers))
