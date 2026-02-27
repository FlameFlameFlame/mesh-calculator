"""
Route-based tower placement pipeline.

Processes user-chosen routes sequentially, placing towers on each corridor
while reusing towers from previously processed routes. Computes visibility
edges, cell coverage, and optionally tags city links.
"""
import json
import logging
import os
from typing import List, Optional

import h3

from ..core.config import MeshConfig, RouteSpec
from ..core.elevation import ElevationProvider
from ..core.grid import H3Cell
from ..core.road_corridor import road_geojson_to_h3_corridor
from ..data.cache import LOSCache
from ..data.exporters import (
    export_coverage_geojson,
    export_tower_coverage_geojson,
    export_towers_geojson,
    export_visibility_edges_geojson,
    generate_report,
)
from ..network.graph import MeshSurface
from ..optimization.city_coverage import tag_city_links
from ..optimization.corridor import install_nodes, place_nodes_along_corridor

logger = logging.getLogger(__name__)


def _build_corridor_cells(
    corridor: List[str],
    elevation_provider: ElevationProvider,
    existing_cells: dict,
) -> dict:
    """
    Build H3Cell objects for corridor cells not already in existing_cells.

    Args:
        corridor: Ordered list of H3 indices.
        elevation_provider: For terrain elevation lookups.
        existing_cells: Already-created cells (skipped to avoid overwriting).

    Returns:
        Dict of new H3Cell objects keyed by H3 index.
    """
    new_cells = {}
    for h3_idx in corridor:
        if h3_idx in existing_cells or h3_idx in new_cells:
            continue
        lat, lon = h3.cell_to_latlng(h3_idx)
        elevation = elevation_provider.get_elevation(lat, lon)
        new_cells[h3_idx] = H3Cell(
            h3_index=h3_idx,
            lat=lat,
            lon=lon,
            elevation=elevation,
            has_road=True,
            is_in_boundary=True,
        )
    return new_cells


def _trim_unfit_ends(corridor: List[str], cells: dict):
    """
    Remove cells marked is_in_unfit_area from both ends of the corridor.

    Returns (trimmed_corridor, entry1_h3, entry2_h3).
    entry1_h3 and entry2_h3 are the first and last non-unfit cells — these
    are the city boundary crossing points and become anchor tower locations.
    Returns ([], None, None) if the entire corridor is unfit.
    """
    start_idx = None
    for i, h3_idx in enumerate(corridor):
        cell = cells.get(h3_idx)
        if cell and not cell.is_in_unfit_area:
            start_idx = i
            break
    if start_idx is None:
        return [], None, None

    end_idx = None
    for i in range(len(corridor) - 1, -1, -1):
        cell = cells.get(corridor[i])
        if cell and not cell.is_in_unfit_area:
            end_idx = i
            break

    if end_idx is None or start_idx > end_idx:
        return [], None, None

    trimmed = corridor[start_idx:end_idx + 1]
    return trimmed, corridor[start_idx], corridor[end_idx]


def _mark_city_cells(
    cells: dict,
    h3_indices: List[str],
    city_boundaries_geojson: dict,
) -> None:
    """
    Mark corridor cells that lie inside city boundary polygons as unfit.

    Sets is_in_unfit_area=True on matching H3Cell objects in ``cells``.
    """
    from shapely.geometry import Point, shape

    polys = [
        shape(f['geometry'])
        for f in city_boundaries_geojson.get('features', [])
        if 'geometry' in f
    ]
    if not polys:
        return

    marked = 0
    for h3_idx in h3_indices:
        cell = cells.get(h3_idx)
        if cell is None or cell.is_in_unfit_area:
            continue
        if any(Point(cell.lon, cell.lat).within(p) for p in polys):
            cell.is_in_unfit_area = True
            marked += 1

    if marked:
        logger.info(
            "Marked %d corridor cells as unfit (inside city boundaries)", marked
        )


def run_route_pipeline(
    routes: List[RouteSpec],
    mesh_config: MeshConfig,
    elevation_path: str,
    city_boundaries_geojson: Optional[dict] = None,
    output_dir: str = "output",
) -> dict:
    """
    Run the full route-based tower placement pipeline.

    Routes are processed sequentially. Towers placed on earlier routes are
    reused as free relay points for later routes (via place_nodes_along_corridor's
    existing-tower-reuse logic).

    After all routes are processed:
    - Visibility edges are computed between all towers (with path loss).
    - Per-cell coverage metrics are computed (path loss per hex).
    - Optionally, towers are tagged with city_link if ≥20% coverage in a city.
    - Results are exported to output_dir.

    Args:
        routes: List of RouteSpec objects to process in order.
        mesh_config: Physics and grid configuration. max_towers_per_route is
                     overridden per-route by RouteSpec.max_towers_per_route.
        elevation_path: Path to GeoTIFF elevation file.
        city_boundaries_geojson: Optional GeoJSON FeatureCollection with city
                                 boundary polygons for city link tagging.
        output_dir: Directory to write output files.

    Returns:
        Summary dict with tower count, route count, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Starting route pipeline: %d route(s), output=%s",
        len(routes), output_dir,
    )

    elevation_provider = ElevationProvider(elevation_path)
    surface = MeshSurface({}, mesh_config, elevation_provider)
    los_cache = LOSCache()

    route_summaries = []

    for route in routes:
        logger.info(
            "Processing route '%s': %d features, max_towers=%d",
            route.route_id, len(route.features), route.max_towers_per_route,
        )

        # Convert GeoJSON features to ordered H3 corridor (site1→site2)
        corridor = road_geojson_to_h3_corridor(
            route.features,
            mesh_config.h3_resolution,
            site1=route.site1,
            site2=route.site2,
        )

        if len(corridor) < 2:
            logger.warning(
                "Route '%s' produced corridor with <2 cells — skipping", route.route_id
            )
            route_summaries.append({
                'route_id': route.route_id,
                'corridor_cells': len(corridor),
                'towers_placed': 0,
                'skipped': True,
            })
            continue

        logger.info("Corridor: %d cells for route '%s'", len(corridor), route.route_id)

        # Determine site H3 indices (for logging only — no corridor extension)
        if (route.site1 and 'lat' in route.site1
                and route.site2 and 'lat' in route.site2):
            site1_h3 = h3.latlng_to_cell(
                route.site1['lat'], route.site1['lon'], mesh_config.h3_resolution)
            site2_h3 = h3.latlng_to_cell(
                route.site2['lat'], route.site2['lon'], mesh_config.h3_resolution)
            logger.info(
                "Site cells for route '%s': %s → %s",
                route.route_id, site1_h3, site2_h3,
            )
            # Do NOT extend corridor to site with h3.grid_path_cells —
            # that creates off-road straight-line paths through cities.

        # Build cells for this corridor and add to shared surface
        new_cells = _build_corridor_cells(corridor, elevation_provider, surface.cells)
        surface.cells.update(new_cells)
        logger.info(
            "Added %d new cells (total: %d)", len(new_cells), len(surface.cells)
        )

        # Mark city-interior corridor cells as unfit
        if city_boundaries_geojson:
            _mark_city_cells(surface.cells, corridor, city_boundaries_geojson)

        # Trim city-interior cells from both corridor ends.
        # Anchor towers go at the boundary entry cells (city edge), not inside the city.
        trimmed_corridor, entry1_h3, entry2_h3 = _trim_unfit_ends(corridor, surface.cells)
        if len(trimmed_corridor) < 2:
            logger.warning(
                "Route '%s' corridor has no eligible cells after city trimming — skipping",
                route.route_id,
            )
            route_summaries.append({
                'route_id': route.route_id,
                'corridor_cells': len(corridor),
                'towers_new': 0,
                'towers_reused': 0,
                'skipped': True,
            })
            continue

        logger.info(
            "Trimmed corridor: %d → %d cells (removed %d unfit city cells)",
            len(corridor), len(trimmed_corridor), len(corridor) - len(trimmed_corridor),
        )

        # Place anchor towers at city boundary entry points
        for anchor_h3 in filter(None, [entry1_h3, entry2_h3]):
            if anchor_h3 in surface.cells and anchor_h3 not in surface.tower_by_h3:
                surface.place_tower(anchor_h3, source='site')
                logger.info("Placed city-boundary anchor tower at %s", anchor_h3)

        # Override max towers for this route
        saved_max = mesh_config.max_towers_per_route
        mesh_config.max_towers_per_route = route.max_towers_per_route

        # Place towers along trimmed corridor (city interior already excluded)
        towers_before = len(surface.towers)
        placed = place_nodes_along_corridor(trimmed_corridor, surface, los_cache)
        install_nodes(placed, surface, source=route.route_id)
        towers_after = len(surface.towers)

        mesh_config.max_towers_per_route = saved_max

        new_tower_count = towers_after - towers_before
        logger.info(
            "Route '%s': %d new tower(s) placed (%d reused)",
            route.route_id, new_tower_count, len(placed) - new_tower_count,
        )

        route_summaries.append({
            'route_id': route.route_id,
            'corridor_cells': len(corridor),
            'towers_new': new_tower_count,
            'towers_reused': len(placed) - new_tower_count,
        })

    # Compute visibility edges between all towers
    logger.info("Computing visibility edges for %d tower(s)...", len(surface.towers))
    surface.update_visibility_edges(los_cache)

    # Compute per-cell coverage (path loss per hex)
    logger.info("Computing cell coverage...")
    surface.compute_cell_coverage(los_cache)

    # Compute tower radial coverage (all hexes within signal range, not just roads)
    logger.info("Computing tower radial coverage...")
    radial_hexes = surface.compute_tower_radial_coverage(los_cache)

    # Tag city links
    if city_boundaries_geojson:
        tag_city_links(surface, city_boundaries_geojson, threshold=0.20)

    # Export results
    towers_path = os.path.join(output_dir, 'towers.geojson')
    coverage_path = os.path.join(output_dir, 'coverage.geojson')
    tower_coverage_path = os.path.join(output_dir, 'tower_coverage.geojson')
    edges_path = os.path.join(output_dir, 'visibility_edges.geojson')
    report_path = os.path.join(output_dir, 'report.json')

    export_towers_geojson(surface, towers_path)
    export_coverage_geojson(surface, coverage_path)
    export_tower_coverage_geojson(radial_hexes, tower_coverage_path)
    export_visibility_edges_geojson(surface, edges_path)
    generate_report(surface, report_path)

    cache_stats = los_cache.stats()
    elev_stats = elevation_provider.cache_stats()

    summary = {
        'routes_processed': len(routes),
        'total_towers': len(surface.towers),
        'total_cells': len(surface.cells),
        'visibility_edges': surface.visibility_graph.edge_count(),
        'route_summaries': route_summaries,
        'los_cache': cache_stats,
        'elevation_cache': elev_stats,
    }

    logger.info(
        "Pipeline complete: %d towers, %d cells, %d visibility edges",
        summary['total_towers'], summary['total_cells'], summary['visibility_edges'],
    )
    return summary
