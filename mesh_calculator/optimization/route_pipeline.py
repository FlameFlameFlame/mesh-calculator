"""
Route-based tower placement pipeline.

Processes user-chosen routes sequentially, placing towers on each corridor
while reusing towers from previously processed routes. Computes visibility
edges, road-cell coverage, and optionally tags city links.
"""
import json
import logging
import os
from typing import Callable, List, Optional

import h3

from ..core.config import MeshConfig, RouteSpec
from ..core.elevation import ElevationProvider
from ..core.geometry import h3_to_lat_lon
from ..core.grid import H3Cell
from ..core.road_corridor import road_geojson_to_h3_corridor
from ..data.cache import LOSCache
from ..data.exporters import (
    export_coverage_geojson,
    export_gap_repair_hexes_geojson,
    export_grid_cells_geojson,
    export_towers_geojson,
    export_visibility_edges_geojson,
    generate_report,
)
from ..network.graph import MeshSurface
from ..optimization.city_coverage import tag_city_links
from ..optimization.corridor import (
    install_nodes,
    place_nodes_along_corridor,
    wire_corridor_edges,
)

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


def _expand_cells_with_buffer(
    road_cells: List[str],
    mesh_config: MeshConfig,
    elevation_provider: ElevationProvider,
    existing_cells: dict,
) -> dict:
    """
    Expand corridor cells with a spatial buffer at the main H3 resolution.

    Uses grid_disk at the working resolution — k rings of neighbors around
    each road cell. Each ring step adds ~one hex-edge-length of radius.

    Returns:
        Dict of new H3Cell objects (not already in existing_cells).
    """
    if mesh_config.road_buffer_m <= 0:
        return {}

    edge_m = h3.average_hexagon_edge_length(mesh_config.h3_resolution, unit='m')
    buffer_rings = max(1, round(mesh_config.road_buffer_m / edge_m))

    road_cell_set = set(road_cells)
    new_main_cells: set = set()
    for road_cell in road_cells:
        for neighbor in h3.grid_disk(road_cell, buffer_rings):
            if neighbor not in road_cell_set and neighbor not in existing_cells:
                new_main_cells.add(neighbor)

    new_cells = {}
    for h3_idx in new_main_cells:
        if h3_idx in existing_cells:
            continue
        lat, lon = h3_to_lat_lon(h3_idx)
        elevation = elevation_provider.get_elevation(lat, lon)
        new_cells[h3_idx] = H3Cell(
            h3_index=h3_idx,
            lat=lat,
            lon=lon,
            elevation=elevation,
            has_road=False,
            is_in_boundary=True,
        )

    logger.info(
        "Buffer expansion: added %d new cells (buffer_m=%.0f, rings=%d, edge_m=%.0f)",
        len(new_cells), mesh_config.road_buffer_m, buffer_rings, edge_m,
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
    strategy: str = 'dp',
    progress_callback: Optional[Callable[[dict], None]] = None,
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
        strategy: Tower placement algorithm — 'dp' (MaxMin DP, default) or
                  'greedy' (furthest-clear-LOS, fewer towers, no gap repair).
        progress_callback: Optional callback receiving structured progress dicts.

    Returns:
        Summary dict with tower count, route count, etc.
    """
    total_routes = len(routes)
    route_weight = (80.0 / total_routes) if total_routes > 0 else 0.0
    has_city_phase = bool(
        city_boundaries_geojson
        and city_boundaries_geojson.get('features')
    )
    city_weight = 2.0 if has_city_phase else 0.0
    export_start_percent = 98.0 if has_city_phase else 96.0

    def _route_label(route: RouteSpec) -> str:
        s1 = route.site1.get('name') if isinstance(route.site1, dict) else None
        s2 = route.site2.get('name') if isinstance(route.site2, dict) else None
        if s1 and s2:
            return f"{s1} ↔ {s2} ({route.route_id})"
        return route.route_id

    def _emit_progress(
        stage: str,
        step: str,
        percent: float,
        route_index: Optional[int] = None,
        route_id: Optional[str] = None,
        route_label: Optional[str] = None,
    ) -> None:
        if progress_callback is None:
            return
        payload = {
            'stage': stage,
            'step': step,
            'percent': max(0.0, min(100.0, float(percent))),
            'route_total': total_routes,
            'route_index': route_index if route_index is not None else 0,
            'route_id': route_id,
            'route_label': route_label,
        }
        try:
            progress_callback(payload)
        except Exception:
            logger.debug("Progress callback failed", exc_info=True)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Starting route pipeline: %d route(s), output=%s",
        len(routes), output_dir,
    )
    _emit_progress('route', 'Starting route pipeline', 0.0)

    elevation_provider = ElevationProvider(elevation_path)
    surface = MeshSurface({}, mesh_config, elevation_provider)
    los_cache = LOSCache()

    route_summaries = []

    for route_idx, route in enumerate(routes, start=1):
        route_base = (route_idx - 1) * route_weight
        route_label = _route_label(route)
        _emit_progress(
            stage='route',
            step='Preparing corridor',
            percent=route_base,
            route_index=route_idx,
            route_id=route.route_id,
            route_label=route_label,
        )

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
        _emit_progress(
            stage='route',
            step='Preparing corridor',
            percent=route_base + route_weight * 0.15,
            route_index=route_idx,
            route_id=route.route_id,
            route_label=route_label,
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
            _emit_progress(
                stage='route',
                step='Route skipped (corridor too short)',
                percent=route_base + route_weight,
                route_index=route_idx,
                route_id=route.route_id,
                route_label=route_label,
            )
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

        # Expand with road buffer: add nearby off-road cells for elevated terrain
        buffer_cells = _expand_cells_with_buffer(
            corridor, mesh_config, elevation_provider, surface.cells
        )
        surface.cells.update(buffer_cells)
        _emit_progress(
            stage='route',
            step='Preparing cells and buffer',
            percent=route_base + route_weight * 0.35,
            route_index=route_idx,
            route_id=route.route_id,
            route_label=route_label,
        )

        logger.info(
            "Added %d corridor + %d buffer cells (total: %d)",
            len(new_cells), len(buffer_cells), len(surface.cells),
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
            _emit_progress(
                stage='route',
                step='Route skipped (no eligible corridor)',
                percent=route_base + route_weight,
                route_index=route_idx,
                route_id=route.route_id,
                route_label=route_label,
            )
            continue

        logger.info(
            "Trimmed corridor: %d → %d cells (removed %d unfit city cells)",
            len(corridor), len(trimmed_corridor), len(corridor) - len(trimmed_corridor),
        )

        # Place anchor towers for each site endpoint:
        #   - City site (inside a city boundary polygon): anchor at the boundary
        #     entry cell (one per road into the city). Two routes on the same road
        #     share the anchor via a 1-ring proximity dedup check.
        #   - Non-city site: single anchor placed at the site's own H3 cell.
        #     All routes sharing that site converge on the same cell.
        city_polys = []
        if city_boundaries_geojson:
            from shapely.geometry import Point, shape as _shape
            city_polys = [
                _shape(f['geometry'])
                for f in city_boundaries_geojson.get('features', [])
                if 'geometry' in f
            ]

        def _site_is_city(site: dict) -> bool:
            if not city_polys or 'lat' not in site:
                return False
            pt = Point(site['lon'], site['lat'])
            return any(pt.within(p) for p in city_polys)

        # entry1_h3 corresponds to site1 end, entry2_h3 to site2 end
        for site, entry_h3 in [
            (route.site1, entry1_h3),
            (route.site2, entry2_h3),
        ]:
            if not site or 'lat' not in site:
                continue

            if _site_is_city(site):
                # City site: anchor at boundary entry cell (one per road entry)
                if entry_h3 is None or entry_h3 not in surface.cells:
                    continue
                neighbors_1ring = h3.grid_disk(entry_h3, 1)
                if any(nb in surface.tower_by_h3 for nb in neighbors_1ring):
                    logger.info(
                        "Skipping city-boundary anchor at %s — nearby tower exists",
                        entry_h3,
                    )
                    continue
                surface.place_tower(entry_h3, source='site')
                logger.info(
                    "Placed city-boundary anchor at %s (%s)",
                    entry_h3, site.get('name', ''),
                )
            else:
                # Non-city site: single anchor at the exact site H3 cell
                site_h3 = h3.latlng_to_cell(
                    site['lat'], site['lon'], mesh_config.h3_resolution
                )
                if site_h3 in surface.tower_by_h3:
                    continue
                if site_h3 not in surface.cells:
                    lat, lon = h3.cell_to_latlng(site_h3)
                    elev = elevation_provider.get_elevation(lat, lon)
                    surface.cells[site_h3] = H3Cell(
                        h3_index=site_h3, lat=lat, lon=lon,
                        elevation=elev, has_road=False, is_in_boundary=False,
                    )
                surface.place_tower(site_h3, source='site')
                logger.info(
                    "Placed non-city site anchor at %s (%s)",
                    site_h3, site.get('name', ''),
                )

        # Override max towers for this route
        saved_max = mesh_config.max_towers_per_route
        mesh_config.max_towers_per_route = route.max_towers_per_route

        # Place towers along trimmed corridor (city interior already excluded)
        towers_before = len(surface.towers)
        placement_meta: dict = {}
        placed = place_nodes_along_corridor(
            trimmed_corridor, surface, los_cache, out_meta=placement_meta,
            strategy=strategy,
        )
        install_nodes(placed, surface, source=route.route_id,
                      placement_meta=placement_meta)
        towers_after = len(surface.towers)
        _emit_progress(
            stage='route',
            step='Placing and installing towers',
            percent=route_base + route_weight * 0.80,
            route_index=route_idx,
            route_id=route.route_id,
            route_label=route_label,
        )

        # Wire corridor-path visibility edges between consecutive placed towers.
        # The DP proved these pairs are LOS-connected along the road path; we
        # register those edges immediately so they appear in the visibility graph
        # even when straight-line LOS (used by update_visibility_edges) is blocked
        # by terrain.
        wire_corridor_edges(placed, trimmed_corridor, surface, los_cache)

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
        _emit_progress(
            stage='route',
            step='Finalizing route links',
            percent=route_base + route_weight,
            route_index=route_idx,
            route_id=route.route_id,
            route_label=route_label,
        )

    # Compute visibility edges between all towers
    _emit_progress('visibility', 'Computing visibility edges', 80.0)
    logger.info("Computing visibility edges for %d tower(s)...", len(surface.towers))
    surface.update_visibility_edges(los_cache)
    _emit_progress('visibility', 'Visibility edges computed', 88.0)

    # Compute per-cell coverage (path loss per hex)
    _emit_progress('coverage', 'Computing road-cell coverage', 88.0)
    logger.info("Computing cell coverage...")
    surface.compute_cell_coverage(los_cache)
    _emit_progress('coverage', 'Coverage computed', 96.0)

    # Tag city links
    if city_boundaries_geojson:
        _emit_progress('city_links', 'Tagging city links', 96.0)
        tag_city_links(surface, city_boundaries_geojson, threshold=0.20)
        _emit_progress('city_links', 'City links tagged', 96.0 + city_weight)

    # Export results
    _emit_progress('export', 'Exporting outputs and report', export_start_percent)
    towers_path = os.path.join(output_dir, 'towers.geojson')
    coverage_path = os.path.join(output_dir, 'coverage.geojson')
    edges_path = os.path.join(output_dir, 'visibility_edges.geojson')
    report_path = os.path.join(output_dir, 'report.json')
    grid_cells_path = os.path.join(output_dir, 'grid_cells.geojson')
    gap_repair_hexes_path = os.path.join(output_dir, 'gap_repair_hexes.geojson')

    export_towers_geojson(surface, towers_path)
    export_coverage_geojson(surface, coverage_path)
    export_visibility_edges_geojson(surface, edges_path)
    export_grid_cells_geojson(surface.cells, grid_cells_path)
    if surface.gap_repair_debug:
        export_gap_repair_hexes_geojson(surface.gap_repair_debug, gap_repair_hexes_path)
    generate_report(surface, report_path)
    _emit_progress('export', 'Outputs exported', 100.0)

    cache_stats = los_cache.stats()
    elev_stats = elevation_provider.cache_stats()

    summary = {
        'routes_processed': len(routes),
        'total_towers': len(surface.towers),
        'total_cells': len(surface.cells),
        'visibility_edges': surface.visibility_graph.edge_count(),
        'num_clusters': len(surface.visibility_graph.connected_components()),
        'route_summaries': route_summaries,
        'los_cache': cache_stats,
        'elevation_cache': elev_stats,
    }

    logger.info(
        "Pipeline complete: %d towers, %d cells, %d visibility edges",
        summary['total_towers'], summary['total_cells'], summary['visibility_edges'],
    )
    _emit_progress('done', 'Pipeline complete', 100.0)
    return summary
