"""
Route-based tower placement pipeline.

Processes user-chosen routes sequentially, placing towers on each corridor
while reusing towers from previously processed routes. Computes visibility
edges, road-cell coverage, and optionally tags city links.
"""
import json
import logging
import os
import time
from typing import Callable, List, Optional

import h3

from ..core.config import MeshConfig, RouteSpec
from ..core.grid_provider import GridProvider
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


def _locate_site_cell(
    grid_provider: GridProvider,
    lat: float,
    lon: float,
    mesh_config: MeshConfig,
    *,
    prefer_road: bool = False,
) -> str:
    return grid_provider.locate_adaptive_cell(
        float(lat),
        float(lon),
        mesh_config.h3_resolution,
        mesh_config,
        prefer_road=prefer_road,
    )


def _prepare_cells_with_buffer(
    corridor: List[str],
    mesh_config: MeshConfig,
    grid_provider: GridProvider,
    existing_cells: dict,
) -> tuple[dict, dict, dict, dict]:
    """
    Build corridor + buffer cells in one provider batch materialization call.

    Returns:
        (new_corridor_cells, new_buffer_cells, materialize_stats, prep_stats)
    """
    prep_started = time.perf_counter()
    existing = set(existing_cells.keys())
    corridor_set = set(corridor)
    missing_corridor = corridor_set - existing

    buffer_candidates: set[str] = set()
    if mesh_config.road_buffer_m > 0:
        full_pool = grid_provider.get_adaptive_full_cells(
            mesh_config.h3_resolution,
            mesh_config,
        )
        expanded = grid_provider.adaptive_union_within_radius(
            corridor,
            mesh_config.road_buffer_m,
            mesh_config.h3_resolution,
            mesh_config,
            candidate_cells=full_pool,
        )
        buffer_candidates = expanded - corridor_set - existing

    all_new = missing_corridor | buffer_candidates
    cells, mat_stats = grid_provider.materialize_cells(
        all_new,
        mesh_config,
        road_cells=missing_corridor,
        is_in_boundary=True,
        include_stats=True,
    )
    new_corridor = {h3_idx: cells[h3_idx] for h3_idx in missing_corridor if h3_idx in cells}
    new_buffer = {h3_idx: cells[h3_idx] for h3_idx in buffer_candidates if h3_idx in cells}
    prep_stats = {
        "corridor_cells_new": len(new_corridor),
        "buffer_cells_new": len(new_buffer),
        "prepared_cells_total": len(cells),
        "prepare_cells_and_buffer_s": time.perf_counter() - prep_started,
    }
    logger.info(
        "Buffer expansion: added %d new cells (buffer_m=%.0f)",
        len(new_buffer),
        mesh_config.road_buffer_m,
    )
    return new_corridor, new_buffer, mat_stats, prep_stats


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
    grid_provider: GridProvider,
    city_boundaries_geojson: Optional[dict] = None,
    boundary_geojson: Optional[dict] = None,
    output_dir: str = "output",
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
        grid_provider: Ready grid provider with elevation + multi-resolution grid access.
        city_boundaries_geojson: Optional GeoJSON FeatureCollection with city
                                 boundary polygons for city link tagging.
        boundary_geojson: Optional boundary geometry used for full-grid export.
        output_dir: Directory to write output files.
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

    if grid_provider is None:
        raise ValueError("run_route_pipeline requires grid_provider")

    base_h3_resolution = mesh_config.h3_resolution
    adaptive_summary = grid_provider.adaptive_resolution_summary(
        base_h3_resolution,
        mesh_config,
    )
    effective_h3_resolution = adaptive_summary.get(
        "effective_h3_resolution_max",
        base_h3_resolution,
    )
    h3_auto_refined = effective_h3_resolution > base_h3_resolution
    h3_auto_refine_reason = None
    if h3_auto_refined:
        h3_auto_refine_reason = (
            f"adaptive_mixed_{adaptive_summary.get('effective_h3_resolution_min')}"
            f"-{adaptive_summary.get('effective_h3_resolution_max')}"
        )
        logger.info(
            "Adaptive mixed H3 mesh active: base=%d range=%d-%d cells_by_resolution=%s",
            base_h3_resolution,
            adaptive_summary.get("effective_h3_resolution_min", base_h3_resolution),
            adaptive_summary.get("effective_h3_resolution_max", base_h3_resolution),
            adaptive_summary.get("cells_by_resolution", {}),
        )

    surface = MeshSurface({}, mesh_config, grid_provider)
    los_cache = LOSCache()

    route_summaries = []
    prep_metrics = {
        "corridor_extract_s": 0.0,
        "prepare_cells_and_buffer_s": 0.0,
        "trim_unfit_s": 0.0,
        "prepared_cells_total": 0,
        "materialized_cache_hits": 0,
        "materialized_from_static": 0,
        "materialized_from_dem": 0,
    }

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
        t_corridor = time.perf_counter()
        corridor = grid_provider.corridor_from_features(
            route.features,
            mesh_config.h3_resolution,
            site1=route.site1,
            site2=route.site2,
            config=mesh_config,
        )
        prep_metrics["corridor_extract_s"] += time.perf_counter() - t_corridor
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
            site1_h3 = _locate_site_cell(
                grid_provider,
                float(route.site1['lat']),
                float(route.site1['lon']),
                mesh_config,
                prefer_road=True,
            )
            site2_h3 = _locate_site_cell(
                grid_provider,
                float(route.site2['lat']),
                float(route.site2['lon']),
                mesh_config,
                prefer_road=True,
            )
            logger.info(
                "Site cells for route '%s': %s → %s",
                route.route_id, site1_h3, site2_h3,
            )
            # Do NOT extend corridor to site with h3.grid_path_cells —
            # that creates off-road straight-line paths through cities.

        # Build corridor + buffer cells in one provider materialization batch.
        new_cells, buffer_cells, materialize_stats, prep_stats = _prepare_cells_with_buffer(
            corridor,
            mesh_config,
            grid_provider,
            surface.cells,
        )
        surface.cells.update(new_cells)
        surface.cells.update(buffer_cells)
        prep_metrics["prepare_cells_and_buffer_s"] += prep_stats["prepare_cells_and_buffer_s"]
        prep_metrics["prepared_cells_total"] += prep_stats["prepared_cells_total"]
        prep_metrics["materialized_cache_hits"] += materialize_stats["cache_hits"]
        prep_metrics["materialized_from_static"] += materialize_stats["from_static"]
        prep_metrics["materialized_from_dem"] += materialize_stats["from_dem"]
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
        logger.info(
            "Route prep stats: prepared=%d cache_hits=%d from_static=%d from_dem=%d prep_s=%.3f",
            prep_stats["prepared_cells_total"],
            materialize_stats["cache_hits"],
            materialize_stats["from_static"],
            materialize_stats["from_dem"],
            prep_stats["prepare_cells_and_buffer_s"],
        )

        # Mark city-interior corridor cells as unfit
        t_trim = time.perf_counter()
        if city_boundaries_geojson:
            _mark_city_cells(surface.cells, corridor, city_boundaries_geojson)

        # Trim city-interior cells from both corridor ends.
        # Anchor towers go at the boundary entry cells (city edge), not inside the city.
        trimmed_corridor, entry1_h3, entry2_h3 = _trim_unfit_ends(corridor, surface.cells)
        prep_metrics["trim_unfit_s"] += time.perf_counter() - t_trim
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

        def _site_height_m(site: dict) -> float:
            try:
                return max(0.0, float(site.get('site_height_m', 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0

        def _apply_site_offset(cell_h3: str, site_height_m: float, site_name: str) -> None:
            cell = surface.cells.get(cell_h3)
            if cell is None:
                return
            prev = float(getattr(cell, 'antenna_height_offset_m', 0.0) or 0.0)
            cell.antenna_height_offset_m = max(prev, site_height_m)
            if cell.antenna_height_offset_m > prev:
                logger.info(
                    "Updated anchor antenna offset at %s for %s: %.2f m",
                    cell_h3,
                    site_name,
                    cell.antenna_height_offset_m,
                )

        # entry1_h3 corresponds to site1 end, entry2_h3 to site2 end
        for site, entry_h3 in [
            (route.site1, entry1_h3),
            (route.site2, entry2_h3),
        ]:
            if not site or 'lat' not in site:
                continue

            endpoint_site_height_m = _site_height_m(site)

            if _site_is_city(site):
                # City site: anchor at boundary entry cell (one per road entry)
                if entry_h3 is None or entry_h3 not in surface.cells:
                    continue
                edge_m = h3.average_hexagon_edge_length(
                    int(h3.get_resolution(entry_h3)),
                    unit='m',
                )
                neighbors_1ring = grid_provider.adaptive_cells_within_radius(
                    entry_h3,
                    max(edge_m * 1.2, 1.0),
                    mesh_config.h3_resolution,
                    mesh_config,
                    candidate_cells=set(surface.tower_by_h3.keys()),
                )
                existing_anchor_h3 = next(
                    (nb for nb in neighbors_1ring if nb in surface.tower_by_h3),
                    None,
                )
                if existing_anchor_h3 is not None:
                    _apply_site_offset(
                        existing_anchor_h3,
                        endpoint_site_height_m,
                        site.get('name', ''),
                    )
                    logger.info(
                        "Skipping city-boundary anchor at %s — nearby tower exists",
                        entry_h3,
                    )
                    continue
                _apply_site_offset(
                    entry_h3,
                    endpoint_site_height_m,
                    site.get('name', ''),
                )
                surface.place_tower(entry_h3, source='site')
                logger.info(
                    "Placed city-boundary anchor at %s (%s)",
                    entry_h3, site.get('name', ''),
                )
            else:
                # Non-city site: single anchor at the exact site H3 cell
                site_h3 = _locate_site_cell(
                    grid_provider,
                    float(site['lat']),
                    float(site['lon']),
                    mesh_config,
                )
                if site_h3 in surface.tower_by_h3:
                    _apply_site_offset(
                        site_h3,
                        endpoint_site_height_m,
                        site.get('name', ''),
                    )
                    continue
                if site_h3 not in surface.cells:
                    surface.cells.update(
                        grid_provider.materialize_cells(
                            [site_h3],
                            mesh_config,
                            road_cells=set(),
                            is_in_boundary=False,
                        )
                    )
                _apply_site_offset(
                    site_h3,
                    endpoint_site_height_m,
                    site.get('name', ''),
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
            'prepared_cells_total': prep_stats["prepared_cells_total"],
            'prepare_cells_and_buffer_s': prep_stats["prepare_cells_and_buffer_s"],
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
    grid_cells_full_path = os.path.join(output_dir, 'grid_cells_full.geojson')
    gap_repair_hexes_path = os.path.join(output_dir, 'gap_repair_hexes.geojson')

    export_towers_geojson(surface, towers_path)
    export_coverage_geojson(surface, coverage_path)
    export_visibility_edges_geojson(surface, edges_path)
    export_grid_cells_geojson(
        surface.cells,
        grid_cells_path,
        base_h3_resolution=base_h3_resolution,
        effective_h3_resolution=effective_h3_resolution,
    )
    if mesh_config.export_full_grid_cells and boundary_geojson:
        try:
            full_grid_cells = grid_provider.build_full_boundary_grid(mesh_config)
            export_grid_cells_geojson(
                full_grid_cells,
                grid_cells_full_path,
                base_h3_resolution=base_h3_resolution,
                effective_h3_resolution=effective_h3_resolution,
            )
        except Exception:
            logger.warning("Skipping full-grid export: provider full-grid build failed", exc_info=True)
    if surface.gap_repair_debug:
        export_gap_repair_hexes_geojson(surface.gap_repair_debug, gap_repair_hexes_path)
    generate_report(surface, report_path)
    _emit_progress('export', 'Outputs exported', 100.0)

    cache_stats = los_cache.stats()
    elev_stats = grid_provider.cache_stats()

    summary = {
        'routes_processed': len(routes),
        'total_towers': len(surface.towers),
        'total_cells': len(surface.cells),
        'visibility_edges': surface.visibility_graph.edge_count(),
        'num_clusters': len(surface.visibility_graph.connected_components()),
        'h3_resolution_mode': adaptive_summary.get('h3_resolution_mode', 'fixed'),
        'base_h3_resolution': base_h3_resolution,
        'effective_h3_resolution_min': adaptive_summary.get(
            'effective_h3_resolution_min',
            base_h3_resolution,
        ),
        'effective_h3_resolution_max': adaptive_summary.get(
            'effective_h3_resolution_max',
            base_h3_resolution,
        ),
        'cells_by_resolution': adaptive_summary.get('cells_by_resolution', {}),
        'effective_h3_resolution': effective_h3_resolution,
        'h3_auto_refined': h3_auto_refined,
        'h3_auto_refine_reason': h3_auto_refine_reason,
        'route_summaries': route_summaries,
        'prep_metrics': prep_metrics,
        'prepared_cells_total': prep_metrics["prepared_cells_total"],
        'materialized_cache_hits': prep_metrics["materialized_cache_hits"],
        'materialized_from_static': prep_metrics["materialized_from_static"],
        'materialized_from_dem': prep_metrics["materialized_from_dem"],
        'los_cache': cache_stats,
        'elevation_cache': elev_stats,
    }

    logger.info(
        "Pipeline complete: %d towers, %d cells, %d visibility edges",
        summary['total_towers'], summary['total_cells'], summary['visibility_edges'],
    )
    _emit_progress('done', 'Pipeline complete', 100.0)
    return summary
