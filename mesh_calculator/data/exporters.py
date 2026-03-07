"""
Data export utilities for GeoJSON and JSON output.
"""
import json
from typing import Dict
import geopandas as gpd
from shapely.geometry import Point
import h3

import structlog

from ..network.graph import MeshSurface, Tower
from ..core.grid import H3Cell

logger = structlog.get_logger(__name__)


def export_towers_geojson(surface: MeshSurface, output_path: str):
    """
    Export towers to GeoJSON file.

    Args:
        surface: Mesh surface with towers
        output_path: Output GeoJSON file path
    """
    features = []

    for tower in surface.towers.values():
        meta = tower.placement_meta or {}
        tower_cell = surface.cells.get(tower.h3_index)
        antenna_offset_m = (
            float(getattr(tower_cell, 'antenna_height_offset_m', 0.0) or 0.0)
            if tower_cell is not None
            else 0.0
        )
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [tower.lon, tower.lat]
            },
            'properties': {
                'tower_id': tower.tower_id,
                'h3_index': tower.h3_index,
                'source': tower.source,
                'route_id': tower.source,
                'virtual': tower.source in {'site'},
                'city_link': getattr(tower, 'city_link', False),
                'coverage_radius_m': getattr(tower, 'coverage_radius_m', 0.0),
                'lat': tower.lat,
                'lon': tower.lon,
                'centroid_lat': tower_cell.lat if tower_cell is not None else None,
                'centroid_lon': tower_cell.lon if tower_cell is not None else None,
                'antenna_height_m': surface.config.mast_height_m + antenna_offset_m,
                'algorithm': meta.get(
                    'algorithm',
                    'site' if tower.source == 'site' else None,
                ),
                'dp_steps': meta.get('dp_steps'),
                'repair_round': meta.get('repair_round'),
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info("Exported towers: count=%d path=%s", len(features), output_path)


def export_coverage_geojson(surface: MeshSurface, output_path: str):
    """
    Export grid cells with coverage metrics to GeoJSON.

    Args:
        surface: Mesh surface with cells
        output_path: Output GeoJSON file path
    """
    features = []

    for h3_idx, cell in surface.cells.items():
        # Get H3 cell boundary
        boundary = h3.cell_to_boundary(h3_idx)

        # Convert to lon, lat format for GeoJSON
        coords = [[lon, lat] for lat, lon in boundary]
        # Close the ring
        coords.append(coords[0])

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            },
            'properties': {
                'h3_index': h3_idx,
                'elevation': cell.elevation,
                'has_road': cell.has_road,
                'has_tower': cell.has_tower,
                'visible_tower_count': cell.visible_tower_count,
                'distance_to_closest_tower': cell.distance_to_closest_tower
                    if cell.distance_to_closest_tower != float('inf') else None,
                'clearance': cell.clearance
                    if cell.clearance is not None
                    and cell.clearance != float('inf')
                    else None,
                'path_loss': cell.path_loss,
                'received_power_dbm': cell.received_power_dbm,
                'is_covered': cell.is_covered
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info(
        "Exported coverage cells: count=%d path=%s",
        len(features),
        output_path,
    )


def export_tower_coverage_geojson(hex_results: list, output_path: str):
    """
    Export tower radial coverage hexes to GeoJSON.

    Args:
        hex_results: List of dicts from MeshSurface.compute_tower_radial_coverage()
        output_path: Output GeoJSON file path
    """
    features = []

    for r in hex_results:
        boundary = h3.cell_to_boundary(r['h3_index'])
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])  # close ring

        props = {k: v for k, v in r.items() if k not in ('lat', 'lon')}
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [coords]},
            'properties': props,
        })

    geojson = {'type': 'FeatureCollection', 'features': features}

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info(
        "Exported tower coverage hexes: count=%d path=%s",
        len(features),
        output_path,
    )


def export_grid_cells_geojson(
    cells: Dict,
    output_path: str,
    effective_h3_resolution: int | None = None,
):
    """
    Export H3 grid cells as GeoJSON polygons.

    Args:
        cells: Dict of H3 index → H3Cell
        output_path: Output GeoJSON file path
    """
    features = []
    for h3_idx, cell in cells.items():
        boundary = h3.cell_to_boundary(h3_idx)
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])
        try:
            h3_resolution = int(h3.get_resolution(h3_idx))
        except Exception:
            h3_resolution = None
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [coords]},
            'properties': {
                'h3_index': h3_idx,
                'elevation': cell.elevation,
                'has_road': cell.has_road,
                'is_in_unfit_area': cell.is_in_unfit_area,
                'h3_resolution': h3_resolution,
                'effective_h3_resolution': (
                    int(effective_h3_resolution)
                    if effective_h3_resolution is not None
                    else h3_resolution
                ),
            },
        })

    geojson = {'type': 'FeatureCollection', 'features': features}
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    logger.info("Exported grid cells: count=%d path=%s", len(features), output_path)


def export_gap_repair_hexes_geojson(debug_hexes: list, output_path: str):
    """
    Export gap repair search hexagons as GeoJSON polygons.

    Each record corresponds to an H3 cell included in an algorithm search pass.

    Args:
        debug_hexes: List of dicts with keys:
            h3_index, algorithm, phase, attempt_id, segment_idx,
            repair_round, search_radius_m, search_ring
        output_path: Output GeoJSON file path
    """
    features = []
    for rec in debug_hexes:
        h3_idx = rec['h3_index']
        boundary = h3.cell_to_boundary(h3_idx)
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [coords]},
            'properties': {
                'h3_index': h3_idx,
                'algorithm': rec.get('algorithm'),
                'phase': rec.get('phase'),
                'attempt_id': rec.get('attempt_id'),
                'segment_idx': rec.get('segment_idx'),
                'repair_round': rec.get('repair_round'),
                'search_radius_m': rec.get('search_radius_m'),
                'search_ring': rec.get('search_ring'),
                'search_scope': rec.get('search_scope'),
                'step_idx': rec.get('step_idx'),
                # Legacy compatibility
                'gap_idx': rec.get('gap_idx', rec.get('segment_idx')),
                'buffer_ring': rec.get('buffer_ring', rec.get('search_ring')),
            },
        })

    geojson = {'type': 'FeatureCollection', 'features': features}
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    logger.info(
        "Exported gap repair hexes: count=%d path=%s",
        len(features),
        output_path,
    )


def export_visibility_edges_geojson(surface: MeshSurface, output_path: str):
    """
    Export visibility graph edges as GeoJSON LineStrings.

    Args:
        surface: Mesh surface with visibility graph
        output_path: Output GeoJSON file path
    """
    features = []

    def _link_type(t1, t2) -> str:
        """Classify a visibility link for color rendering.

        green  — both towers placed by normal DP (or are site anchors)
        yellow — either tower was placed by gap-repair DP
        red    — either tower fell back to peak/endpoint fallback
        """
        algs = {
            (t1.placement_meta or {}).get('algorithm'),
            (t2.placement_meta or {}).get('algorithm'),
        }
        if algs & {'peak_fallback', 'endpoint_fallback'}:
            return 'red'
        if algs & {'dp_repair'}:
            return 'yellow'
        return 'green'

    for t1_id, t2_id, data in surface.visibility_graph.graph.edges(data=True):
        t1 = surface.towers[t1_id]
        t2 = surface.towers[t2_id]
        c1 = surface.cells.get(t1.h3_index)
        c2 = surface.cells.get(t2.h3_index)
        h1 = surface.config.mast_height_m + float(
            getattr(c1, 'antenna_height_offset_m', 0.0) or 0.0
        )
        h2 = surface.config.mast_height_m + float(
            getattr(c2, 'antenna_height_offset_m', 0.0) or 0.0
        )
        clearance_m = data.get('clearance_m')
        is_nlos = (clearance_m is not None and clearance_m < 0)
        los_state = 'nlos' if is_nlos else 'los'

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[t1.lon, t1.lat], [t2.lon, t2.lat]]
            },
            'properties': {
                'source_id': t1_id,
                'target_id': t2_id,
                'source_h3': t1.h3_index,
                'target_h3': t2.h3_index,
                'source_lat': round(t1.lat, 6),
                'source_lon': round(t1.lon, 6),
                'target_lat': round(t2.lat, 6),
                'target_lon': round(t2.lon, 6),
                'source_centroid_lat': round(c1.lat, 6) if c1 is not None else None,
                'source_centroid_lon': round(c1.lon, 6) if c1 is not None else None,
                'target_centroid_lat': round(c2.lat, 6) if c2 is not None else None,
                'target_centroid_lon': round(c2.lon, 6) if c2 is not None else None,
                'source_source': t1.source,
                'target_source': t2.source,
                'distance_m': data.get('distance_m'),
                'clearance_m': clearance_m,
                'path_loss_db': data.get('path_loss_db'),
                'mast_height_m': surface.config.mast_height_m,
                'source_elevation_m': c1.elevation if c1 is not None else None,
                'target_elevation_m': c2.elevation if c2 is not None else None,
                'source_antenna_height_m': h1,
                'target_antenna_height_m': h2,
                'edge_origin': data.get('edge_origin'),
                'visibility_policy': data.get('visibility_policy'),
                'link_budget_db': data.get('link_budget_db'),
                'path_loss_margin_db': data.get('path_loss_margin_db'),
                'max_allowed_fresnel_obstruction_ratio': data.get(
                    'max_allowed_fresnel_obstruction_ratio'
                ),
                'fresnel_obstruction_ratio': data.get('fresnel_obstruction_ratio'),
                'fresnel_obstruction_margin_ratio': data.get(
                    'fresnel_obstruction_margin_ratio'
                ),
                'accepted_by_budget': data.get('accepted_by_budget'),
                'accepted_by_fresnel_policy': data.get('accepted_by_fresnel_policy'),
                'source_algorithm': (t1.placement_meta or {}).get(
                    'algorithm',
                    'site' if t1.source == 'site' else None,
                ),
                'target_algorithm': (t2.placement_meta or {}).get(
                    'algorithm',
                    'site' if t2.source == 'site' else None,
                ),
                'link_type': _link_type(t1, t2),
                'is_nlos': is_nlos,
                'los_state': los_state,
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info(
        "Exported visibility edges: count=%d path=%s",
        len(features),
        output_path,
    )


def generate_report(surface: MeshSurface, output_path: str):
    """
    Generate JSON report with statistics.

    Args:
        surface: Mesh surface
        output_path: Output JSON file path
    """
    # Count towers by source
    VIRTUAL_SOURCES = {'site'}
    towers_by_source = {}
    for tower in surface.towers.values():
        source = tower.source
        towers_by_source[source] = towers_by_source.get(source, 0) + 1

    physical_tower_count = sum(
        1 for t in surface.towers.values()
        if t.source not in VIRTUAL_SOURCES
    )
    virtual_tower_count = len(surface.towers) - physical_tower_count

    # Get clusters
    clusters = surface.get_tower_clusters()
    num_clusters = len(set(clusters.values()))

    # Count cells with towers
    cells_with_towers = sum(1 for cell in surface.cells.values() if cell.has_tower)

    total_covered = sum(1 for c in surface.cells.values() if c.is_covered)
    coverage_pct = round(100.0 * total_covered / len(surface.cells), 1) if surface.cells else 0.0

    report = {
        'total_cells': len(surface.cells),
        'cells_with_towers': cells_with_towers,
        'total_towers': len(surface.towers),
        'physical_towers': physical_tower_count,
        'virtual_towers': virtual_tower_count,
        'towers_by_source': towers_by_source,
        'num_clusters': num_clusters,
        'cluster_info': {
            'total_clusters': num_clusters,
            'cluster_sizes': {
                cluster_id: sum(1 for c in clusters.values() if c == cluster_id)
                for cluster_id in set(clusters.values())
            }
        },
        'coverage': {
            'covered_cells': total_covered,
            'total_cells': len(surface.cells),
            'coverage_pct': coverage_pct,
            'tx_power_dbm': round(surface.config.tx_power_dbm, 2),
            'link_budget_db': round(surface.config.link_budget_db, 2),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("Report generated",
                path=output_path,
                total_cells=report['total_cells'],
                total_towers=report['total_towers'],
                physical_towers=physical_tower_count,
                virtual_towers=virtual_tower_count,
                num_clusters=num_clusters,
                towers_by_source=towers_by_source)
