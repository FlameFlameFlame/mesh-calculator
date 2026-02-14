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
                'lat': tower.lat,
                'lon': tower.lon
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info("Exported towers", count=len(features), path=output_path)


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
                'path_loss': cell.path_loss
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info("Exported coverage cells", count=len(features), path=output_path)


def export_visibility_edges_geojson(surface: MeshSurface, output_path: str):
    """
    Export visibility graph edges as GeoJSON LineStrings.

    Args:
        surface: Mesh surface with visibility graph
        output_path: Output GeoJSON file path
    """
    features = []

    for t1_id, t2_id, data in surface.visibility_graph.graph.edges(data=True):
        t1 = surface.towers[t1_id]
        t2 = surface.towers[t2_id]

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[t1.lon, t1.lat], [t2.lon, t2.lat]]
            },
            'properties': {
                'source_id': t1_id,
                'target_id': t2_id,
                'distance_m': data.get('distance_m'),
                'clearance_m': data.get('clearance_m'),
                'path_loss_db': data.get('path_loss_db'),
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    logger.info("Exported visibility edges", count=len(features), path=output_path)


def generate_report(surface: MeshSurface, output_path: str):
    """
    Generate JSON report with statistics.

    Args:
        surface: Mesh surface
        output_path: Output JSON file path
    """
    # Count towers by source
    towers_by_source = {}
    for tower in surface.towers.values():
        source = tower.source
        towers_by_source[source] = towers_by_source.get(source, 0) + 1

    # Get clusters
    clusters = surface.get_tower_clusters()
    num_clusters = len(set(clusters.values()))

    # Count cells with towers
    cells_with_towers = sum(1 for cell in surface.cells.values() if cell.has_tower)

    report = {
        'total_cells': len(surface.cells),
        'cells_with_towers': cells_with_towers,
        'total_towers': len(surface.towers),
        'towers_by_source': towers_by_source,
        'num_clusters': num_clusters,
        'cluster_info': {
            'total_clusters': num_clusters,
            'cluster_sizes': {
                cluster_id: sum(1 for c in clusters.values() if c == cluster_id)
                for cluster_id in set(clusters.values())
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("Report generated",
                path=output_path,
                total_cells=report['total_cells'],
                total_towers=report['total_towers'],
                num_clusters=num_clusters,
                towers_by_source=towers_by_source)
