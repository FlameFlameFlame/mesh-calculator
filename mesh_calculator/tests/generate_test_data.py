"""
Generate synthetic test geodata for testing.

Creates:
- Small boundary polygon
- Simple road network
- Flat elevation GeoTIFF
- Test sites with priorities
"""
import json
import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd


def generate_test_data(output_dir='test_data'):
    """
    Generate all test data files.

    Args:
        output_dir: Directory to write test data
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating test data in {output_dir}/...")

    # Define a small test region (10x10 km area)
    # Center: roughly 40.0°N, -74.0°W (arbitrary location)
    center_lat, center_lon = 40.0, -74.0

    # Create boundary (small square ~10x10 km)
    # Approximately 0.1 degrees ≈ 11 km at this latitude
    delta = 0.05
    boundary_coords = [
        [center_lon - delta, center_lat - delta],
        [center_lon + delta, center_lat - delta],
        [center_lon + delta, center_lat + delta],
        [center_lon - delta, center_lat + delta],
        [center_lon - delta, center_lat - delta]
    ]

    boundary_geojson = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [boundary_coords]
            },
            'properties': {'name': 'Test Region'}
        }]
    }

    boundary_path = os.path.join(output_dir, 'boundary.geojson')
    with open(boundary_path, 'w') as f:
        json.dump(boundary_geojson, f, indent=2)
    print(f"  ✓ Created boundary: {boundary_path}")

    # Create simple road network (3 roads connecting sites)
    roads_features = []

    # Road 1: Horizontal line across top
    road1 = LineString([
        [center_lon - delta, center_lat + delta * 0.5],
        [center_lon + delta, center_lat + delta * 0.5]
    ])
    roads_features.append({
        'type': 'Feature',
        'geometry': road1.__geo_interface__,
        'properties': {'name': 'Road 1', 'type': 'primary'}
    })

    # Road 2: Horizontal line across bottom
    road2 = LineString([
        [center_lon - delta, center_lat - delta * 0.5],
        [center_lon + delta, center_lat - delta * 0.5]
    ])
    roads_features.append({
        'type': 'Feature',
        'geometry': road2.__geo_interface__,
        'properties': {'name': 'Road 2', 'type': 'primary'}
    })

    # Road 3: Vertical line connecting them
    road3 = LineString([
        [center_lon, center_lat + delta * 0.5],
        [center_lon, center_lat - delta * 0.5]
    ])
    roads_features.append({
        'type': 'Feature',
        'geometry': road3.__geo_interface__,
        'properties': {'name': 'Road 3', 'type': 'secondary'}
    })

    roads_geojson = {
        'type': 'FeatureCollection',
        'features': roads_features
    }

    roads_path = os.path.join(output_dir, 'roads.geojson')
    with open(roads_path, 'w') as f:
        json.dump(roads_geojson, f, indent=2)
    print(f"  ✓ Created roads: {roads_path}")

    # Create test sites with priorities
    sites_features = []

    # Site A: Top left (Priority 1)
    sites_features.append({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [center_lon - delta * 0.7, center_lat + delta * 0.5]
        },
        'properties': {'name': 'Site A', 'priority': 1}
    })

    # Site B: Top right (Priority 1)
    sites_features.append({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [center_lon + delta * 0.7, center_lat + delta * 0.5]
        },
        'properties': {'name': 'Site B', 'priority': 1}
    })

    # Site C: Center (Priority 2)
    sites_features.append({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [center_lon, center_lat]
        },
        'properties': {'name': 'Site C', 'priority': 2}
    })

    sites_geojson = {
        'type': 'FeatureCollection',
        'features': sites_features
    }

    sites_path = os.path.join(output_dir, 'sites.geojson')
    with open(sites_path, 'w') as f:
        json.dump(sites_geojson, f, indent=2)
    print(f"  ✓ Created sites: {sites_path}")

    # Create elevation GeoTIFF (flat terrain with slight variations)
    elev_path = os.path.join(output_dir, 'elevation.tif')

    # Create 100x100 grid
    width, height = 100, 100

    # Bounds
    west = center_lon - delta
    east = center_lon + delta
    south = center_lat - delta
    north = center_lat + delta

    # Generate elevation data (flat with small random variations)
    # Base elevation: 100m, variations: ±10m
    elevation_data = np.random.uniform(90, 110, (height, width)).astype(np.float32)

    # Write GeoTIFF
    transform = from_bounds(west, south, east, north, width, height)

    with rasterio.open(
        elev_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=elevation_data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(elevation_data, 1)

    print(f"  ✓ Created elevation: {elev_path}")

    # Create test configuration
    config = {
        'parameters': {
            'h3_resolution': 8,
            'mast_height_m': 28,
            'frequency_hz': 868000000,
            'max_towers_per_route': 5
        },
        'inputs': {
            'boundary': boundary_path,
            'elevation': elev_path,
            'roads': roads_path,
            'target_sites': sites_path
        },
        'outputs': {
            'towers': os.path.join(output_dir, 'output/towers.geojson'),
            'coverage': os.path.join(output_dir, 'output/coverage.geojson'),
            'report': os.path.join(output_dir, 'output/report.json')
        }
    }

    config_path = os.path.join(output_dir, 'test_config.yaml')
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  ✓ Created config: {config_path}")

    print(f"\nTest data generation complete!")
    print(f"  Boundary: {boundary_path}")
    print(f"  Roads: {roads_path}")
    print(f"  Sites: {sites_path}")
    print(f"  Elevation: {elev_path}")
    print(f"  Config: {config_path}")

    return config_path


if __name__ == '__main__':
    generate_test_data()
