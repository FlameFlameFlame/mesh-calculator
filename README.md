# Mesh Network Tower Placement Optimizer

Python implementation of a mesh network tower placement system that connects user-specified sites (cities) along roads using hierarchical priority levels.

## Overview

This system optimizes wireless mesh network tower placement to connect target locations based on:
- **Hierarchical Priorities**: Priority 1 sites fully interconnected, lower priorities connect to higher
- **Road-Based Placement**: Nodes placed in H3 cells containing roads
- **Line-of-Sight Physics**: Fresnel clearance and path loss calculations
- **Multithreading**: Parallel LOS computations for performance

## Features

- ✅ H3 hexagonal grid-based spatial indexing
- ✅ Fresnel zone clearance with earth curvature
- ✅ FSPL + knife-edge diffraction path loss
- ✅ Dijkstra pathfinding along road corridors
- ✅ Hierarchical site connectivity (priority-based)
- ✅ Per-road node limits
- ✅ Multithreaded LOS calculations
- ✅ GeoJSON input/output

## Installation

### Prerequisites

- Python 3.10+
- GDAL (for rasterio/fiona)

### Install Dependencies

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install gdal-bin libgdal-dev libspatialindex-dev

# Install Python package
uv sync --group dev
```

## Usage

### 1. Prepare Input Data

You need:
- **Boundary**: GeoJSON polygon defining the region
- **Elevation**: GeoTIFF elevation data (e.g., GEBCO)
- **Roads**: GeoJSON road network
- **Sites**: GeoJSON points with priority levels

**Example sites.geojson**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [44.5152, 40.1872]},
      "properties": {"name": "Yerevan", "priority": 1}
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [43.8403, 40.7942]},
      "properties": {"name": "Gyumri", "priority": 1}
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [44.4883, 40.8128]},
      "properties": {"name": "Vanadzor", "priority": 1}
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [44.7628, 40.5017]},
      "properties": {"name": "Hrazdan", "priority": 2}
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [44.8625, 40.7416]},
      "properties": {"name": "Dilijan", "priority": 2}
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [44.9467, 40.5334]},
      "properties": {"name": "Sevan", "priority": 2}
    }
  ]
}
```

### 2. Create Configuration File

```yaml
# config.yaml
parameters:
  h3_resolution: 8
  max_visibility_m: 70000
  max_nodes_per_road: 10
  mast_height_m: 28
  frequency_hz: 868000000

inputs:
  boundary: data/boundary.geojson
  elevation: data/elevation.tif
  roads: data/roads.geojson
  target_sites: data/sites.geojson

outputs:
  towers: output/towers.geojson
  coverage: output/coverage.geojson
  report: output/report.json
```

### 3. Run Optimizer

```bash
mesh-calculator --config config.yaml --output output/
```

### 4. View Results

Output files:
- `output/towers.geojson`: Tower placements
- `output/coverage.geojson`: Grid cells with metrics
- `output/report.json`: Statistics

## Algorithm

### Hierarchical Connectivity

1. **Priority 1**: Connect all Priority 1 sites in full mesh (all-to-all)
2. **Priority 2+**: Each lower-priority site connects to nearest higher-priority site
3. **Corridor Finding**: Use Dijkstra along roads
4. **Node Placement**: Place nodes along corridor ensuring LOS
5. **Node Limits**: Optimize to respect per-road limits

### Line-of-Sight Calculation

- **Fresnel Clearance**: Sample elevation along path, account for earth curvature
- **Path Loss**: FSPL + knife-edge diffraction
- **Caching**: Thread-safe LOS cache to avoid redundant calculations

### Multithreading

- Parallel LOS computations using `ThreadPoolExecutor`
- Shared read-only elevation data
- Thread-safe cache with locking

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h3_resolution` | 8 | H3 grid resolution (~0.7 km²/cell) |
| `max_visibility_m` | 70000 | Maximum LOS distance (70 km) |
| `tower_separation_m` | 5000 | Minimum tower spacing (5 km) |
| `mast_height_m` | 28 | Tower mast height (meters) |
| `frequency_hz` | 868e6 | Radio frequency (868 MHz) |
| `max_nodes_per_road` | 10 | Node limit per road segment |
| `hop_limit` | 7 | Maximum hops in cluster |

## Architecture

```
mesh_calculator/
├── core/           # Grid, elevation, geometry
├── physics/        # LOS, Fresnel, path loss
├── network/        # Graphs, routing, clustering
├── optimization/   # Hierarchical connectivity
├── data/           # I/O, caching, sites
├── parallel/       # Multithreading
├── cli/            # Command-line interface
└── utils/          # Utilities
```

## Example: Armenia Test Case

```bash
# Prepare data
mkdir -p data output

# Download boundary, elevation, roads from OSM/GEBCO
# Create sites.geojson with Yerevan (P1), Gyumri (P1), Vanadzor (P1),
#   Hrazdan (P2), Dilijan (P2), Sevan (P2)

# Run
mesh-calculator --config armenia_config.yaml --output output/

# Results
#   - Yerevan ↔ Gyumri ↔ Vanadzor: Full mesh (P1)
#   - Hrazdan, Dilijan, Sevan → nearest P1 site
```

## References

Based on the original PostgreSQL/PostGIS implementation:
- `h3-mesh-placement/functions/h3_visibility_clearance.sql`
- `h3-mesh-placement/functions/h3_path_loss.sql`
- `h3-mesh-placement/functions/mesh_tower_clusters.sql`

## License

MIT License

## Contributing

Pull requests welcome! Areas for contribution:
- Additional optimization phases (cluster-slim, wiggle)
- GPU acceleration for LOS
- Web-based visualization
- Load balancing and redundancy
