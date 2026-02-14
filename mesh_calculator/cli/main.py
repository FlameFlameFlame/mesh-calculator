"""
Command-line interface for mesh calculator.
"""
import click
import os
from pathlib import Path

import structlog

from ..logging_config import setup_logging
from ..utils.perf import PerfTimer
from ..data.loaders import load_config
from ..data.sites import load_sites, snap_sites_to_roads
from ..data.cache import LOSCache
from ..data.exporters import (
    export_towers_geojson, export_coverage_geojson,
    export_visibility_edges_geojson, generate_report,
)
from ..core.elevation import ElevationProvider
from ..core.grid import load_boundary, load_roads, generate_road_grid
from ..network.graph import MeshSurface
from ..network.routing import build_routing_graph
from ..optimization.hierarchical import connect_sites_by_priority

logger = structlog.get_logger(__name__)


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to YAML configuration file')
@click.option('--output', type=click.Path(), default='output',
              help='Output directory path')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', is_flag=True, help='Suppress info-level logging')
def main(config: str, output: str, verbose: bool, quiet: bool):
    """
    Mesh Network Tower Placement Optimizer

    Connects user-specified sites (cities) via mesh network nodes placed along roads
    with hierarchical priority levels.
    """
    setup_logging(verbose=verbose, quiet=quiet)

    logger.info("Mesh Network Tower Placement Optimizer")

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Load configuration
    logger.info("[1/9] Loading configuration")
    with PerfTimer("load_configuration"):
        cfg = load_config(config)
    logger.info("Configuration loaded",
                h3_resolution=cfg.parameters.h3_resolution,
                max_visibility_km=cfg.parameters.max_visibility_m / 1000,
                max_nodes_per_road=cfg.parameters.max_nodes_per_road)

    # Load input data
    logger.info("[2/9] Loading input data")
    with PerfTimer("load_input_data"):
        logger.info("Loading boundary")
        boundary = load_boundary(cfg.inputs.boundary)
        logger.info("Boundary loaded", area_sq_deg=round(boundary.area, 4))

        logger.info("Loading roads")
        roads_gdf = load_roads(cfg.inputs.roads)
        logger.info("Roads loaded", features=len(roads_gdf))

        logger.info("Loading target sites")
        sites = load_sites(cfg.inputs.target_sites, cfg.parameters.h3_resolution)
        logger.info("Sites loaded", count=len(sites))
        for site in sites:
            logger.debug("Site found", name=site.name, priority=site.priority)

    # Initialize elevation provider
    logger.info("[3/9] Loading elevation data")
    with PerfTimer("load_elevation"):
        elevation_provider = ElevationProvider(cfg.inputs.elevation)
    logger.info("Elevation provider initialized")

    # Generate H3 grid (only cells with roads)
    logger.info("[4/9] Generating H3 grid")
    with PerfTimer("generate_h3_grid"):
        cells = generate_road_grid(boundary, roads_gdf, elevation_provider, cfg.parameters)

    # Snap sites to nearest road cell
    logger.info("[4.5/9] Snapping sites to road cells")
    snap_sites_to_roads(sites, cells)

    # Create mesh surface
    logger.info("[5/9] Creating mesh surface")
    with PerfTimer("create_mesh_surface"):
        surface = MeshSurface(cells, cfg.parameters,
                              elevation_provider=elevation_provider)
    logger.info("Mesh surface created", cells=len(surface.cells))

    # Initialize LOS cache
    logger.info("[6/9] Initializing LOS cache")
    los_cache = LOSCache()
    logger.info("LOS cache initialized")

    # Build routing graph
    logger.info("[7/9] Building routing graph")
    with PerfTimer("build_routing_graph"):
        routing_graph = build_routing_graph(cells, roads_gdf, cfg.parameters)
    logger.info("Routing graph built",
                nodes=routing_graph.number_of_nodes(),
                edges=routing_graph.number_of_edges())

    # Connect sites by priority hierarchy
    logger.info("[8/9] Connecting sites by priority")
    with PerfTimer("connect_sites"):
        connect_sites_by_priority(sites, surface, routing_graph, los_cache)

    # Export results
    logger.info("[9/9] Exporting results")
    with PerfTimer("export_results"):
        towers_path = os.path.join(output, 'towers.geojson')
        export_towers_geojson(surface, towers_path)

        coverage_path = os.path.join(output, 'coverage.geojson')
        export_coverage_geojson(surface, coverage_path)

        report_path = os.path.join(output, 'report.json')
        generate_report(surface, report_path)

        edges_path = os.path.join(output, 'visibility_edges.geojson')
        export_visibility_edges_geojson(surface, edges_path)

    # Log cache stats
    cache_stats = los_cache.stats()
    logger.info("Cache statistics",
                los_entries=cache_stats['size'],
                los_hit_rate=f"{cache_stats['hit_rate']:.1%}")

    elev_stats = elevation_provider.cache_stats()
    logger.info("Elevation cache", entries=elev_stats['cache_size'])

    logger.info("Optimization complete",
                towers_placed=len(surface.towers),
                output_dir=output)


if __name__ == '__main__':
    main()
