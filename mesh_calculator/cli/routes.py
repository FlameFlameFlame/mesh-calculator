"""
CLI subcommand: mesh-calculator routes

Runs the route-based tower placement pipeline from a routes.json file
produced by mesh-generator or manually crafted.
"""
import json
import logging

import click

from ..core.config import MeshConfig, RouteSpec
from ..core.grid_provider import GridProvider
from ..logging_config import setup_logging
from ..optimization.route_pipeline import run_route_pipeline

logger = logging.getLogger(__name__)


@click.command('routes')
@click.option(
    '--routes', 'routes_path',
    type=click.Path(exists=True), required=True,
    help='Path to routes.json file',
)
@click.option(
    '--grid-bundle',
    'grid_bundle',
    type=click.Path(exists=True), required=True,
    help='Path to persisted grid bundle JSON',
)
@click.option(
    '--city-boundaries',
    type=click.Path(exists=True), default=None,
    help='Optional path to city boundaries GeoJSON for city link tagging',
)
@click.option(
    '--output', default='output',
    help='Output directory path (default: output/)',
)
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', is_flag=True, help='Suppress info-level logging')
def routes_cmd(
    routes_path: str,
    grid_bundle: str,
    city_boundaries: str,
    output: str,
    verbose: bool,
    quiet: bool,
):
    """
    Place mesh towers along user-chosen road routes.

    Reads a routes.json file containing route GeoJSON features and
    per-route tower limits. Outputs towers.geojson, coverage.geojson,
    visibility_edges.geojson, and report.json to the output directory.

    routes.json format:

    \b
    {
      "parameters": {"h3_resolution": 8, "frequency_hz": 868000000, ...},
      "routes": [
        {
          "route_id": "route_0",
          "site1": {"name": "Yerevan", "lat": 40.18, "lon": 44.51},
          "site2": {"name": "Gyumri",  "lat": 40.79, "lon": 43.84},
          "max_towers": 8,
          "features": [ ...GeoJSON feature dicts... ]
        }
      ]
    }
    """
    setup_logging(verbose=verbose, quiet=quiet)
    logger.info("Mesh Calculator — Route Mode")

    with open(routes_path) as f:
        data = json.load(f)

    # Build MeshConfig from parameters section (all fields optional)
    params = data.get('parameters', {})
    mesh_config = MeshConfig(**{
        k: v for k, v in params.items()
        if k in MeshConfig.__dataclass_fields__
    })

    # Build RouteSpec list
    route_specs = []
    for r in data.get('routes', []):
        route_specs.append(RouteSpec(
            route_id=r['route_id'],
            features=r['features'],
            site1=r.get('site1', {}),
            site2=r.get('site2', {}),
            max_towers_per_route=r.get('max_towers_per_route', mesh_config.max_towers_per_route),
        ))

    if not route_specs:
        raise click.UsageError("routes.json contains no routes")

    logger.info("Loaded %d route(s) from %s", len(route_specs), routes_path)

    # Load city boundaries if provided
    city_boundaries_geojson = None
    if city_boundaries:
        with open(city_boundaries) as f:
            city_boundaries_geojson = json.load(f)
        logger.info("Loaded city boundaries from %s", city_boundaries)

    with GridProvider.from_bundle(grid_bundle) as grid_provider:
        summary = run_route_pipeline(
            routes=route_specs,
            mesh_config=mesh_config,
            grid_provider=grid_provider,
            city_boundaries_geojson=city_boundaries_geojson,
            output_dir=output,
        )

    click.echo(
        f"\nDone: {summary['total_towers']} towers, "
        f"{summary['visibility_edges']} links, "
        f"{summary['total_cells']} coverage cells"
    )
    click.echo(f"Output: {output}/")
