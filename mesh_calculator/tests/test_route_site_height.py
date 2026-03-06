"""
Route-pipeline tests for per-site endpoint height handling.
"""
from __future__ import annotations

import h3

from ..core.config import MeshConfig, RouteSpec
from ..optimization import route_pipeline as rp


class _FakeElevationProvider:
    def __init__(self, _path: str):
        pass

    def get_elevation(self, _lat: float, _lon: float) -> float:
        return 100.0

    def cache_stats(self) -> dict:
        return {}

    def close(self) -> None:
        return None


def _prepare_pipeline_monkeypatch(monkeypatch, captured: dict) -> None:
    monkeypatch.setattr(rp, "ElevationProvider", _FakeElevationProvider)
    monkeypatch.setattr(rp, "_expand_cells_with_buffer", lambda *_a, **_k: {})
    monkeypatch.setattr(rp, "place_nodes_along_corridor", lambda *_a, **_k: [])
    monkeypatch.setattr(rp, "install_nodes", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "wire_corridor_edges", lambda *_a, **_k: None)
    monkeypatch.setattr(rp.MeshSurface, "update_visibility_edges", lambda self, _cache: None)
    monkeypatch.setattr(rp.MeshSurface, "compute_cell_coverage", lambda self, _cache: None)

    def _capture_surface(surface, _path):
        captured["surface"] = surface

    monkeypatch.setattr(rp, "export_towers_geojson", _capture_surface)
    monkeypatch.setattr(rp, "export_coverage_geojson", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "export_visibility_edges_geojson", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "export_grid_cells_geojson", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "export_gap_repair_hexes_geojson", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "generate_report", lambda *_a, **_k: None)


def test_non_city_anchor_site_height_offset_uses_max_on_reuse(monkeypatch, tmp_path):
    captured = {}
    _prepare_pipeline_monkeypatch(monkeypatch, captured)

    cfg = MeshConfig(h3_resolution=8)

    site_a = {"name": "A", "lat": 40.2000, "lon": 44.5000}
    site_b = {"name": "B", "lat": 40.2200, "lon": 44.5200}
    site_c = {"name": "C", "lat": 40.2400, "lon": 44.5400}

    def _corridor(_features, resolution, site1, site2):
        return [
            h3.latlng_to_cell(site1["lat"], site1["lon"], resolution),
            h3.latlng_to_cell(site2["lat"], site2["lon"], resolution),
        ]

    monkeypatch.setattr(rp, "road_geojson_to_h3_corridor", _corridor)

    routes = [
        RouteSpec(
            route_id="r1",
            features=[{}],
            site1={**site_a, "site_height_m": 4.0},
            site2={**site_b, "site_height_m": 1.0},
            max_towers_per_route=5,
        ),
        RouteSpec(
            route_id="r2",
            features=[{}],
            site1={**site_a, "site_height_m": 12.0},
            site2={**site_c, "site_height_m": 0.0},
            max_towers_per_route=5,
        ),
    ]

    rp.run_route_pipeline(
        routes=routes,
        mesh_config=cfg,
        elevation_path=str(tmp_path / "fake.tif"),
        city_boundaries_geojson=None,
        output_dir=str(tmp_path),
    )

    surface = captured["surface"]
    site_a_h3 = h3.latlng_to_cell(site_a["lat"], site_a["lon"], cfg.h3_resolution)
    assert surface.cells[site_a_h3].antenna_height_offset_m == 12.0


def test_city_anchor_site_height_applied_to_entry_cell(monkeypatch, tmp_path):
    captured = {}
    _prepare_pipeline_monkeypatch(monkeypatch, captured)

    cfg = MeshConfig(h3_resolution=8)

    city_site = {"name": "CityA", "lat": 40.1000, "lon": 44.1000, "site_height_m": 9.0}
    remote_site = {"name": "Remote", "lat": 40.3000, "lon": 44.3000, "site_height_m": 0.0}

    entry_h3 = h3.latlng_to_cell(40.1100, 44.1100, cfg.h3_resolution)
    end_h3 = h3.latlng_to_cell(remote_site["lat"], remote_site["lon"], cfg.h3_resolution)

    monkeypatch.setattr(rp, "road_geojson_to_h3_corridor", lambda *_a, **_k: [entry_h3, end_h3])
    monkeypatch.setattr(rp, "_trim_unfit_ends", lambda corridor, _cells: (corridor, entry_h3, end_h3))

    city_poly = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [44.05, 40.05],
                    [44.15, 40.05],
                    [44.15, 40.15],
                    [44.05, 40.15],
                    [44.05, 40.05],
                ]],
            },
            "properties": {"name": "CityA"},
        }],
    }

    routes = [
        RouteSpec(
            route_id="r_city",
            features=[{}],
            site1=city_site,
            site2=remote_site,
            max_towers_per_route=5,
        )
    ]

    rp.run_route_pipeline(
        routes=routes,
        mesh_config=cfg,
        elevation_path=str(tmp_path / "fake.tif"),
        city_boundaries_geojson=city_poly,
        output_dir=str(tmp_path),
    )

    surface = captured["surface"]
    assert surface.cells[entry_h3].antenna_height_offset_m == 9.0
