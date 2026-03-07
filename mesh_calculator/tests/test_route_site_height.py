"""
Route-pipeline tests for per-site endpoint height handling.
"""
from __future__ import annotations

import h3

from ..core.config import MeshConfig, RouteSpec
from ..optimization import route_pipeline as rp


class _FakeElevationProvider:
    def get_elevation(self, _lat: float, _lon: float) -> float:
        return 100.0

    def cache_stats(self) -> dict:
        return {}

    def resolve_effective_resolution(self, _routes, base_resolution: int, _config):
        return base_resolution, False, None, None

    def adaptive_resolution_summary(self, base_resolution: int, _config):
        return {
            "h3_resolution_mode": "adaptive_mixed",
            "base_h3_resolution": int(base_resolution),
            "effective_h3_resolution_min": int(base_resolution),
            "effective_h3_resolution_max": int(base_resolution),
            "cells_by_resolution": {int(base_resolution): 0},
        }

    def get_or_create_cell(
        self,
        h3_index: str,
        _config,
        *,
        has_road: bool = False,
        is_in_boundary: bool = True,
    ):
        from ..core.grid import H3Cell
        lat, lon = h3.cell_to_latlng(h3_index)
        return H3Cell(
            h3_index=h3_index,
            lat=lat,
            lon=lon,
            elevation=100.0,
            has_road=has_road,
            is_in_boundary=is_in_boundary,
        )

    def materialize_cells(
        self,
        h3_indices,
        config,
        *,
        road_cells=None,
        is_in_boundary=True,
        include_stats=False,
    ):
        road_set = road_cells or set()
        out = {}
        for h3_idx in h3_indices:
            out[h3_idx] = self.get_or_create_cell(
                h3_idx,
                config,
                has_road=(h3_idx in road_set),
                is_in_boundary=is_in_boundary,
            )
        if include_stats:
            return out, {
                "requested": len(out),
                "cache_hits": 0,
                "from_static": len(out),
                "from_dem": 0,
                "materialized": len(out),
            }
        return out

    def radius_m_to_ring(self, _radius_m: float, _resolution: int, minimum_one: bool = False) -> int:
        return 1 if minimum_one else 0

    def expand_disk(self, h3_index: str, rings: int) -> set[str]:
        return set(h3.grid_disk(h3_index, rings))

    def corridor_from_features(self, _features, resolution: int, site1=None, site2=None, config=None):
        if not site1 or not site2:
            return []
        return [
            h3.latlng_to_cell(site1["lat"], site1["lon"], resolution),
            h3.latlng_to_cell(site2["lat"], site2["lon"], resolution),
        ]

    def locate_adaptive_cell(self, lat, lon, resolution, _config, prefer_road=False):
        return h3.latlng_to_cell(lat, lon, resolution)

    def get_adaptive_full_cells(self, _base_resolution, _config):
        return set()

    def adaptive_union_within_radius(self, _centers, _radius_m, _base_resolution, _config, *, candidate_cells=None):
        return set()

    def adaptive_cells_within_radius(self, _center_h3, _radius_m, _base_resolution, _config, *, candidate_cells=None):
        return set()

    def close(self) -> None:
        return None


def _prepare_pipeline_monkeypatch(monkeypatch, captured: dict) -> None:
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

    provider = _FakeElevationProvider()

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
        grid_provider=provider,
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

    provider = _FakeElevationProvider()
    monkeypatch.setattr(
        provider,
        "corridor_from_features",
        lambda *_a, **_k: [entry_h3, end_h3],
    )
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
        grid_provider=provider,
        city_boundaries_geojson=city_poly,
        output_dir=str(tmp_path),
    )

    surface = captured["surface"]
    assert surface.cells[entry_h3].antenna_height_offset_m == 9.0
