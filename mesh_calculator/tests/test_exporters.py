"""
Unit tests for data exporters.
"""
import json
import os
import tempfile
import unittest

from ..network.graph import Tower, VisibilityGraph, MeshSurface
from ..core.grid import H3Cell
from ..core.config import MeshConfig
from ..data.exporters import (
    export_gap_repair_hexes_geojson,
    export_grid_cells_geojson,
    export_visibility_edges_geojson,
)


def _make_surface_with_edges():
    """Create a MeshSurface with 3 towers and 2 visibility edges."""
    config = MeshConfig()
    # Use fake h3 indices (they won't be resolved, we just need dict keys)
    cells = {
        "8828c00001fffff": H3Cell("8828c00001fffff", 40.0, 44.0, 500.0, has_road=True),
        "8828c00003fffff": H3Cell("8828c00003fffff", 40.1, 44.1, 600.0, has_road=True),
        "8828c00005fffff": H3Cell("8828c00005fffff", 40.2, 44.2, 550.0, has_road=True),
    }
    surface = MeshSurface(cells, config)

    # Manually place towers (bypass place_tower to avoid h3 lookups)
    t1 = Tower(1, "8828c00001fffff", 40.0, 44.0, "seed")
    t2 = Tower(2, "8828c00003fffff", 40.1, 44.1, "route")
    t3 = Tower(3, "8828c00005fffff", 40.2, 44.2, "corridor")
    surface.towers = {1: t1, 2: t2, 3: t3}

    surface.visibility_graph = VisibilityGraph()
    surface.visibility_graph.add_tower(t1)
    surface.visibility_graph.add_tower(t2)
    surface.visibility_graph.add_tower(t3)
    surface.visibility_graph.add_visibility_edge(
        1, 2,
        distance_m=12000.0,
        clearance_m=15.5,
        path_loss_db=120.3,
        edge_origin="global_visibility",
        visibility_policy="budget_and_fresnel_40pct",
        link_budget_db=130.0,
        path_loss_margin_db=9.7,
        max_allowed_fresnel_obstruction_ratio=0.4,
        fresnel_obstruction_ratio=0.12,
        fresnel_obstruction_margin_ratio=0.28,
        accepted_by_budget=True,
        accepted_by_fresnel_policy=True,
    )
    surface.visibility_graph.add_visibility_edge(2, 3, distance_m=8000.0, clearance_m=-22.0, path_loss_db=115.1)

    return surface


class TestExportVisibilityEdges(unittest.TestCase):
    """Test visibility edges GeoJSON export."""

    def test_feature_count(self):
        """Export produces one feature per edge."""
        surface = _make_surface_with_edges()
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_visibility_edges_geojson(surface, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["type"], "FeatureCollection")
            self.assertEqual(len(data["features"]), 2)
        finally:
            os.unlink(path)

    def test_linestring_geometry(self):
        """Each feature is a LineString with two coordinate pairs."""
        surface = _make_surface_with_edges()
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_visibility_edges_geojson(surface, path)
            with open(path) as f:
                data = json.load(f)
            for feat in data["features"]:
                self.assertEqual(feat["type"], "Feature")
                self.assertEqual(feat["geometry"]["type"], "LineString")
                self.assertEqual(len(feat["geometry"]["coordinates"]), 2)
        finally:
            os.unlink(path)

    def test_coordinate_order_lon_lat(self):
        """Coordinates are in GeoJSON [lon, lat] order."""
        surface = _make_surface_with_edges()
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_visibility_edges_geojson(surface, path)
            with open(path) as f:
                data = json.load(f)
            # Find edge 1->2
            feat = data["features"][0]
            coords = feat["geometry"]["coordinates"]
            # Tower 1: lat=40.0, lon=44.0 → [44.0, 40.0]
            self.assertAlmostEqual(coords[0][0], 44.0)
            self.assertAlmostEqual(coords[0][1], 40.0)
        finally:
            os.unlink(path)

    def test_properties(self):
        """Edge properties include IDs, RF metrics, and endpoint antenna heights."""
        surface = _make_surface_with_edges()
        surface.cells["8828c00001fffff"].antenna_height_offset_m = 3.0
        surface.cells["8828c00003fffff"].antenna_height_offset_m = 1.5
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_visibility_edges_geojson(surface, path)
            with open(path) as f:
                data = json.load(f)
            props = data["features"][0]["properties"]
            self.assertIn("source_id", props)
            self.assertIn("target_id", props)
            self.assertIn("distance_m", props)
            self.assertIn("clearance_m", props)
            self.assertIn("path_loss_db", props)
            self.assertIn("mast_height_m", props)
            self.assertIn("source_elevation_m", props)
            self.assertIn("target_elevation_m", props)
            self.assertIn("source_antenna_height_m", props)
            self.assertIn("target_antenna_height_m", props)
            self.assertIn("edge_origin", props)
            self.assertIn("visibility_policy", props)
            self.assertIn("link_budget_db", props)
            self.assertIn("path_loss_margin_db", props)
            self.assertIn("max_allowed_fresnel_obstruction_ratio", props)
            self.assertIn("fresnel_obstruction_ratio", props)
            self.assertIn("fresnel_obstruction_margin_ratio", props)
            self.assertIn("accepted_by_budget", props)
            self.assertIn("accepted_by_fresnel_policy", props)
            self.assertIn("source_algorithm", props)
            self.assertIn("target_algorithm", props)
            self.assertAlmostEqual(props["distance_m"], 12000.0)
            self.assertAlmostEqual(props["clearance_m"], 15.5)
            self.assertAlmostEqual(props["path_loss_db"], 120.3)
            self.assertAlmostEqual(props["mast_height_m"], surface.config.mast_height_m)
            self.assertAlmostEqual(
                props["source_antenna_height_m"],
                surface.config.mast_height_m + 3.0,
            )
            self.assertAlmostEqual(
                props["target_antenna_height_m"],
                surface.config.mast_height_m + 1.5,
            )
            self.assertAlmostEqual(props["source_elevation_m"], 500.0)
            self.assertAlmostEqual(props["target_elevation_m"], 600.0)
            self.assertEqual(props["edge_origin"], "global_visibility")
            self.assertEqual(props["visibility_policy"], "budget_and_fresnel_40pct")
            self.assertAlmostEqual(props["link_budget_db"], 130.0)
            self.assertAlmostEqual(props["path_loss_margin_db"], 9.7)
            self.assertAlmostEqual(props["max_allowed_fresnel_obstruction_ratio"], 0.4)
            self.assertAlmostEqual(props["fresnel_obstruction_ratio"], 0.12)
            self.assertAlmostEqual(props["fresnel_obstruction_margin_ratio"], 0.28)
            self.assertTrue(props["accepted_by_budget"])
            self.assertTrue(props["accepted_by_fresnel_policy"])
        finally:
            os.unlink(path)

    def test_los_state_properties(self):
        """Edges export geometric LOS classification from clearance sign."""
        surface = _make_surface_with_edges()
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_visibility_edges_geojson(surface, path)
            with open(path) as f:
                data = json.load(f)

            by_pair = {
                (feat["properties"]["source_id"], feat["properties"]["target_id"]): feat["properties"]
                for feat in data["features"]
            }

            pos = by_pair[(1, 2)]
            self.assertFalse(pos["is_nlos"])
            self.assertEqual(pos["los_state"], "los")

            neg = by_pair[(2, 3)]
            self.assertTrue(neg["is_nlos"])
            self.assertEqual(neg["los_state"], "nlos")
        finally:
            os.unlink(path)

    def test_empty_graph(self):
        """Export works with no edges."""
        config = MeshConfig()
        cells = {"8828c00001fffff": H3Cell("8828c00001fffff", 40.0, 44.0, 500.0)}
        surface = MeshSurface(cells, config)
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_visibility_edges_geojson(surface, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data["features"]), 0)
        finally:
            os.unlink(path)


class TestExportGapRepairHexes(unittest.TestCase):
    def test_richer_search_debug_properties_export(self):
        debug_hexes = [{
            "h3_index": "8828c00001fffff",
            "algorithm": "dp",
            "phase": "gap_repair",
            "attempt_id": 1,
            "segment_idx": 3,
            "repair_round": 2,
            "search_radius_m": 600.0,
            "search_ring": 2,
            "search_scope": "gap_repair_subcorridor",
            "step_idx": None,
        }]
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_gap_repair_hexes_geojson(debug_hexes, path)
            with open(path) as f:
                data = json.load(f)
            props = data["features"][0]["properties"]
            self.assertEqual(props["algorithm"], "dp")
            self.assertEqual(props["phase"], "gap_repair")
            self.assertEqual(props["attempt_id"], 1)
            self.assertEqual(props["segment_idx"], 3)
            self.assertEqual(props["search_radius_m"], 600.0)
            self.assertEqual(props["search_ring"], 2)
            self.assertEqual(props["search_scope"], "gap_repair_subcorridor")
            self.assertIsNone(props["step_idx"])
        finally:
            os.unlink(path)


class TestExportGridCells(unittest.TestCase):
    def test_grid_resolution_metadata_export(self):
        cells = {
            "8828c00001fffff": H3Cell("8828c00001fffff", 40.0, 44.0, 500.0, has_road=True),
        }
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            path = f.name
        try:
            export_grid_cells_geojson(cells, path, effective_h3_resolution=9)
            with open(path) as f:
                data = json.load(f)
            props = data["features"][0]["properties"]
            self.assertEqual(props["h3_resolution"], 8)
            self.assertEqual(props["effective_h3_resolution"], 9)
        finally:
            os.unlink(path)
