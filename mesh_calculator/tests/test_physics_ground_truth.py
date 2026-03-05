"""
Deterministic tower physics ground-truth tests.

This suite builds a synthetic terrain where one direction from a tower is
blocked by a high ridge while two other directions remain clear.
"""
import unittest

import h3

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..network.tower_coverage import CoverageSource, compute_h3_tower_coverage
from ..physics.los import compute_los


class SyntheticTerrain:
    """Simple deterministic terrain model for visibility assertions."""

    def get_elevation(self, lat: float, lon: float) -> float:
        # Base terrain everywhere.
        if 44.007 <= lon <= 44.013 and 39.995 <= lat <= 40.005:
            # East-side ridge that should block the east-directed link.
            return 1800.0
        return 100.0


class TestTowerPhysicsGroundTruth(unittest.TestCase):
    """Ground-truth visibility and coverage checks from one tower."""

    def setUp(self):
        self.resolution = 9
        self.tower_lat = 40.0
        self.tower_lon = 44.0
        self.tower_h3 = h3.latlng_to_cell(
            self.tower_lat, self.tower_lon, self.resolution
        )

        self.points = {
            "west_clear": (40.0, 43.98),
            "north_clear": (40.03, 44.0),
            "east_blocked": (40.0, 44.02),
        }
        self.point_h3 = {
            name: h3.latlng_to_cell(lat, lon, self.resolution)
            for name, (lat, lon) in self.points.items()
        }

        self.cells = {
            self.tower_h3: H3Cell(
                h3_index=self.tower_h3,
                lat=self.tower_lat,
                lon=self.tower_lon,
                elevation=100.0,
                has_road=True,
                is_in_boundary=True,
            )
        }
        for name, (lat, lon) in self.points.items():
            h3_idx = self.point_h3[name]
            self.cells[h3_idx] = H3Cell(
                h3_index=h3_idx,
                lat=lat,
                lon=lon,
                elevation=100.0,
                has_road=True,
                is_in_boundary=True,
            )

        common = {
            "h3_resolution": self.resolution,
            "frequency_hz": 868e6,
            "mast_height_m": 20.0,
            "tx_power_mw": 500.0,
            "antenna_gain_dbi": 2.0,
            "receiver_sensitivity_dbm": -137.0,
            "los_dense_sample_step_m": 25.0,
            "max_coverage_radius_m": 4000.0,
        }
        self.config_strict = MeshConfig(
            **common,
            min_fresnel_clearance_m=0.0,
        )
        self.config_budget = MeshConfig(
            **common,
            min_fresnel_clearance_m=None,
        )
        self.terrain = SyntheticTerrain()

    def _los(self, point_name: str, config: MeshConfig):
        return compute_los(
            self.tower_h3,
            self.point_h3[point_name],
            self.cells,
            config,
            elevation_provider=self.terrain,
        )

    def test_strict_los_ground_truth(self):
        west = self._los("west_clear", self.config_strict)
        north = self._los("north_clear", self.config_strict)
        east = self._los("east_blocked", self.config_strict)

        self.assertTrue(west.is_visible)
        self.assertGreater(west.clearance_m, 0.0)
        self.assertLess(west.path_loss_db, self.config_strict.link_budget_db)

        self.assertTrue(north.is_visible)
        self.assertGreater(north.clearance_m, 0.0)
        self.assertLess(north.path_loss_db, self.config_strict.link_budget_db)

        self.assertFalse(east.is_visible)
        self.assertLess(east.clearance_m, 0.0)
        self.assertGreater(east.path_loss_db, self.config_strict.link_budget_db)

    def test_budget_mode_still_blocks_ridge_by_path_loss(self):
        west = self._los("west_clear", self.config_budget)
        north = self._los("north_clear", self.config_budget)
        east = self._los("east_blocked", self.config_budget)

        self.assertTrue(west.is_visible)
        self.assertTrue(north.is_visible)

        self.assertFalse(
            east.is_visible,
            "Blocked east link must remain rejected by link budget even in budget mode.",
        )
        self.assertGreater(east.path_loss_db, self.config_budget.link_budget_db)

    def test_tower_coverage_includes_clear_points_excludes_blocked_point(self):
        coverage = compute_h3_tower_coverage(
            sources=[
                CoverageSource(
                    source_id="tower",
                    h3_index=self.tower_h3,
                    lat=self.tower_lat,
                    lon=self.tower_lon,
                )
            ],
            base_cells=self.cells,
            config=self.config_strict,
            elevation_provider=self.terrain,
            max_radius_m=4000.0,
        )
        covered_h3 = {rec["h3_index"] for rec in coverage}

        self.assertIn(self.tower_h3, covered_h3)
        self.assertIn(self.point_h3["west_clear"], covered_h3)
        self.assertIn(self.point_h3["north_clear"], covered_h3)
        self.assertNotIn(self.point_h3["east_blocked"], covered_h3)


if __name__ == "__main__":
    unittest.main()
