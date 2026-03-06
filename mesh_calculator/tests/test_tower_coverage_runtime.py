"""
Tests for standalone runtime tower coverage computation.
"""
import unittest
from unittest.mock import patch

import h3

from ..core.config import MeshConfig
from ..network.tower_coverage import CoverageSource, compute_h3_tower_coverage


class TestTowerCoverageRuntime(unittest.TestCase):
    def setUp(self):
        self.config = MeshConfig(h3_resolution=8, max_coverage_radius_m=1200.0)
        self.src_h3 = h3.latlng_to_cell(40.1772, 44.5035, self.config.h3_resolution)
        self.src_lat, self.src_lon = h3.cell_to_latlng(self.src_h3)

    def test_source_cell_included_with_zero_path_loss(self):
        results = compute_h3_tower_coverage(
            sources=[CoverageSource(1, self.src_h3, self.src_lat, self.src_lon)],
            base_cells={},
            config=self.config,
            elevation_provider=None,
        )
        by_h3 = {r["h3_index"]: r for r in results}
        self.assertIn(self.src_h3, by_h3)
        self.assertEqual(by_h3[self.src_h3]["distance_m"], 0.0)
        self.assertEqual(by_h3[self.src_h3]["path_loss_db"], 0.0)
        self.assertEqual(by_h3[self.src_h3]["closest_tower_id"], 1)
        self.assertTrue(by_h3[self.src_h3]["is_covered"])

    def test_duplicate_sources_are_deduped_by_h3(self):
        results = compute_h3_tower_coverage(
            sources=[
                CoverageSource(10, self.src_h3, self.src_lat, self.src_lon),
                CoverageSource(99, self.src_h3, self.src_lat, self.src_lon),
            ],
            base_cells={},
            config=self.config,
            elevation_provider=None,
        )
        source_rows = [r for r in results if r["h3_index"] == self.src_h3]
        self.assertEqual(len(source_rows), 1)
        self.assertEqual(source_rows[0]["closest_tower_id"], 10)

    def test_sources_are_snapped_to_requested_resolution(self):
        stale_h3 = h3.latlng_to_cell(self.src_lat, self.src_lon, 8)
        config_res9 = MeshConfig(h3_resolution=9, max_coverage_radius_m=1200.0)
        expected_h3 = h3.latlng_to_cell(self.src_lat, self.src_lon, 9)

        results = compute_h3_tower_coverage(
            sources=[CoverageSource(1, stale_h3, self.src_lat, self.src_lon)],
            base_cells={},
            config=config_res9,
            elevation_provider=None,
        )
        by_h3 = {r["h3_index"]: r for r in results}
        self.assertIn(expected_h3, by_h3)
        self.assertNotIn(stale_h3, by_h3)
        self.assertEqual(by_h3[expected_h3]["closest_tower_id"], 1)

    @patch("mesh_calculator.network.tower_coverage._compute_shadow_link")
    def test_negative_clearance_links_are_blocked(self, mock_shadow_link):
        neighbor_h3 = next(
            h for h in h3.grid_disk(self.src_h3, 1)
            if h != self.src_h3
        )

        def _mock_shadow_link(source_cell, target_cell, *_args, **_kwargs):
            if source_cell.h3_index == target_cell.h3_index:
                return (0.0, 1.5, 0.0, True)
            if source_cell.h3_index == self.src_h3 and target_cell.h3_index == neighbor_h3:
                return (800.0, -2.0, float("inf"), False)
            return (20000.0, -10.0, float("inf"), False)

        mock_shadow_link.side_effect = _mock_shadow_link
        results = compute_h3_tower_coverage(
            sources=[CoverageSource(1, self.src_h3, self.src_lat, self.src_lon)],
            base_cells={},
            config=self.config,
            elevation_provider=None,
        )
        by_h3 = {r["h3_index"]: r for r in results}
        self.assertIn(neighbor_h3, by_h3)
        self.assertFalse(by_h3[neighbor_h3]["is_covered"])
        self.assertIsNone(by_h3[neighbor_h3]["path_loss_db"])

    @patch("mesh_calculator.network.tower_coverage._compute_shadow_link")
    def test_serving_tower_uses_strongest_link(self, mock_shadow_link):
        ring1 = list(h3.grid_ring(self.src_h3, 1))
        target_h3 = ring1[0]
        src2_h3 = next(h for h in ring1 if h != target_h3)
        src2_lat, src2_lon = h3.cell_to_latlng(src2_h3)

        def _mock_shadow_link(source_cell, target_cell, *_args, **_kwargs):
            if source_cell.h3_index == target_cell.h3_index:
                return (0.0, 1.5, 0.0, True)
            if source_cell.h3_index == self.src_h3 and target_cell.h3_index == target_h3:
                return (600.0, 1.0, 130.0, True)
            if source_cell.h3_index == src2_h3 and target_cell.h3_index == target_h3:
                return (900.0, 1.0, 100.0, True)
            return (20000.0, -10.0, float("inf"), False)

        mock_shadow_link.side_effect = _mock_shadow_link
        results = compute_h3_tower_coverage(
            sources=[
                CoverageSource(1, self.src_h3, self.src_lat, self.src_lon),
                CoverageSource(2, src2_h3, src2_lat, src2_lon),
            ],
            base_cells={},
            config=self.config,
            elevation_provider=None,
        )
        by_h3 = {r["h3_index"]: r for r in results}
        self.assertIn(target_h3, by_h3)
        target = by_h3[target_h3]
        self.assertEqual(target["closest_tower_id"], 1)
        self.assertEqual(target["serving_tower_id"], 2)
        self.assertEqual(target["path_loss_db"], 100.0)


if __name__ == "__main__":
    unittest.main()
