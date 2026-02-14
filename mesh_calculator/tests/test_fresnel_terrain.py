"""
Tests for Fix #1: Fresnel clearance must check terrain for cells not in the grid.

The bug: compute_fresnel_clearance skips cells not in the grid dict.
Since the grid only contains road cells, terrain obstacles (mountains)
between road segments are invisible to LOS checks.

The fix: accept an optional elevation_provider parameter. When a path cell
is not in the grid, query elevation from the provider instead of skipping it.
"""
import unittest
from unittest.mock import Mock
import h3

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..physics.fresnel import compute_fresnel_clearance, has_line_of_sight
from ..physics.los import compute_los, has_los


class TestFresnelTerrainVisibility(unittest.TestCase):
    """Test that Fresnel clearance accounts for terrain between road cells."""

    def setUp(self):
        """Set up test cells with known positions."""
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            h3_resolution=8,
        )

        # Two points ~8.5km apart at lat 40N
        self.src_h3 = h3.latlng_to_cell(40.0, 44.0, 8)
        self.dst_h3 = h3.latlng_to_cell(40.0, 44.1, 8)

        # Get actual centroids
        src_lat, src_lon = h3.cell_to_latlng(self.src_h3)
        dst_lat, dst_lon = h3.cell_to_latlng(self.dst_h3)

        # Create cells dict with ONLY endpoints (simulating road-only grid)
        self.cells = {
            self.src_h3: H3Cell(
                h3_index=self.src_h3, lat=src_lat, lon=src_lon,
                elevation=100.0, has_road=True,
            ),
            self.dst_h3: H3Cell(
                h3_index=self.dst_h3, lat=dst_lat, lon=dst_lon,
                elevation=100.0, has_road=True,
            ),
        }

        # Verify intermediate cells exist and are NOT in grid
        self.path_cells = list(h3.grid_path_cells(self.src_h3, self.dst_h3))
        self.intermediate_cells = [
            c for c in self.path_cells
            if c != self.src_h3 and c != self.dst_h3
        ]
        assert len(self.intermediate_cells) > 0, "Need intermediate cells for test"

    def test_mountain_invisible_without_elevation_provider(self):
        """Without elevation_provider, intermediate terrain is skipped (the bug)."""
        clearance, distance, _, _ = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config
        )

        # Without intermediate cells checked, clearance is optimistic (mast height)
        self.assertGreater(clearance, 0,
            "Without elevation_provider, missing terrain is skipped "
            "— clearance is incorrectly positive")

    def test_mountain_detected_with_elevation_provider(self):
        """With elevation_provider, a mountain between endpoints blocks LOS."""
        mock_provider = Mock()
        mock_provider.get_elevation.return_value = 2000.0  # Mountain!

        clearance, distance, _, _ = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=mock_provider,
        )

        # Mountain at 2000m between endpoints at 100m should block LOS
        self.assertLess(clearance, 0,
            "Mountain between endpoints should block LOS (negative clearance)")

        # Provider should have been called for intermediate cells
        self.assertGreater(mock_provider.get_elevation.call_count, 0,
            "Elevation provider should be queried for cells not in grid")

    def test_low_terrain_still_clear_with_provider(self):
        """With elevation_provider returning low elevation, LOS should be clear."""
        mock_provider = Mock()
        # Intermediate terrain at 50m — well below the 110m line altitude
        # (endpoints at 100m + 10m mast), leaving room for Fresnel zone
        mock_provider.get_elevation.return_value = 50.0

        clearance, distance, _, _ = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=mock_provider,
        )

        # Low terrain should allow LOS
        self.assertGreater(clearance, 0,
            "Low terrain between endpoints should not block LOS")

    def test_backward_compatible_without_provider(self):
        """Without elevation_provider, behavior matches original (skip unknown cells)."""
        clearance_without, _, _, _ = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config
        )
        clearance_with_none, _, _, _ = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=None,
        )

        self.assertEqual(clearance_without, clearance_with_none)

    def test_provider_only_called_for_missing_cells(self):
        """Elevation provider should NOT be called for cells already in the grid."""
        mock_provider = Mock()
        mock_provider.get_elevation.return_value = 100.0

        compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=mock_provider,
        )

        # Check that provider was NOT called with src/dst coordinates
        # (those are already in the cells dict)
        called_coords = [
            (call.args[0], call.args[1])
            for call in mock_provider.get_elevation.call_args_list
        ]

        src_lat, src_lon = h3.cell_to_latlng(self.src_h3)
        dst_lat, dst_lon = h3.cell_to_latlng(self.dst_h3)

        for lat, lon in called_coords:
            self.assertFalse(
                abs(lat - src_lat) < 0.0001 and abs(lon - src_lon) < 0.0001,
                "Provider should not be called for src cell (already in grid)"
            )
            self.assertFalse(
                abs(lat - dst_lat) < 0.0001 and abs(lon - dst_lon) < 0.0001,
                "Provider should not be called for dst cell (already in grid)"
            )


class TestLOSWithElevationProvider(unittest.TestCase):
    """Test that has_los and compute_los pass elevation_provider through."""

    def setUp(self):
        """Set up same scenario as terrain visibility tests."""
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_visibility_m=70000.0,
            h3_resolution=8,
        )

        self.src_h3 = h3.latlng_to_cell(40.0, 44.0, 8)
        self.dst_h3 = h3.latlng_to_cell(40.0, 44.1, 8)

        src_lat, src_lon = h3.cell_to_latlng(self.src_h3)
        dst_lat, dst_lon = h3.cell_to_latlng(self.dst_h3)

        self.cells = {
            self.src_h3: H3Cell(
                h3_index=self.src_h3, lat=src_lat, lon=src_lon,
                elevation=100.0, has_road=True,
            ),
            self.dst_h3: H3Cell(
                h3_index=self.dst_h3, lat=dst_lat, lon=dst_lon,
                elevation=100.0, has_road=True,
            ),
        }

    def test_has_los_with_mountain_provider(self):
        """has_los should detect mountain when given elevation_provider."""
        mock_provider = Mock()
        mock_provider.get_elevation.return_value = 2000.0

        result = has_los(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=mock_provider,
        )

        self.assertFalse(result, "Mountain should block LOS via has_los")

    def test_has_los_without_provider_misses_mountain(self):
        """has_los without provider should miss the mountain (existing bug behavior)."""
        result = has_los(
            self.src_h3, self.dst_h3, self.cells, self.config,
        )

        # Without provider, intermediate terrain is skipped
        self.assertTrue(result,
            "Without provider, mountain is invisible — LOS incorrectly True")

    def test_compute_los_with_mountain_provider(self):
        """compute_los should return is_visible=False when mountain detected."""
        mock_provider = Mock()
        mock_provider.get_elevation.return_value = 2000.0

        result = compute_los(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=mock_provider,
        )

        self.assertFalse(result.is_visible,
            "compute_los should report not visible with mountain")
        self.assertLess(result.clearance_m, 0,
            "Clearance should be negative with mountain")

    def test_has_line_of_sight_with_provider(self):
        """has_line_of_sight (in fresnel.py) should also accept elevation_provider."""
        mock_provider = Mock()
        mock_provider.get_elevation.return_value = 2000.0

        result = has_line_of_sight(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=mock_provider,
        )

        self.assertFalse(result, "Mountain should block has_line_of_sight")


if __name__ == '__main__':
    unittest.main()
