"""
Tests for Fix #1: Fresnel clearance must check terrain for cells not in the grid.

The bug: compute_fresnel_clearance skips cells not in the grid dict.
Since the grid only contains road cells, terrain obstacles (mountains)
between road segments are invisible to LOS checks.

The fix: accept an optional elevation_provider parameter. When a path cell
is not in the grid, query elevation from the provider instead of skipping it.
"""
import unittest
import math
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


class TestCorridorCellsRemoved(unittest.TestCase):
    """
    Regression tests for Bug 1 fix: corridor_cells parameter removed.

    After the fix, compute_fresnel_clearance and compute_los no longer accept
    a corridor_cells argument — they always use h3.grid_path_cells (straight-line path).
    """

    def setUp(self):
        self.config = MeshConfig(mast_height_m=10.0, h3_resolution=8)
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

    def test_corridor_cells_removed_from_compute_los_signature(self):
        """compute_los must not accept corridor_cells — parameter was removed."""
        import inspect
        from ..physics.los import compute_los as _compute_los
        self.assertNotIn(
            'corridor_cells',
            inspect.signature(_compute_los).parameters,
            "corridor_cells parameter must be removed from compute_los",
        )

    def test_corridor_cells_removed_from_compute_fresnel_clearance_signature(self):
        """compute_fresnel_clearance must not accept corridor_cells."""
        import inspect
        self.assertNotIn(
            'corridor_cells',
            inspect.signature(compute_fresnel_clearance).parameters,
            "corridor_cells parameter must be removed from compute_fresnel_clearance",
        )

    def test_straight_line_mountain_blocks_los(self):
        """Mountain on the straight-line RF path must block LOS regardless of any road detour."""
        mountain_provider = Mock()
        mountain_provider.get_elevation.return_value = 2000.0  # high mountain on straight-line path

        clearance, *_ = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config,
            elevation_provider=mountain_provider,
        )
        self.assertLess(
            clearance, 0,
            "Straight-line mountain must block LOS regardless of road path",
        )


class TestFracsCosineCorrectionAccuracy(unittest.TestCase):
    """
    Regression tests for Bug 2 fix: fracs projection uses cos(lat) scaling.

    For a purely E-W path (Δlat=0), a sample at the midpoint should project
    to frac ≈ 0.5.  The old formula (no cos(lat)) produced the same result for
    E-W paths, so the key test is that d1+d2 == total_distance for all samples.
    """

    def setUp(self):
        self.config = MeshConfig(mast_height_m=10.0, h3_resolution=8)
        self.src_h3 = h3.latlng_to_cell(40.0, 44.0, 8)
        self.dst_h3 = h3.latlng_to_cell(40.0, 44.1, 8)
        src_lat, src_lon = h3.cell_to_latlng(self.src_h3)
        dst_lat, dst_lon = h3.cell_to_latlng(self.dst_h3)

        # Build cells for every intermediate path cell
        path_cells = list(h3.grid_path_cells(self.src_h3, self.dst_h3))
        self.cells = {}
        for cell in path_cells:
            clat, clon = h3.cell_to_latlng(cell)
            self.cells[cell] = H3Cell(
                h3_index=cell, lat=clat, lon=clon,
                elevation=100.0, has_road=True,
            )

    def test_d1_plus_d2_equals_total_distance(self):
        """d1 + d2 must equal total_distance for the worst-clearance cell."""
        clearance, total_dist, d1, d2 = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config,
        )
        self.assertAlmostEqual(
            d1 + d2, total_dist, delta=total_dist * 0.01,
            msg="d1 + d2 must equal total_distance (within 1%)",
        )

    def test_clearance_is_finite(self):
        """Clearance must be a real finite number."""
        import math as _math
        clearance, *_ = compute_fresnel_clearance(
            self.src_h3, self.dst_h3, self.cells, self.config,
        )
        self.assertTrue(_math.isfinite(clearance), "Clearance must be finite")


class TestWorstClearanceNotEqualMaxTerrain(unittest.TestCase):
    """Regression: worst obstruction is argmin(clearance), not argmax(terrain)."""

    class _SlopedTerrainWithRidge:
        def __init__(self, src_lon: float, dst_lon: float):
            self.src_lon = src_lon
            self.dst_lon = dst_lon

        def _frac(self, lon: float) -> float:
            denom = (self.dst_lon - self.src_lon)
            if abs(denom) < 1e-12:
                return 0.0
            return max(0.0, min(1.0, (lon - self.src_lon) / denom))

        def get_elevation(self, lat: float, lon: float) -> float:
            frac = self._frac(lon)
            # Monotonic rise to destination (destination is global max terrain).
            base = 120.0 + 650.0 * frac
            # Mid/late ridge that is lower than destination but blocks Fresnel.
            ridge = 120.0 * math.exp(-((frac - 0.75) / 0.04) ** 2)
            return base + ridge

    def test_midpath_ridge_can_block_even_if_max_terrain_is_at_endpoint(self):
        config = MeshConfig(mast_height_m=12.0, h3_resolution=8)
        src_h3 = h3.latlng_to_cell(40.0, 44.0, 8)
        dst_h3 = h3.latlng_to_cell(40.0, 44.35, 8)
        src_lat, src_lon = h3.cell_to_latlng(src_h3)
        dst_lat, dst_lon = h3.cell_to_latlng(dst_h3)

        terrain = self._SlopedTerrainWithRidge(src_lon, dst_lon)
        src_elev = terrain.get_elevation(src_lat, src_lon)
        dst_elev = terrain.get_elevation(dst_lat, dst_lon)

        cells = {
            src_h3: H3Cell(
                h3_index=src_h3, lat=src_lat, lon=src_lon,
                elevation=src_elev, has_road=True,
            ),
            dst_h3: H3Cell(
                h3_index=dst_h3, lat=dst_lat, lon=dst_lon,
                elevation=dst_elev, has_road=True,
            ),
        }

        clearance, total_dist, d1, d2 = compute_fresnel_clearance(
            src_h3, dst_h3, cells, config, elevation_provider=terrain
        )

        self.assertLess(
            clearance, 0.0,
            "Midpath ridge must block link even when destination has highest terrain.",
        )
        self.assertLess(
            d1 / total_dist, 0.98,
            "Worst-clearance point should not collapse to endpoint-only max-terrain check.",
        )
        self.assertAlmostEqual(d1 + d2, total_dist, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
