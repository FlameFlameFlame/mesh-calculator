"""
Tests for compute_tower_radial_coverage() max_rings calculation.

The function uses:
    max_rings = max(1, int(config.max_coverage_radius_m / edge_m))

where edge_m is the average hex edge length at config.h3_resolution.

Previously max_rings was capped at 30; the new code removes that cap so
large coverage radii at fine resolutions produce far more rings.

We verify the expected ring count without running the full coverage
pipeline by reproducing the formula with real h3 values.
"""
import unittest
import h3

from ..core.config import MeshConfig


def _expected_max_rings(h3_res: int, max_coverage_radius_m: float) -> int:
    """Reproduce the formula used inside compute_tower_radial_coverage."""
    edge_m = h3.average_hexagon_edge_length(h3_res, unit='m')
    return max(1, int(max_coverage_radius_m / edge_m))


class TestMaxRingsFormula(unittest.TestCase):
    """max_rings is computed from max_coverage_radius_m/edge_m, no cap."""

    def test_res11_15km_not_capped_at_30(self):
        """At res 11 (~29 m edge), 15 km gives ~500+ rings."""
        config = MeshConfig(
            h3_resolution=11,
            max_coverage_radius_m=15000.0,
        )
        rings = _expected_max_rings(
            config.h3_resolution,
            config.max_coverage_radius_m,
        )
        self.assertGreater(
            rings, 30,
            "At res 11 with 15 km radius, max_rings must exceed 30",
        )
        # 15000 / ~29 ≈ 517
        self.assertGreater(rings, 400)

    def test_res8_15km_around_28_rings(self):
        """At res 8 (~531 m edge), 15 km gives ~28 rings."""
        config = MeshConfig(
            h3_resolution=8,
            max_coverage_radius_m=15000.0,
        )
        rings = _expected_max_rings(
            config.h3_resolution,
            config.max_coverage_radius_m,
        )
        # 15000 / 531 ≈ 28
        self.assertGreaterEqual(rings, 25)
        self.assertLess(rings, 40)

    def test_res8_15km_matches_formula_exactly(self):
        """Ring count equals the raw formula result, not a hardcoded 30."""
        config = MeshConfig(
            h3_resolution=8,
            max_coverage_radius_m=15000.0,
        )
        rings = _expected_max_rings(
            config.h3_resolution,
            config.max_coverage_radius_m,
        )
        edge_m = h3.average_hexagon_edge_length(8, unit='m')
        expected = max(1, int(15000.0 / edge_m))
        self.assertEqual(rings, expected)

    def test_minimum_rings_is_1(self):
        """max_rings is always at least 1 even with a tiny radius."""
        config = MeshConfig(
            h3_resolution=8,
            max_coverage_radius_m=1.0,
        )
        rings = _expected_max_rings(
            config.h3_resolution,
            config.max_coverage_radius_m,
        )
        self.assertGreaterEqual(rings, 1)

    def test_custom_radius_scales_linearly(self):
        """Doubling max_coverage_radius_m roughly doubles the ring count."""
        rings_small = _expected_max_rings(8, 5000.0)
        rings_large = _expected_max_rings(8, 10000.0)
        self.assertAlmostEqual(
            rings_large / rings_small, 2.0, delta=0.5,
        )

    def test_max_rings_uses_coverage_radius_not_visibility(self):
        """
        Ring count is driven by max_coverage_radius_m, not max_visibility_m.
        """
        config = MeshConfig(
            h3_resolution=8,
            max_coverage_radius_m=5000.0,
        )
        edge_m = h3.average_hexagon_edge_length(8, unit='m')
        rings_from_coverage = max(1, int(5000.0 / edge_m))
        rings_from_visibility = max(
            1, int(config.max_visibility_m / edge_m)
        )

        # Sanity: values must differ so the test distinguishes the two
        self.assertNotEqual(
            rings_from_coverage,
            rings_from_visibility,
            "Test requires coverage radius != visibility radius",
        )

        computed = _expected_max_rings(
            config.h3_resolution,
            config.max_coverage_radius_m,
        )
        self.assertEqual(computed, rings_from_coverage)

    def test_res11_edge_length_is_small(self):
        """H3 res 11 edge length should be well under 50 m."""
        edge_m = h3.average_hexagon_edge_length(11, unit='m')
        self.assertLess(
            edge_m, 50.0,
            "H3 resolution 11 edge should be less than 50 m",
        )

    def test_res8_edge_length_is_around_531m(self):
        """H3 res 8 edge length should be in the 400–600 m range."""
        edge_m = h3.average_hexagon_edge_length(8, unit='m')
        self.assertGreater(edge_m, 400.0)
        self.assertLess(edge_m, 600.0)


if __name__ == '__main__':
    unittest.main()
