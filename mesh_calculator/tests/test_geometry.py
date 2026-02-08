"""
Unit tests for geometry utilities.
"""
import unittest
import h3
from ..core.geometry import (
    great_circle_distance,
    h3_to_lat_lon,
    lat_lon_to_h3,
    h3_distance,
    calculate_line_fraction,
    get_h3_neighbors,
    cells_within_radius
)


class TestGeometry(unittest.TestCase):
    """Test geometric utility functions."""

    def test_great_circle_distance(self):
        """Test great circle distance calculation."""
        # Distance between two known points (roughly)
        # New York to London is approximately 5570 km
        ny_lat, ny_lon = 40.7128, -74.0060
        london_lat, london_lon = 51.5074, -0.1278

        distance = great_circle_distance(ny_lat, ny_lon, london_lat, london_lon)

        # Should be approximately 5.57 million meters
        self.assertGreater(distance, 5_500_000)
        self.assertLess(distance, 5_600_000)

    def test_h3_conversions(self):
        """Test H3 index conversions."""
        lat, lon = 40.7128, -74.0060
        resolution = 8

        # Convert to H3
        h3_idx = lat_lon_to_h3(lat, lon, resolution)
        self.assertIsNotNone(h3_idx)

        # Convert back
        lat2, lon2 = h3_to_lat_lon(h3_idx)

        # Should be close (within same cell)
        self.assertAlmostEqual(lat, lat2, delta=0.01)
        self.assertAlmostEqual(lon, lon2, delta=0.01)

    def test_h3_distance(self):
        """Test distance between H3 cells."""
        # Two cells far enough apart to be in different H3 cells at res 8
        h3_a = lat_lon_to_h3(40.7128, -74.0060, 8)
        h3_b = lat_lon_to_h3(40.7200, -74.0160, 8)

        distance = h3_distance(h3_a, h3_b)

        # Should be small distance (< 2 km) but > 0
        self.assertGreater(distance, 0)
        self.assertLess(distance, 2000)

    def test_get_h3_neighbors(self):
        """Test getting H3 neighbors."""
        h3_idx = lat_lon_to_h3(40.7128, -74.0060, 8)

        # Get immediate neighbors (k=1)
        neighbors = get_h3_neighbors(h3_idx, k=1)

        # Should have 6 neighbors for hexagon
        self.assertEqual(len(neighbors), 6)

        # All neighbors should be different
        self.assertNotIn(h3_idx, neighbors)

    def test_cells_within_radius(self):
        """Test finding cells within radius."""
        center = lat_lon_to_h3(40.7128, -74.0060, 8)

        # Find cells within 1 km
        cells = cells_within_radius(center, 1000, 8)

        # Should include center
        self.assertIn(center, cells)

        # Should have multiple cells
        self.assertGreater(len(cells), 1)

    def test_calculate_line_fraction(self):
        """Test line fraction calculation."""
        # Create a simple line
        start = lat_lon_to_h3(40.0, -74.0, 8)
        end = lat_lon_to_h3(40.1, -74.0, 8)
        mid = lat_lon_to_h3(40.05, -74.0, 8)

        # Fraction should be between 0 and 1
        frac = calculate_line_fraction(mid, start, end)

        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

        # Start should have fraction ~0
        frac_start = calculate_line_fraction(start, start, end)
        self.assertLess(frac_start, 0.1)

        # End should have fraction ~1
        frac_end = calculate_line_fraction(end, start, end)
        self.assertGreater(frac_end, 0.9)


if __name__ == '__main__':
    unittest.main()
