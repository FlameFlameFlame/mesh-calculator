"""
Unit tests for physics calculations.
"""
import unittest
import math
from unittest.mock import patch
from ..physics.path_loss import compute_path_loss, fspl_only
from ..physics.los import compute_los
from ..core.config import MeshConfig
from ..core.grid import H3Cell


class TestPathLoss(unittest.TestCase):
    """Test path loss calculations."""

    def test_fspl_only(self):
        """Test free-space path loss calculation."""
        # Test known values
        distance_m = 1000  # 1 km
        frequency_hz = 868e6  # 868 MHz

        fspl = fspl_only(distance_m, frequency_hz)

        # FSPL = 20*log10(1) + 20*log10(868) + 32.44
        # = 0 + 58.77 + 32.44 = 91.21 dB
        expected = 20 * math.log10(1) + 20 * math.log10(868) + 32.44

        self.assertAlmostEqual(fspl, expected, places=1)

    def test_path_loss_clear_los(self):
        """Test path loss with clear LOS (positive clearance)."""
        distance_m = 10000  # 10 km
        frequency_hz = 868e6
        clearance_m = 10.0  # Clear by 10 meters

        path_loss = compute_path_loss(distance_m, frequency_hz, clearance_m)

        # Should equal FSPL (no diffraction)
        fspl = fspl_only(distance_m, frequency_hz)

        self.assertAlmostEqual(path_loss, fspl, places=1)

    def test_path_loss_obstructed(self):
        """Test path loss with obstruction (negative clearance)."""
        distance_m = 10000  # 10 km
        frequency_hz = 868e6
        clearance_m = -5.0  # Obstructed by 5 meters

        path_loss = compute_path_loss(distance_m, frequency_hz, clearance_m)

        # Should be greater than FSPL (diffraction loss added)
        fspl = fspl_only(distance_m, frequency_hz)

        self.assertGreater(path_loss, fspl)

    def test_path_loss_validation(self):
        """Test input validation."""
        frequency_hz = 868e6

        # Invalid distance
        with self.assertRaises(ValueError):
            compute_path_loss(0, frequency_hz, 10.0)

        with self.assertRaises(ValueError):
            compute_path_loss(-100, frequency_hz, 10.0)

        # Invalid frequency
        with self.assertRaises(ValueError):
            compute_path_loss(1000, 0, 10.0)

        with self.assertRaises(ValueError):
            compute_path_loss(1000, -868e6, 10.0)


class TestFresnelClearance(unittest.TestCase):
    """Test Fresnel clearance calculations."""

    def test_wavelength_calculation(self):
        """Test wavelength calculation in config."""
        config = MeshConfig()

        # Wavelength = c / f
        # For 868 MHz: 299792458 / 868e6 ≈ 0.345 meters
        expected = 299792458 / 868e6

        self.assertAlmostEqual(config.wavelength_m, expected, places=3)

    def test_effective_earth_radius(self):
        """Test effective earth radius calculation."""
        config = MeshConfig()

        # Effective radius = 4/3 * earth_radius
        expected = (4.0 / 3.0) * 6371000

        self.assertAlmostEqual(config.effective_earth_radius_m, expected, places=0)


class TestLOSVisibilityRule(unittest.TestCase):
    """Visibility should follow link budget, not only positive clearance."""

    @patch('mesh_calculator.physics.los.compute_path_loss')
    @patch('mesh_calculator.physics.los.compute_fresnel_clearance')
    @patch('mesh_calculator.physics.los.h3_distance')
    def test_negative_clearance_can_still_be_visible_if_budget_allows(
        self, mock_distance, mock_fresnel, mock_path_loss
    ):
        config = MeshConfig(
            frequency_hz=868e6,
            mast_height_m=2.0,
            tx_power_mw=500.0,
            antenna_gain_dbi=2.0,
            receiver_sensitivity_dbm=-137.0,
        )
        mock_distance.return_value = 1000.0
        mock_fresnel.return_value = (-10.0, 1000.0, 500.0, 500.0)
        mock_path_loss.return_value = config.link_budget_db - 5.0
        cells = {
            "a": H3Cell("a", 40.0, 44.0, 100.0, has_road=True),
            "b": H3Cell("b", 40.0, 44.01, 100.0, has_road=True),
        }

        result = compute_los("a", "b", cells, config)

        self.assertTrue(result.is_visible)
        self.assertLess(result.clearance_m, 0.0)

    @patch('mesh_calculator.physics.los.compute_path_loss')
    @patch('mesh_calculator.physics.los.compute_fresnel_clearance')
    @patch('mesh_calculator.physics.los.h3_distance')
    def test_link_budget_exceeded_is_not_visible_even_if_clearance_positive(
        self, mock_distance, mock_fresnel, mock_path_loss
    ):
        config = MeshConfig()
        mock_distance.return_value = 1000.0
        mock_fresnel.return_value = (10.0, 1000.0, 500.0, 500.0)
        mock_path_loss.return_value = config.link_budget_db + 1.0
        cells = {
            "a": H3Cell("a", 40.0, 44.0, 100.0, has_road=True),
            "b": H3Cell("b", 40.0, 44.01, 100.0, has_road=True),
        }

        result = compute_los("a", "b", cells, config)

        self.assertFalse(result.is_visible)


if __name__ == '__main__':
    unittest.main()
