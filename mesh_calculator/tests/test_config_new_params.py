"""
Tests for new MeshConfig fields: road_buffer_m and max_coverage_radius_m.

Covers:
  - Default values are correct
  - MeshCalculatorConfig.from_dict() handles these new params
  - from_dict() works without them (backward compatibility)
"""
import unittest

from ..core.config import MeshConfig, MeshCalculatorConfig


class TestMeshConfigDefaults(unittest.TestCase):
    """New MeshConfig fields have the right default values."""

    def test_road_buffer_m_default_is_zero(self):
        config = MeshConfig()
        self.assertEqual(config.road_buffer_m, 0.0)

    def test_max_coverage_radius_m_default_is_15000(self):
        config = MeshConfig()
        self.assertEqual(config.max_coverage_radius_m, 15000.0)

    def test_road_buffer_m_can_be_set(self):
        config = MeshConfig(road_buffer_m=300.0)
        self.assertEqual(config.road_buffer_m, 300.0)

    def test_max_coverage_radius_m_can_be_set(self):
        config = MeshConfig(max_coverage_radius_m=5000.0)
        self.assertEqual(config.max_coverage_radius_m, 5000.0)

    def test_other_defaults_unaffected(self):
        """Ensure existing defaults are not disturbed by the new fields."""
        config = MeshConfig()
        self.assertEqual(config.h3_resolution, 8)
        self.assertAlmostEqual(config.frequency_hz, 868e6)
        self.assertEqual(config.mast_height_m, 28.0)

    def test_min_fresnel_clearance_default_is_none(self):
        config = MeshConfig()
        self.assertIsNone(config.min_fresnel_clearance_m)

    def test_min_fresnel_clearance_can_be_set(self):
        config = MeshConfig(min_fresnel_clearance_m=0.0)
        self.assertEqual(config.min_fresnel_clearance_m, 0.0)


class TestMeshCalculatorConfigFromDict(unittest.TestCase):
    """MeshCalculatorConfig.from_dict() correctly handles new params."""

    def test_road_buffer_m_passed_via_parameters(self):
        d = {'parameters': {'road_buffer_m': 500.0}}
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertEqual(cfg.parameters.road_buffer_m, 500.0)

    def test_max_coverage_radius_m_passed_via_parameters(self):
        d = {'parameters': {'max_coverage_radius_m': 8000.0}}
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertEqual(cfg.parameters.max_coverage_radius_m, 8000.0)

    def test_both_new_params_together(self):
        d = {
            'parameters': {
                'road_buffer_m': 200.0,
                'max_coverage_radius_m': 12000.0,
            }
        }
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertEqual(cfg.parameters.road_buffer_m, 200.0)
        self.assertEqual(cfg.parameters.max_coverage_radius_m, 12000.0)

    def test_backward_compat_no_road_buffer_m(self):
        """from_dict without road_buffer_m should not raise and use default."""
        d = {'parameters': {'h3_resolution': 9}}
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertEqual(cfg.parameters.road_buffer_m, 0.0)

    def test_backward_compat_no_max_coverage_radius_m(self):
        """from_dict without max_coverage_radius_m should not raise and use default."""
        d = {'parameters': {'h3_resolution': 9}}
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertEqual(cfg.parameters.max_coverage_radius_m, 15000.0)

    def test_backward_compat_empty_parameters(self):
        """from_dict with empty parameters dict uses all defaults."""
        cfg = MeshCalculatorConfig.from_dict({})
        self.assertEqual(cfg.parameters.road_buffer_m, 0.0)
        self.assertEqual(cfg.parameters.max_coverage_radius_m, 15000.0)

    def test_from_dict_strips_removed_max_visibility_m(self):
        """Legacy max_visibility_m key is silently dropped (it is now a property)."""
        d = {'parameters': {'max_visibility_m': 99999.0}}
        # Should not raise a TypeError about unexpected keyword argument
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertIsInstance(cfg.parameters, MeshConfig)

    def test_new_params_coexist_with_other_settings(self):
        d = {
            'parameters': {
                'road_buffer_m': 150.0,
                'max_coverage_radius_m': 10000.0,
                'min_fresnel_clearance_m': -1.5,
                'mast_height_m': 35.0,
                'routing_k_ring': 3,
            }
        }
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertEqual(cfg.parameters.road_buffer_m, 150.0)
        self.assertEqual(cfg.parameters.max_coverage_radius_m, 10000.0)
        self.assertEqual(cfg.parameters.min_fresnel_clearance_m, -1.5)
        self.assertEqual(cfg.parameters.mast_height_m, 35.0)
        self.assertEqual(cfg.parameters.routing_k_ring, 3)

    def test_legacy_output_tower_coverage_key_is_ignored(self):
        d = {
            'outputs': {
                'towers': 'towers.geojson',
                'coverage': 'coverage.geojson',
                'tower_coverage': 'tower_coverage.geojson',
                'report': 'report.json',
            }
        }
        cfg = MeshCalculatorConfig.from_dict(d)
        self.assertEqual(cfg.outputs.towers, 'towers.geojson')
        self.assertEqual(cfg.outputs.coverage, 'coverage.geojson')
        self.assertEqual(cfg.outputs.report, 'report.json')
        self.assertFalse(hasattr(cfg.outputs, 'tower_coverage'))


if __name__ == '__main__':
    unittest.main()
