"""
Unit tests for caching.
"""
import unittest
import h3
from unittest.mock import patch

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..data.cache import LOSCache, LOSResult
from ..parallel.los_compute import compute_los_batch


class TestLOSCache(unittest.TestCase):
    """Test LOS cache functionality."""

    def setUp(self):
        """Set up test cache."""
        self.cache = LOSCache()

    def test_cache_put_get(self):
        """Test storing and retrieving from cache."""
        result = LOSResult(
            clearance_m=10.0,
            path_loss_db=100.0,
            distance_m=5000.0,
            is_visible=True
        )

        # Put in cache
        self.cache.put('h3_a', 'h3_b', 28.0, 28.0, 868e6, result)

        # Get from cache
        cached = self.cache.get('h3_a', 'h3_b', 28.0, 28.0, 868e6)

        self.assertIsNotNone(cached)
        self.assertEqual(cached.clearance_m, 10.0)
        self.assertEqual(cached.path_loss_db, 100.0)
        self.assertEqual(cached.is_visible, True)

    def test_cache_symmetry(self):
        """Test cache key symmetry (a->b same as b->a)."""
        result = LOSResult(
            clearance_m=10.0,
            path_loss_db=100.0,
            distance_m=5000.0,
            is_visible=True
        )

        # Put a->b
        self.cache.put('h3_a', 'h3_b', 28.0, 28.0, 868e6, result)

        # Get b->a (should be same due to symmetry)
        cached = self.cache.get('h3_b', 'h3_a', 28.0, 28.0, 868e6)

        self.assertIsNotNone(cached)

    def test_cache_miss(self):
        """Test cache miss."""
        cached = self.cache.get('h3_x', 'h3_y', 28.0, 28.0, 868e6)

        self.assertIsNone(cached)

    def test_cache_stats(self):
        """Test cache statistics."""
        result = LOSResult(10.0, 100.0, 5000.0, True)

        # Put one entry
        self.cache.put('h3_a', 'h3_b', 28.0, 28.0, 868e6, result)

        # Hit
        self.cache.get('h3_a', 'h3_b', 28.0, 28.0, 868e6)

        # Miss
        self.cache.get('h3_x', 'h3_y', 28.0, 28.0, 868e6)

        stats = self.cache.stats()

        self.assertEqual(stats['size'], 1)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)

    def test_cache_clear(self):
        """Test cache clearing."""
        result = LOSResult(10.0, 100.0, 5000.0, True)

        self.cache.put('h3_a', 'h3_b', 28.0, 28.0, 868e6, result)

        self.assertEqual(self.cache.stats()['size'], 1)

        self.cache.clear()

        self.assertEqual(self.cache.stats()['size'], 0)

    def test_cache_key_includes_receiver_sensitivity(self):
        """Different RX sensitivity values must produce different cache keys."""
        result = LOSResult(5.0, 140.0, 10000.0, True)
        self.cache.put(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6, result,
            tx_power_mw=500.0,
            antenna_gain_dbi=2.0,
            receiver_sensitivity_dbm=-137.0,
            min_fresnel_clearance_m=None,
        )

        cached_same = self.cache.get(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6,
            tx_power_mw=500.0,
            antenna_gain_dbi=2.0,
            receiver_sensitivity_dbm=-137.0,
            min_fresnel_clearance_m=None,
        )
        cached_diff = self.cache.get(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6,
            tx_power_mw=500.0,
            antenna_gain_dbi=2.0,
            receiver_sensitivity_dbm=-120.0,
            min_fresnel_clearance_m=None,
        )

        self.assertIsNotNone(cached_same)
        self.assertIsNone(cached_diff)

    def test_cache_key_includes_min_fresnel_threshold(self):
        """Different min clearance policy values must not reuse cached LOS."""
        result = LOSResult(-3.0, 150.0, 12000.0, True)
        self.cache.put(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6, result,
            tx_power_mw=500.0,
            antenna_gain_dbi=2.0,
            receiver_sensitivity_dbm=-137.0,
            min_fresnel_clearance_m=None,
        )

        cached_none = self.cache.get(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6,
            tx_power_mw=500.0,
            antenna_gain_dbi=2.0,
            receiver_sensitivity_dbm=-137.0,
            min_fresnel_clearance_m=None,
        )
        cached_zero = self.cache.get(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6,
            tx_power_mw=500.0,
            antenna_gain_dbi=2.0,
            receiver_sensitivity_dbm=-137.0,
            min_fresnel_clearance_m=0.0,
        )

        self.assertIsNotNone(cached_none)
        self.assertIsNone(cached_zero)

    def test_cache_key_includes_dense_profile_settings(self):
        """Different dense-verify settings must not reuse cached LOS."""
        result = LOSResult(-1.0, 150.0, 12000.0, False)
        self.cache.put(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6, result,
            los_dense_sample_step_m=50.0,
            los_dense_max_samples=400,
            los_verification_mode='hybrid_accept_verify',
        )

        cached_same = self.cache.get(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6,
            los_dense_sample_step_m=50.0,
            los_dense_max_samples=400,
            los_verification_mode='hybrid_accept_verify',
        )
        cached_diff = self.cache.get(
            'h3_a', 'h3_b', 28.0, 28.0, 868e6,
            los_dense_sample_step_m=25.0,
            los_dense_max_samples=400,
            los_verification_mode='hybrid_accept_verify',
        )

        self.assertIsNotNone(cached_same)
        self.assertIsNone(cached_diff)

    def test_cache_key_distinguishes_endpoint_mast_heights(self):
        """Per-endpoint mast heights must be part of cache identity."""
        result = LOSResult(2.0, 120.0, 5000.0, True)
        self.cache.put(
            'h3_a', 'h3_b',
            28.0, 30.0,
            868e6,
            result,
        )

        cached_same = self.cache.get(
            'h3_a', 'h3_b',
            28.0, 30.0,
            868e6,
        )
        cached_diff = self.cache.get(
            'h3_a', 'h3_b',
            28.0, 31.0,
            868e6,
        )

        self.assertIsNotNone(cached_same)
        self.assertIsNone(cached_diff)

    def test_cache_key_includes_endpoint_coordinates(self):
        """Same H3 pair with different endpoint anchors must not collide."""
        result = LOSResult(1.0, 100.0, 1000.0, True)
        self.cache.put(
            'h3_a', 'h3_b',
            10.0, 10.0,
            868e6,
            result,
            src_lat=40.0001,
            src_lon=44.0001,
            dst_lat=40.0002,
            dst_lon=44.0002,
        )

        cached_same = self.cache.get(
            'h3_a', 'h3_b',
            10.0, 10.0,
            868e6,
            src_lat=40.0001,
            src_lon=44.0001,
            dst_lat=40.0002,
            dst_lon=44.0002,
        )
        cached_diff = self.cache.get(
            'h3_a', 'h3_b',
            10.0, 10.0,
            868e6,
            src_lat=40.0101,
            src_lon=44.0101,
            dst_lat=40.0002,
            dst_lon=44.0002,
        )

        self.assertIsNotNone(cached_same)
        self.assertIsNone(cached_diff)

    def test_cache_key_endpoint_coordinates_remain_symmetric(self):
        """Symmetry should still hold when endpoint coordinates are provided."""
        result = LOSResult(1.0, 100.0, 1000.0, True)
        self.cache.put(
            'h3_a', 'h3_b',
            10.0, 11.0,
            868e6,
            result,
            src_lat=40.1,
            src_lon=44.1,
            dst_lat=40.2,
            dst_lon=44.2,
        )

        cached = self.cache.get(
            'h3_b', 'h3_a',
            11.0, 10.0,
            868e6,
            src_lat=40.2,
            src_lon=44.2,
            dst_lat=40.1,
            dst_lon=44.1,
        )
        self.assertIsNotNone(cached)

class TestLOSBatchExecution(unittest.TestCase):
    """Batch LOS computation is deterministic and reuses canonical pair work."""

    def setUp(self):
        self.config = MeshConfig()
        self.cells = {
            "a": H3Cell("a", 40.0, 44.0, 100.0, has_road=True),
            "b": H3Cell("b", 40.1, 44.1, 100.0, has_road=True),
            "c": H3Cell("c", 40.2, 44.2, 100.0, has_road=True),
        }

    @patch('mesh_calculator.parallel.los_compute.compute_los')
    def test_batch_returns_results_for_duplicate_and_symmetric_pairs(self, mock_compute_los):
        def _fake(src, dst, *_args, **_kwargs):
            is_visible = {src, dst} != {"a", "c"}
            return LOSResult(
                clearance_m=1.0,
                path_loss_db=100.0,
                distance_m=1000.0,
                is_visible=is_visible,
            )

        mock_compute_los.side_effect = _fake
        pairs = [("a", "b"), ("b", "a"), ("a", "b"), ("a", "c")]

        results = compute_los_batch(pairs, self.cells, self.config)

        self.assertEqual(set(results.keys()), set(pairs))
        self.assertTrue(results[("a", "b")].is_visible)
        self.assertTrue(results[("b", "a")].is_visible)
        self.assertFalse(results[("a", "c")].is_visible)
        self.assertEqual(mock_compute_los.call_count, 2)

    @patch('mesh_calculator.parallel.los_compute.compute_los')
    def test_batch_matches_serial_result_objects(self, mock_compute_los):
        def _fake(src, dst, *_args, **_kwargs):
            distance = 1000.0 if {src, dst} == {"a", "b"} else 2000.0
            return LOSResult(
                clearance_m=5.0,
                path_loss_db=95.0,
                distance_m=distance,
                is_visible=True,
            )

        mock_compute_los.side_effect = _fake
        pairs = [("a", "b"), ("b", "c")]

        results = compute_los_batch(pairs, self.cells, self.config, max_workers=2)

        self.assertEqual(results[("a", "b")].distance_m, 1000.0)
        self.assertEqual(results[("b", "c")].distance_m, 2000.0)

    @patch('mesh_calculator.parallel.los_compute.compute_los')
    def test_batch_can_force_serial_mode_via_threshold(
        self,
        mock_compute_los,
    ):
        mock_compute_los.return_value = LOSResult(
            clearance_m=5.0,
            path_loss_db=95.0,
            distance_m=1000.0,
            is_visible=True,
        )
        pairs = [("a", "b"), ("b", "a"), ("a", "b")]

        results = compute_los_batch(
            pairs,
            self.cells,
            self.config,
            min_pairs_for_parallel=999,
        )

        self.assertEqual(set(results.keys()), set(pairs))
        self.assertEqual(mock_compute_los.call_count, 1)

    @patch('mesh_calculator.parallel.los_compute.compute_los')
    def test_batch_parallel_matches_serial_exactly(self, mock_compute_los):
        def _fake(src, dst, *_args, **_kwargs):
            pair_key = ''.join(sorted([src, dst]))
            distance = 1000.0 + (100.0 * len(pair_key))
            return LOSResult(
                clearance_m=1.0 + len(pair_key),
                path_loss_db=80.0 + len(pair_key),
                distance_m=distance,
                is_visible=True,
            )

        mock_compute_los.side_effect = _fake
        pairs = [("a", "b"), ("b", "c"), ("a", "c"), ("c", "b"), ("a", "b")]

        serial_results = compute_los_batch(
            pairs,
            self.cells,
            self.config,
            max_workers=1,
            min_pairs_for_parallel=1,
        )
        parallel_results = compute_los_batch(
            pairs,
            self.cells,
            self.config,
            max_workers=3,
            min_pairs_for_parallel=1,
        )

        self.assertEqual(set(serial_results.keys()), set(parallel_results.keys()))
        for key in serial_results:
            self.assertEqual(
                serial_results[key].clearance_m,
                parallel_results[key].clearance_m,
            )
            self.assertEqual(
                serial_results[key].path_loss_db,
                parallel_results[key].path_loss_db,
            )
            self.assertEqual(
                serial_results[key].distance_m,
                parallel_results[key].distance_m,
            )
            self.assertEqual(
                serial_results[key].is_visible,
                parallel_results[key].is_visible,
            )


if __name__ == '__main__':
    unittest.main()


class _FlatElevationProvider:
    def get_elevation(self, _lat: float, _lon: float) -> float:
        return 100.0


class TestLOSBatchDiagnosticsAndRealDeterminism(unittest.TestCase):
    def setUp(self):
        self.config = MeshConfig(
            h3_resolution=8,
            mast_height_m=12.0,
            min_fresnel_clearance_m=0.0,
        )
        start = h3.latlng_to_cell(40.0, 44.0, self.config.h3_resolution)
        candidates = list(h3.grid_disk(start, 2))
        self.real_cells = {}
        for h3_idx in candidates:
            lat, lon = h3.cell_to_latlng(h3_idx)
            self.real_cells[h3_idx] = H3Cell(
                h3_index=h3_idx,
                lat=lat,
                lon=lon,
                elevation=100.0,
                has_road=True,
                is_in_boundary=True,
            )
        keys = list(self.real_cells.keys())[:8]
        self.real_pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                self.real_pairs.append((keys[i], keys[j]))

    def test_batch_collects_failures_in_diagnostics(self):
        cells = {
            "a": H3Cell("a", 40.0, 44.0, 100.0, has_road=True),
            "b": H3Cell("b", 40.1, 44.1, 100.0, has_road=True),
            "c": H3Cell("c", 40.2, 44.2, 100.0, has_road=True),
        }
        pairs = [("a", "b"), ("b", "c")]

        def _fake_compute(src, dst, *_a, **_k):
            if {src, dst} == {"b", "c"}:
                raise RuntimeError("boom")
            return LOSResult(
                clearance_m=5.0,
                path_loss_db=90.0,
                distance_m=1000.0,
                is_visible=True,
            )

        diagnostics = {}
        results = compute_los_batch(
            pairs,
            cells,
            self.config,
            max_workers=2,
            min_pairs_for_parallel=1,
            compute_fn=_fake_compute,
            diagnostics=diagnostics,
        )
        self.assertIn(("a", "b"), results)
        self.assertNotIn(("b", "c"), results)
        self.assertEqual(diagnostics["pairs_requested"], 2)
        self.assertEqual(diagnostics["unique_pairs"], 2)
        self.assertEqual(diagnostics["pairs_computed"], 1)
        self.assertEqual(diagnostics["pairs_failed"], 1)

    def test_batch_strict_failures_raises(self):
        cells = {
            "a": H3Cell("a", 40.0, 44.0, 100.0, has_road=True),
            "b": H3Cell("b", 40.1, 44.1, 100.0, has_road=True),
        }
        pairs = [("a", "b")]

        def _fail(*_a, **_k):
            raise RuntimeError("fail")

        with self.assertRaises(RuntimeError):
            compute_los_batch(
                pairs,
                cells,
                self.config,
                max_workers=2,
                min_pairs_for_parallel=1,
                compute_fn=_fail,
                strict_failures=True,
            )

    def test_parallel_matches_serial_with_real_los(self):
        serial = compute_los_batch(
            self.real_pairs,
            self.real_cells,
            self.config,
            max_workers=1,
            min_pairs_for_parallel=1,
            elevation_provider=_FlatElevationProvider(),
        )
        parallel = compute_los_batch(
            self.real_pairs,
            self.real_cells,
            self.config,
            max_workers=4,
            min_pairs_for_parallel=1,
            elevation_provider=_FlatElevationProvider(),
        )
        self.assertEqual(set(serial.keys()), set(parallel.keys()))
        for key in serial:
            self.assertAlmostEqual(serial[key].clearance_m, parallel[key].clearance_m, places=6)
            self.assertAlmostEqual(serial[key].path_loss_db, parallel[key].path_loss_db, places=6)
            self.assertAlmostEqual(serial[key].distance_m, parallel[key].distance_m, places=6)
            self.assertEqual(serial[key].is_visible, parallel[key].is_visible)

    def test_parallel_is_repeatable_with_real_los(self):
        baseline = compute_los_batch(
            self.real_pairs,
            self.real_cells,
            self.config,
            max_workers=4,
            min_pairs_for_parallel=1,
            elevation_provider=_FlatElevationProvider(),
        )
        for _ in range(4):
            current = compute_los_batch(
                self.real_pairs,
                self.real_cells,
                self.config,
                max_workers=4,
                min_pairs_for_parallel=1,
                elevation_provider=_FlatElevationProvider(),
            )
            self.assertEqual(set(baseline.keys()), set(current.keys()))
            for key in baseline:
                self.assertAlmostEqual(baseline[key].clearance_m, current[key].clearance_m, places=6)
                self.assertAlmostEqual(baseline[key].path_loss_db, current[key].path_loss_db, places=6)
                self.assertAlmostEqual(baseline[key].distance_m, current[key].distance_m, places=6)
                self.assertEqual(baseline[key].is_visible, current[key].is_visible)
