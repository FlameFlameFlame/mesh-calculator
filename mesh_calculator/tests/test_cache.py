"""
Unit tests for caching.
"""
import unittest
from ..data.cache import LOSCache, LOSResult


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


if __name__ == '__main__':
    unittest.main()
