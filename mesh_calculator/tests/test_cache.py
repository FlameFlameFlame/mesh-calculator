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


if __name__ == '__main__':
    unittest.main()
