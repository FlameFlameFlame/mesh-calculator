"""
LOS calculation cache management.
"""
import threading
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class LOSResult:
    """
    Line-of-sight calculation result.

    Attributes:
        clearance_m: Fresnel clearance in meters
        path_loss_db: Path loss in dB
        distance_m: Distance in meters
        is_visible: Whether LOS exists (clearance > 0)
    """
    clearance_m: float
    path_loss_db: float
    distance_m: float
    is_visible: bool


class LOSCache:
    """
    Thread-safe cache for LOS calculations.

    Caches results to avoid redundant expensive Fresnel clearance calculations.
    """

    def __init__(self):
        """Initialize empty cache with thread lock."""
        self._cache: Dict[Tuple, LOSResult] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        h3_src: str,
        h3_dst: str,
        mast_height_src: float,
        mast_height_dst: float,
        frequency_hz: float
    ) -> Tuple:
        """
        Create normalized cache key (always src <= dst for symmetry).

        Args:
            h3_src, h3_dst: H3 cell indices
            mast_height_src, mast_height_dst: Mast heights
            frequency_hz: Radio frequency

        Returns:
            Normalized tuple key
        """
        if h3_src <= h3_dst:
            return (h3_src, h3_dst, mast_height_src, mast_height_dst, frequency_hz)
        else:
            return (h3_dst, h3_src, mast_height_dst, mast_height_src, frequency_hz)

    def get(
        self,
        h3_src: str,
        h3_dst: str,
        mast_height_src: float,
        mast_height_dst: float,
        frequency_hz: float
    ) -> Optional[LOSResult]:
        """
        Get cached LOS result if available.

        Args:
            h3_src, h3_dst: H3 cell indices
            mast_height_src, mast_height_dst: Mast heights
            frequency_hz: Radio frequency

        Returns:
            LOSResult if cached, None otherwise
        """
        key = self._make_key(h3_src, h3_dst, mast_height_src, mast_height_dst, frequency_hz)

        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put(
        self,
        h3_src: str,
        h3_dst: str,
        mast_height_src: float,
        mast_height_dst: float,
        frequency_hz: float,
        result: LOSResult
    ):
        """
        Store LOS result in cache.

        Args:
            h3_src, h3_dst: H3 cell indices
            mast_height_src, mast_height_dst: Mast heights
            frequency_hz: Radio frequency
            result: LOSResult to cache
        """
        key = self._make_key(h3_src, h3_dst, mast_height_src, mast_height_dst, frequency_hz)

        with self._lock:
            self._cache[key] = result

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, and hit rate
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'total_queries': total,
                'hit_rate': hit_rate,
                'memory_mb': len(self._cache) * 64 / (1024 * 1024),  # Approx
            }

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
