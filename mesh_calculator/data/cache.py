"""
LOS calculation cache management.
"""
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
        is_visible: Whether link is accepted by LOS policy.
    """
    clearance_m: float
    path_loss_db: float
    distance_m: float
    is_visible: bool


class LOSCache:
    """
    Cache for LOS calculations.

    Caches results to avoid redundant expensive Fresnel clearance calculations.
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[Tuple, LOSResult] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        h3_src: str,
        h3_dst: str,
        mast_height_src: float,
        mast_height_dst: float,
        frequency_hz: float,
        tx_power_mw: float,
        antenna_gain_dbi: float,
        receiver_sensitivity_dbm: float,
        min_fresnel_clearance_m: Optional[float],
        los_dense_sample_step_m: float,
        los_dense_max_samples: int,
        los_verification_mode: str,
        src_lat: Optional[float],
        src_lon: Optional[float],
        dst_lat: Optional[float],
        dst_lon: Optional[float],
    ) -> Tuple:
        """
        Create normalized cache key (always src <= dst for symmetry).

        Args:
            h3_src, h3_dst: H3 cell indices
            mast_height_src, mast_height_dst: Mast heights
            frequency_hz: Radio frequency
            tx_power_mw: TX power in mW
            antenna_gain_dbi: Antenna gain in dBi
            receiver_sensitivity_dbm: Receiver sensitivity in dBm
            min_fresnel_clearance_m: Optional policy threshold
            los_dense_sample_step_m: Dense-profile sample interval
            los_dense_max_samples: Dense-profile sample cap
            los_verification_mode: LOS verification mode sentinel

        Returns:
            Normalized tuple key
        """
        if h3_src <= h3_dst:
            return (
                h3_src, h3_dst, mast_height_src, mast_height_dst, frequency_hz,
                tx_power_mw, antenna_gain_dbi, receiver_sensitivity_dbm,
                min_fresnel_clearance_m,
                los_dense_sample_step_m, los_dense_max_samples, los_verification_mode,
                round(src_lat, 6) if src_lat is not None else None,
                round(src_lon, 6) if src_lon is not None else None,
                round(dst_lat, 6) if dst_lat is not None else None,
                round(dst_lon, 6) if dst_lon is not None else None,
            )
        else:
            return (
                h3_dst, h3_src, mast_height_dst, mast_height_src, frequency_hz,
                tx_power_mw, antenna_gain_dbi, receiver_sensitivity_dbm,
                min_fresnel_clearance_m,
                los_dense_sample_step_m, los_dense_max_samples, los_verification_mode,
                round(dst_lat, 6) if dst_lat is not None else None,
                round(dst_lon, 6) if dst_lon is not None else None,
                round(src_lat, 6) if src_lat is not None else None,
                round(src_lon, 6) if src_lon is not None else None,
            )

    def get(
        self,
        h3_src: str,
        h3_dst: str,
        mast_height_src: float,
        mast_height_dst: float,
        frequency_hz: float,
        tx_power_mw: float = 500.0,
        antenna_gain_dbi: float = 2.0,
        receiver_sensitivity_dbm: float = -137.0,
        min_fresnel_clearance_m: Optional[float] = None,
        los_dense_sample_step_m: float = 50.0,
        los_dense_max_samples: int = 400,
        los_verification_mode: str = "hybrid_accept_verify",
        src_lat: Optional[float] = None,
        src_lon: Optional[float] = None,
        dst_lat: Optional[float] = None,
        dst_lon: Optional[float] = None,
    ) -> Optional[LOSResult]:
        """
        Get cached LOS result if available.

        Args:
            h3_src, h3_dst: H3 cell indices
            mast_height_src, mast_height_dst: Mast heights
            frequency_hz: Radio frequency
            tx_power_mw: TX power in mW
            antenna_gain_dbi: Antenna gain in dBi
            receiver_sensitivity_dbm: Receiver sensitivity in dBm
            min_fresnel_clearance_m: Optional policy threshold
            los_dense_sample_step_m: Dense-profile sample interval
            los_dense_max_samples: Dense-profile sample cap
            los_verification_mode: LOS verification mode sentinel

        Returns:
            LOSResult if cached, None otherwise
        """
        key = self._make_key(
            h3_src, h3_dst, mast_height_src, mast_height_dst, frequency_hz,
            tx_power_mw, antenna_gain_dbi, receiver_sensitivity_dbm,
            min_fresnel_clearance_m,
            los_dense_sample_step_m, los_dense_max_samples, los_verification_mode,
            src_lat, src_lon, dst_lat, dst_lon,
        )

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
        result: LOSResult,
        tx_power_mw: float = 500.0,
        antenna_gain_dbi: float = 2.0,
        receiver_sensitivity_dbm: float = -137.0,
        min_fresnel_clearance_m: Optional[float] = None,
        los_dense_sample_step_m: float = 50.0,
        los_dense_max_samples: int = 400,
        los_verification_mode: str = "hybrid_accept_verify",
        src_lat: Optional[float] = None,
        src_lon: Optional[float] = None,
        dst_lat: Optional[float] = None,
        dst_lon: Optional[float] = None,
    ):
        """
        Store LOS result in cache.

        Args:
            h3_src, h3_dst: H3 cell indices
            mast_height_src, mast_height_dst: Mast heights
            frequency_hz: Radio frequency
            result: LOSResult to cache
            tx_power_mw: TX power in mW
            antenna_gain_dbi: Antenna gain in dBi
            receiver_sensitivity_dbm: Receiver sensitivity in dBm
            min_fresnel_clearance_m: Optional policy threshold
            los_dense_sample_step_m: Dense-profile sample interval
            los_dense_max_samples: Dense-profile sample cap
            los_verification_mode: LOS verification mode sentinel
        """
        key = self._make_key(
            h3_src, h3_dst, mast_height_src, mast_height_dst, frequency_hz,
            tx_power_mw, antenna_gain_dbi, receiver_sensitivity_dbm,
            min_fresnel_clearance_m,
            los_dense_sample_step_m, los_dense_max_samples, los_verification_mode,
            src_lat, src_lon, dst_lat, dst_lon,
        )

        self._cache[key] = result

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, and hit rate
        """
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
        self._cache.clear()
        self._hits = 0
        self._misses = 0
