"""
Elevation data handling with caching for mesh calculator.
"""
from typing import Optional, Tuple
import rasterio
from rasterio.transform import rowcol
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ElevationProvider:
    """
    Provides elevation data from GeoTIFF raster with caching.

    Attributes:
        dataset: Rasterio dataset handle
        transform: Affine transform for coordinate conversion
        _cache: Dictionary cache for elevation lookups
    """

    def __init__(self, tif_path: str):
        """
        Initialize elevation provider from GeoTIFF file.

        Args:
            tif_path: Path to GeoTIFF elevation file
        """
        self.dataset = None
        self.dataset = rasterio.open(tif_path)
        self.transform = self.dataset.transform
        self._cache = {}
        self._data = None  # Lazy-load full array if needed

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Get elevation at a specific latitude/longitude with caching.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Elevation in meters
        """
        # Round to 6 decimal places for cache key (~10cm precision)
        cache_key = (round(lat, 6), round(lon, 6))

        if cache_key not in self._cache:
            # Convert lat/lon to row/col
            try:
                row, col = rowcol(self.transform, lon, lat)

                # Check bounds
                if (0 <= row < self.dataset.height and
                    0 <= col < self.dataset.width):
                    # Read single pixel value
                    window = rasterio.windows.Window(col, row, 1, 1)
                    elevation = self.dataset.read(1, window=window)[0, 0]

                    # Handle nodata values
                    if self.dataset.nodata is not None and elevation == self.dataset.nodata:
                        elevation = 0.0
                else:
                    # Out of bounds - return 0
                    elevation = 0.0

                self._cache[cache_key] = float(elevation)

            except Exception as e:
                logger.warning("Failed to get elevation", lat=lat, lon=lon, error=str(e))
                self._cache[cache_key] = 0.0

        return self._cache[cache_key]

    def get_elevation_bulk(self, coords: list[Tuple[float, float]]) -> np.ndarray:
        """
        Get elevations for multiple coordinates efficiently.

        Args:
            coords: List of (lat, lon) tuples

        Returns:
            NumPy array of elevations
        """
        elevations = np.zeros(len(coords))

        for i, (lat, lon) in enumerate(coords):
            elevations[i] = self.get_elevation(lat, lon)

        return elevations

    def cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and other stats
        """
        return {
            'cache_size': len(self._cache),
            'cache_memory_mb': len(self._cache) * 24 / (1024 * 1024),  # Approx
        }

    def clear_cache(self):
        """Clear the elevation cache."""
        self._cache.clear()

    def close(self):
        """Close the rasterio dataset."""
        if getattr(self, 'dataset', None):
            self.dataset.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure dataset is closed."""
        self.close()
