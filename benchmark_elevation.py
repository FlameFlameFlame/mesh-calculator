"""Benchmark elevation lookups before/after optimization."""
import time
import numpy as np
from mesh_calculator.core.elevation import ElevationProvider

TIF = "test_data/elevation.tif"

def benchmark():
    provider = ElevationProvider(TIF)
    
    # Get bounds from the dataset to generate valid coordinates
    bounds = provider.dataset.bounds
    n_points = 50_000
    
    np.random.seed(42)
    lats = np.random.uniform(bounds.bottom, bounds.top, n_points)
    lons = np.random.uniform(bounds.left, bounds.right, n_points)
    coords = list(zip(lats, lons))
    
    # Benchmark get_elevation in a loop
    provider.clear_cache()
    t0 = time.perf_counter()
    for lat, lon in coords:
        provider.get_elevation(lat, lon)
    t1 = time.perf_counter()
    print(f"get_elevation x{n_points}: {t1-t0:.3f}s ({(t1-t0)/n_points*1000:.3f}ms/call)")
    
    # Benchmark get_elevation_bulk
    provider.clear_cache()
    t0 = time.perf_counter()
    result = provider.get_elevation_bulk(coords)
    t1 = time.perf_counter()
    print(f"get_elevation_bulk x{n_points}: {t1-t0:.3f}s")
    
    # Benchmark get_elevation_bilinear in a loop
    provider.clear_cache()
    t0 = time.perf_counter()
    for lat, lon in coords[:10_000]:
        provider.get_elevation_bilinear(lat, lon)
    t1 = time.perf_counter()
    print(f"get_elevation_bilinear x10000: {t1-t0:.3f}s ({(t1-t0)/10000*1000:.3f}ms/call)")
    
    provider.close()

if __name__ == "__main__":
    benchmark()
