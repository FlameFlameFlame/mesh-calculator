"""Benchmark dense Fresnel profile sampling before/after vectorization."""
import time
import numpy as np
from mesh_calculator.core.elevation import ElevationProvider

TIF = "../mesh-generator/save/elevation.tif"


def benchmark():
    provider = ElevationProvider(TIF)
    bounds = provider.dataset.bounds

    # Simulate what _dense_profile_samples does: sample N points along a line
    np.random.seed(42)
    n_lines = 200
    samples_per_line = 100  # typical for ~1km link at 10m step

    # Generate random line endpoints within bounds
    src_lats = np.random.uniform(bounds.bottom, bounds.top, n_lines)
    src_lons = np.random.uniform(bounds.left, bounds.right, n_lines)
    dst_lats = np.random.uniform(bounds.bottom, bounds.top, n_lines)
    dst_lons = np.random.uniform(bounds.left, bounds.right, n_lines)

    total_points = n_lines * samples_per_line

    # Scalar path (old way): call get_elevation_bilinear per point
    t0 = time.perf_counter()
    for i in range(n_lines):
        fracs = np.arange(1, samples_per_line) / samples_per_line
        for frac in fracs:
            lat = src_lats[i] + (dst_lats[i] - src_lats[i]) * frac
            lon = src_lons[i] + (dst_lons[i] - src_lons[i]) * frac
            provider.get_elevation_bilinear(lat, lon)
    t1 = time.perf_counter()
    print(f"Scalar bilinear ({n_lines} lines × {samples_per_line} pts): {t1-t0:.3f}s")

    # Bulk path (new way): call get_elevation_bilinear_bulk per line
    t0 = time.perf_counter()
    for i in range(n_lines):
        fracs = np.arange(1, samples_per_line) / samples_per_line
        lats = src_lats[i] + (dst_lats[i] - src_lats[i]) * fracs
        lons = src_lons[i] + (dst_lons[i] - src_lons[i]) * fracs
        coords = list(zip(lats, lons))
        provider.get_elevation_bilinear_bulk(coords)
    t1 = time.perf_counter()
    print(f"Bulk bilinear   ({n_lines} lines × {samples_per_line} pts): {t1-t0:.3f}s")

    # Mega-bulk: all points at once
    all_coords = []
    for i in range(n_lines):
        fracs = np.arange(1, samples_per_line) / samples_per_line
        lats = src_lats[i] + (dst_lats[i] - src_lats[i]) * fracs
        lons = src_lons[i] + (dst_lons[i] - src_lons[i]) * fracs
        all_coords.extend(zip(lats, lons))

    t0 = time.perf_counter()
    provider.get_elevation_bilinear_bulk(all_coords)
    t1 = time.perf_counter()
    print(f"Single bulk call ({total_points} pts total):          {t1-t0:.3f}s")

    provider.close()


if __name__ == "__main__":
    benchmark()
