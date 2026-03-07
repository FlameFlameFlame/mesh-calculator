# Testing Guide

## Unit Tests Created ✅

### Test Files
1. **[test_geometry.py](mesh_calculator/tests/test_geometry.py)** - Geometric utilities
   - Great circle distance calculation
   - H3 conversions (lat/lon ↔ H3 index)
   - H3 distance calculations
   - Neighbor finding
   - Line fraction calculations

2. **[test_physics.py](mesh_calculator/tests/test_physics.py)** - Physics calculations
   - Free-space path loss (FSPL)
   - Path loss with clear LOS
   - Path loss with obstruction (diffraction)
   - Input validation
   - Wavelength and earth radius calculations

3. **[test_cache.py](mesh_calculator/tests/test_cache.py)** - LOS caching
   - Put/get operations
   - Cache key symmetry
   - Cache misses
   - Statistics tracking
   - Cache clearing

4. **[test_integration.py](mesh_calculator/tests/test_integration.py)** - Full pipeline
   - Complete end-to-end test with synthetic data
   - Verifies all components work together

### Test Data Generator ✅

**[generate_test_data.py](mesh_calculator/tests/generate_test_data.py)** creates:
- **Boundary**: Small 10x10 km polygon
- **Roads**: 3 simple roads (2 horizontal, 1 vertical)
- **Sites**: 3 test sites with priorities
  - Site A (Priority 1) - Top left
  - Site B (Priority 1) - Top right
  - Site C (Priority 2) - Center
- **Elevation**: Flat terrain GeoTIFF (100±10 meters)
- **Configuration**: Complete test config YAML

## Running Tests

### 1. Install Dependencies

```bash
cd mesh_calculator
uv sync --group dev
```

### 2. Run Unit Tests

```bash
# All tests
uv run pytest mesh_calculator/tests/ -v

# Specific test file
uv run pytest mesh_calculator/tests/test_geometry.py -v

# With coverage
uv run pytest mesh_calculator/tests/ --cov=mesh_calculator --cov-report=html
```

Or run directly with Python:
```bash
python3 mesh_calculator/tests/test_geometry.py
python3 mesh_calculator/tests/test_physics.py
python3 mesh_calculator/tests/test_cache.py
```

### 3. Generate Test Data

```bash
python3 mesh_calculator/tests/generate_test_data.py
```

This creates `test_data/` directory with:
- boundary.geojson
- roads.geojson
- sites.geojson
- elevation.tif
- test_config.yaml

### 4. Run Integration Test

```bash
uv run pytest mesh_calculator/tests/test_integration.py -v -s
```

Or directly:
```bash
python3 mesh_calculator/tests/test_integration.py
```

This will:
1. Generate synthetic test data
2. Run complete pipeline
3. Verify towers are placed
4. Report statistics
5. Clean up temporary files

## Test Coverage

### Components Tested

✅ **Core**
- Configuration loading
- H3 grid generation
- Elevation provider
- Geometric utilities

✅ **Physics**
- Fresnel clearance
- Path loss (FSPL + diffraction)
- LOS calculations

✅ **Caching**
- Thread-safe LOS cache
- Hit/miss tracking
- Symmetry

✅ **Network**
- Routing graph construction
- Site connectivity

✅ **Optimization**
- Hierarchical connectivity
- Corridor node placement

✅ **Integration**
- End-to-end pipeline

## Expected Test Results

### Unit Tests
- **test_geometry.py**: 6 tests should pass
- **test_physics.py**: 5 tests should pass
- **test_cache.py**: 6 tests should pass

### Integration Test
Should produce output like:
```
============================================================
Running Full Pipeline Integration Test
============================================================

[1/8] Loading configuration...
  ✓ Configuration loaded

[2/8] Loading input data...
  ✓ Boundary loaded: 0.0100 sq degrees
  ✓ Roads loaded: 3 features
  ✓ Sites loaded: 3 sites
    - Priority 1: 2 sites
    - Priority 2: 1 sites

[3/8] Loading elevation data...
  ✓ Elevation provider initialized

[4/8] Generating H3 grid...
  Total cells in boundary: ~150
  Cells on roads: ~30
  Valid cells (boundary + roads): ~30
  ✓ Grid generated: ~30 cells

[5/8] Creating mesh surface...
  ✓ Mesh surface created

[6/8] Initializing LOS cache...
  ✓ LOS cache initialized

[7/8] Building routing graph...
  ✓ Routing graph: ~30 nodes, ~150 edges

[8/8] Connecting sites by priority...
  Priority 1: Connecting 2 sites (full mesh)
    [1/1] Connecting Site A ↔ Site B...
      ✓ Corridor: ~15 cells, 3-5 nodes placed

  Priority 2: Connecting 1 sites...
    Connecting Site C (P2) → Site A (P1)...
      ✓ Corridor: ~8 cells, 2-3 nodes placed

  ✓ Towers placed: 5-8

  Verification:
    - Total towers: 5-8
    - Total sites: 3

  Cache stats:
    - Entries: 50-100
    - Hit rate: 30-50%
    - Elevation cache: 100-150 entries

============================================================
Integration Test PASSED
============================================================
```

## Troubleshooting

### Missing Dependencies
```bash
uv sync --group dev
```

### Import Errors
Make sure you're in the mesh_calculator root directory:
```bash
cd /path/to/mesh_calculator
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### GDAL/Rasterio Issues
Install system dependencies:
```bash
# macOS
brew install gdal

# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# Then reinstall rasterio
uv pip install --reinstall --no-cache-dir rasterio
```

## Manual Testing

To test manually with synthetic data:

```bash
# 1. Generate test data
python3 mesh_calculator/tests/generate_test_data.py

# 2. Run optimizer
mesh-calculator --config test_data/test_config.yaml --output test_data/output/

# 3. Check outputs
ls test_data/output/
# Should see: towers.geojson, coverage.geojson, report.json
```

## Performance Benchmarks

On a typical laptop (4-8 cores):
- **Test data generation**: < 1 second
- **Unit tests**: < 5 seconds
- **Integration test**: 10-30 seconds
  - Grid generation: 1-2 seconds
  - Routing graph: 1-2 seconds
  - LOS calculations: 5-15 seconds
  - Node placement: 2-5 seconds

## Continuous Integration

To run in CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y gdal-bin libgdal-dev libspatialindex-dev

      - name: Install Python dependencies
        run: |
          uv sync --group dev

      - name: Run tests
        run: uv run pytest mesh_calculator/tests/ -v --cov=mesh_calculator

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Next Steps

- [ ] Add more edge case tests
- [ ] Add performance benchmarks
- [ ] Add test for large-scale data
- [ ] Add visualization tests (if frontend added)
- [ ] Add CLI argument tests
- [ ] Add configuration validation tests
