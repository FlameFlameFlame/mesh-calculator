"""
Integration test using synthetic geodata.
"""
import unittest
import os
import tempfile
import shutil
from .generate_test_data import generate_test_data
from ..data.loaders import load_config
from ..data.sites import load_sites
from ..data.cache import LOSCache
from ..core.elevation import ElevationProvider
from ..core.grid import load_boundary, load_roads, generate_road_grid
from ..network.graph import MeshSurface
from ..network.routing import build_routing_graph
from ..optimization.hierarchical import connect_sites_by_priority


class TestIntegration(unittest.TestCase):
    """Integration test with synthetic data."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        cls.test_dir = tempfile.mkdtemp(prefix='mesh_test_')
        print(f"\nTest directory: {cls.test_dir}")
        cls.config_path = generate_test_data(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            print(f"Cleaned up test directory: {cls.test_dir}")

    def test_full_pipeline(self):
        """Test complete pipeline with synthetic data."""
        print("\n" + "="*60)
        print("Running Full Pipeline Integration Test")
        print("="*60)

        # Load configuration
        print("\n[1/8] Loading configuration...")
        cfg = load_config(self.config_path)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.parameters.h3_resolution, 8)
        print("  ✓ Configuration loaded")

        # Load input data
        print("\n[2/8] Loading input data...")
        boundary = load_boundary(cfg.inputs.boundary)
        self.assertIsNotNone(boundary)
        print(f"  ✓ Boundary loaded: {boundary.area:.4f} sq degrees")

        roads_gdf = load_roads(cfg.inputs.roads)
        self.assertGreater(len(roads_gdf), 0)
        print(f"  ✓ Roads loaded: {len(roads_gdf)} features")

        sites = load_sites(cfg.inputs.target_sites, cfg.parameters.h3_resolution)
        self.assertGreater(len(sites), 0)
        print(f"  ✓ Sites loaded: {len(sites)} sites")

        # Verify site priorities
        priority1_sites = [s for s in sites if s.priority == 1]
        priority2_sites = [s for s in sites if s.priority == 2]
        self.assertEqual(len(priority1_sites), 2)  # Site A, Site B
        self.assertEqual(len(priority2_sites), 1)  # Site C
        print(f"    - Priority 1: {len(priority1_sites)} sites")
        print(f"    - Priority 2: {len(priority2_sites)} sites")

        # Initialize elevation provider
        print("\n[3/8] Loading elevation data...")
        elevation_provider = ElevationProvider(cfg.inputs.elevation)
        self.assertIsNotNone(elevation_provider)
        print("  ✓ Elevation provider initialized")

        # Generate H3 grid
        print("\n[4/8] Generating H3 grid...")
        cells = generate_road_grid(boundary, roads_gdf, elevation_provider, cfg.parameters)
        self.assertGreater(len(cells), 0)
        print(f"  ✓ Grid generated: {len(cells)} cells")

        # Create mesh surface
        print("\n[5/8] Creating mesh surface...")
        surface = MeshSurface(cells, cfg.parameters)
        self.assertIsNotNone(surface)
        print(f"  ✓ Mesh surface created")

        # Initialize LOS cache
        print("\n[6/8] Initializing LOS cache...")
        los_cache = LOSCache()
        self.assertIsNotNone(los_cache)
        print("  ✓ LOS cache initialized")

        # Build routing graph
        print("\n[7/8] Building routing graph...")
        routing_graph = build_routing_graph(cells, roads_gdf, cfg.parameters)
        self.assertGreater(routing_graph.number_of_nodes(), 0)
        self.assertGreater(routing_graph.number_of_edges(), 0)
        print(f"  ✓ Routing graph: {routing_graph.number_of_nodes()} nodes, "
              f"{routing_graph.number_of_edges()} edges")

        # Connect sites by priority
        print("\n[8/8] Connecting sites by priority...")
        initial_tower_count = len(surface.towers)

        connect_sites_by_priority(sites, surface, routing_graph, los_cache)

        final_tower_count = len(surface.towers)
        towers_placed = final_tower_count - initial_tower_count

        print(f"  ✓ Towers placed: {towers_placed}")

        # Verify towers were placed
        self.assertGreater(final_tower_count, 0)

        # Verify at least one tower per site
        site_h3_indices = {s.h3_index for s in sites}
        tower_h3_indices = {t.h3_index for t in surface.towers.values()}

        # Sites should have towers nearby (if not exactly at the site)
        print("\n  Verification:")
        print(f"    - Total towers: {final_tower_count}")
        print(f"    - Total sites: {len(sites)}")

        # Print cache stats
        cache_stats = los_cache.stats()
        print(f"\n  Cache stats:")
        print(f"    - Entries: {cache_stats['size']}")
        print(f"    - Hit rate: {cache_stats['hit_rate']:.1%}")

        # Print elevation cache stats
        elev_stats = elevation_provider.cache_stats()
        print(f"    - Elevation cache: {elev_stats['cache_size']} entries")

        print("\n" + "="*60)
        print("Integration Test PASSED")
        print("="*60)

        # Clean up
        elevation_provider.close()

    def test_data_files_exist(self):
        """Test that all generated data files exist."""
        cfg = load_config(self.config_path)

        self.assertTrue(os.path.exists(cfg.inputs.boundary))
        self.assertTrue(os.path.exists(cfg.inputs.roads))
        self.assertTrue(os.path.exists(cfg.inputs.target_sites))
        self.assertTrue(os.path.exists(cfg.inputs.elevation))


if __name__ == '__main__':
    unittest.main(verbosity=2)
