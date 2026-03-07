"""
Tests for corridor placement algorithm (MaxMin DP).

The legacy _walk_segment algorithm has been replaced with a MaxMin Bottleneck
Path DP that finds the globally optimal tower chain maximising minimum Fresnel
clearance.  These tests verify the DP's core behaviours:

  - Skips cells that have no LOS to the current tower
  - Finds the furthest/best visible cell (not just the immediate neighbour)
  - Always includes both corridor endpoints
  - Falls back to peak-based placement when no feasible chain exists
  - Short corridors (2 cells) are handled correctly
"""
import unittest
from unittest.mock import patch

from ..core.config import MeshConfig
from ..core.grid import H3Cell
from ..data.cache import LOSResult
from ..network.graph import MeshSurface
from ..optimization.corridor import place_nodes_along_corridor
from ..optimization import corridor as corridor_mod


def make_cells(n, elevation=100.0):
    """Create n fake H3 cells labeled cell_0..cell_n-1."""
    cells = {}
    for i in range(n):
        h3_idx = f"cell_{i}"
        cells[h3_idx] = H3Cell(
            h3_index=h3_idx,
            lat=40.0 + i * 0.005,
            lon=44.0 + i * 0.005,
            elevation=elevation,
            has_road=True,
        )
    return cells


def make_corridor(n):
    """Create a corridor: ['cell_0', 'cell_1', ..., 'cell_n-1']."""
    return [f"cell_{i}" for i in range(n)]


def make_compute_los_func(los_pairs, default_visible=False, clearance=10.0):
    """
    Return a compute_los mock that reads visibility from los_pairs.

    Keys are (src, dst) tuples; value is True/False.  Pairs not in the dict
    default to default_visible.  Always returns a LOSResult.
    """
    def _compute_los(src, dst, cells_arg, config, cache=None,
                     elevation_provider=None, corridor_cells=None):
        is_vis = los_pairs.get((src, dst), default_visible)
        return LOSResult(
            clearance_m=clearance if is_vis else -999.0,
            path_loss_db=50.0,
            distance_m=1000.0,
            is_visible=is_vis,
        )
    return _compute_los


# ---------- DP: non-adjacent LOS ----------

class TestDPNonAdjacentLOS(unittest.TestCase):
    """DP correctly connects through the best visible cell, skipping others."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_skips_nonvisible_finds_further_visible(
        self, mock_distance, mock_compute_los
    ):
        """
        When intermediate cells have no LOS but a further cell does,
        the DP should select the further cell, not fall back to adjacent.

        Corridor: [0, 1, 2, 3, 4]
        LOS: 0→3 only (not 0→1, 0→2, 0→4); 3→4.
        Expected: [cell_0, cell_3, cell_4]  (cell_1 and cell_2 absent)
        """
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config, elevation_provider=object())
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): False,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): True,
            ("cell_0", "cell_4"): False,
            ("cell_3", "cell_4"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertNotIn("cell_1", nodes,
            "cell_1 has no LOS from cell_0 and should be absent")
        self.assertNotIn("cell_2", nodes,
            "cell_2 has no LOS from cell_0 and should be absent")
        self.assertIn("cell_3", nodes,
            "cell_3 has LOS from cell_0 and should be selected")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_immediate_neighbour_chosen_when_only_option(
        self, mock_distance, mock_compute_los
    ):
        """
        When the immediate neighbour is the only cell with LOS, it is chosen.

        Corridor: [0, 1, 2, 3]
        LOS: 0→1 only; 1→3.
        """
        corridor = make_corridor(4)
        cells = make_cells(4)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): False,
            ("cell_1", "cell_3"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_1", nodes,
            "cell_1 is the only cell with LOS from cell_0 and must be chosen")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_finds_peak_past_valley(self, mock_distance, mock_compute_los):
        """
        DP should find a peak cell past a valley (no-LOS cell).

        Corridor: [0, 1, 2, 3, 4, 5]
        LOS: 0→1, 0→2, 0→4 (hilltop, past valley 3); 4→5.
        Expected: cell_4 chosen, not cell_2 (cell_4 is further and visible).
        """
        corridor = make_corridor(6)
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): True,
            ("cell_0", "cell_3"): False,   # valley
            ("cell_0", "cell_4"): True,    # hilltop
            ("cell_0", "cell_5"): False,
            ("cell_4", "cell_5"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_4", nodes,
            "Hilltop cell_4 should be selected; it has LOS from cell_0")
        self.assertNotIn("cell_2", nodes,
            "cell_2 should be skipped — cell_4 is a further, visible relay")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_scans_through_multiple_gaps(self, mock_distance, mock_compute_los):
        """
        DP finds the best relay even when several cells between them lack LOS.

        Corridor: [0..7]; 0→1, 0→4, 0→6 (furthest); 6→7.
        Expected: cell_6 chosen over cell_4.
        """
        corridor = make_corridor(8)
        cells = make_cells(8)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0

        mock_compute_los.side_effect = make_compute_los_func({
            ("cell_0", "cell_1"): True,
            ("cell_0", "cell_2"): False,
            ("cell_0", "cell_3"): False,
            ("cell_0", "cell_4"): True,
            ("cell_0", "cell_5"): False,
            ("cell_0", "cell_6"): True,    # furthest visible from cell_0
            ("cell_0", "cell_7"): False,
            ("cell_6", "cell_7"): True,
        })

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_6", nodes,
            "cell_6 (furthest visible from cell_0) should be selected")
        self.assertNotIn("cell_4", nodes,
            "cell_4 should be skipped — cell_6 provides a better relay")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_direct_los_skips_intermediates(
        self, mock_distance, mock_compute_los
    ):
        """
        When all cells have LOS to each other, the DP picks only endpoints
        (max clearance path uses fewest hops).
        """
        corridor = make_corridor(6)
        cells = make_cells(6)
        surface = MeshSurface(cells, self.config)

        def distance_fn(src, dst):
            si = int(src.split("_")[1])
            di = int(dst.split("_")[1])
            return abs(di - si) * 20000.0

        mock_distance.side_effect = distance_fn
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], corridor[0], "First endpoint always included")
        self.assertEqual(nodes[-1], corridor[-1], "Last endpoint always included")


# ---------- DP: endpoints ----------

class TestEndpointHandling(unittest.TestCase):
    """Both corridor endpoints are always included."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_endpoints_included_all_los_clear(
        self, mock_distance, mock_compute_los
    ):
        """When all LOS is clear, both endpoints are in the result."""
        corridor = make_corridor(5)
        cells = make_cells(5)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], "cell_0")
        self.assertEqual(nodes[-1], "cell_4")

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_short_corridor_both_cells_returned(
        self, mock_distance, mock_compute_los
    ):
        """A 2-cell corridor returns exactly both cells."""
        corridor = make_corridor(2)
        cells = make_cells(2)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0, path_loss_db=50.0,
            distance_m=1000.0, is_visible=True,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes, ["cell_0", "cell_1"])

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_fallback_includes_endpoints_when_no_los(
        self, mock_distance, mock_compute_los
    ):
        """
        When no feasible chain exists (all LOS blocked), the peak-based
        fallback still includes both corridor endpoints.
        """
        corridor = make_corridor(8)
        cells = make_cells(8)
        surface = MeshSurface(cells, self.config)
        mock_distance.return_value = 1000.0
        mock_compute_los.return_value = LOSResult(
            clearance_m=-999.0, path_loss_db=999.0,
            distance_m=1000.0, is_visible=False,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertIn("cell_0", nodes, "Start endpoint must always be present")
        self.assertIn("cell_7", nodes, "End endpoint must always be present")


class TestDPGapRepairBeforeFallback(unittest.TestCase):
    """DP should run local broken-gap repair before widening fallback attempts."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=6,
            road_buffer_m=100.0,
            gap_repair_rounds=2,
            fallback_initial_search_radius_ladder_m=[300.0],
            gap_repair_search_radius_ladder_m=[0.0, 200.0],
        )

    @patch('mesh_calculator.optimization.corridor.h3.grid_disk')
    @patch('mesh_calculator.optimization.corridor._repair_broken_gaps')
    @patch('mesh_calculator.optimization.corridor._dp_place_towers_with_meta')
    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_repair_runs_before_fallback_widening(
        self,
        mock_distance,
        mock_compute_los,
        mock_dp,
        mock_repair,
        mock_grid_disk,
    ):
        corridor = make_corridor(3)
        cells = make_cells(3)
        surface = MeshSurface(cells, self.config, elevation_provider=object())

        mock_grid_disk.side_effect = lambda cell, ring: [cell]
        mock_distance.return_value = 1000.0
        mock_dp.side_effect = lambda segment, *_a, **_k: ([segment[0], segment[-1]], 2)
        mock_compute_los.return_value = LOSResult(
            clearance_m=-999.0, path_loss_db=999.0,
            distance_m=1000.0, is_visible=False,
        )

        repair_calls = []

        def _repair_passthrough(
            chain, _corridor, _corridor_pos, *,
            search_radius_m, repair_round, attempt_id,
            surface, cache, user_budget, node_meta=None, los_max_workers=None,
        ):
            repair_calls.append((int(attempt_id), float(search_radius_m)))
            return chain

        mock_repair.side_effect = _repair_passthrough

        place_nodes_along_corridor(corridor, surface)

        self.assertEqual(
            repair_calls,
            [(0, 100.0), (1, 100.0)],
            "Run broken-gap wiggle on initial attempt first, then fallback attempt(s)",
        )
        self.assertEqual(mock_dp.call_count, 2, "Initial + one widened fallback attempt expected")

    @patch('mesh_calculator.optimization.corridor.h3.grid_disk')
    @patch('mesh_calculator.optimization.corridor._repair_broken_gaps')
    @patch('mesh_calculator.optimization.corridor._dp_place_towers_with_meta')
    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_successful_repair_skips_fallback_attempts(
        self,
        mock_distance,
        mock_compute_los,
        mock_dp,
        mock_repair,
        mock_grid_disk,
    ):
        corridor = make_corridor(3)
        cells = make_cells(3)
        surface = MeshSurface(cells, self.config)

        mock_grid_disk.side_effect = lambda cell, ring: [cell]
        mock_distance.return_value = 1000.0
        mock_dp.return_value = (["cell_0", "cell_2"], 2)

        def _los(src, dst, *_a, **_k):
            visible = (src, dst) in {("cell_0", "cell_1"), ("cell_1", "cell_2")}
            return LOSResult(
                clearance_m=10.0 if visible else -999.0,
                path_loss_db=50.0 if visible else 999.0,
                distance_m=1000.0,
                is_visible=visible,
            )

        mock_compute_los.side_effect = _los
        mock_repair.side_effect = lambda chain, *_a, **_k: ["cell_0", "cell_1", "cell_2"]

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes, ["cell_0", "cell_1", "cell_2"])
        self.assertEqual(mock_repair.call_count, 1, "Gap repair should run in initial attempt")
        self.assertEqual(mock_dp.call_count, 1, "Fallback attempts should be skipped after repair success")


class TestDPBufferCandidateMaterialization(unittest.TestCase):
    """Buffer candidate scoring should handle pending (not-yet-materialized) cells."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=6,
            road_buffer_m=100.0,
            gap_repair_rounds=0,
        )

    @patch('mesh_calculator.optimization.corridor._adaptive_cells_within_radius')
    @patch('mesh_calculator.optimization.corridor._cell_profile')
    @patch('mesh_calculator.optimization.corridor._dp_place_towers_with_meta')
    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_pending_buffer_cell_does_not_keyerror(
        self,
        mock_distance,
        mock_compute_los,
        mock_dp,
        mock_cell_profile,
        mock_adaptive_cells,
    ):
        corridor = make_corridor(3)
        cells = make_cells(3)
        surface = MeshSurface(cells, self.config, elevation_provider=object())
        pending_cell = "892c0547577ffff"

        mock_distance.return_value = 1000.0
        mock_adaptive_cells.return_value = {pending_cell}
        mock_cell_profile.return_value = (120.0, 40.1234, 44.5678)
        mock_dp.side_effect = lambda segment, *_a, **_k: ([segment[0], segment[-1]], 2)
        mock_compute_los.return_value = LOSResult(
            clearance_m=10.0,
            path_loss_db=50.0,
            distance_m=1000.0,
            is_visible=True,
        )

        nodes = place_nodes_along_corridor(corridor, surface)

        self.assertEqual(nodes[0], "cell_0")
        self.assertTrue(nodes, "Placement should return a non-empty chain")
        self.assertIn(pending_cell, surface.cells, "Pending buffer cell should be materialized")


class TestDPParallelDeterminism(unittest.TestCase):
    """Parallel and serial DP runs should produce identical chains."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_parallel_and_serial_dp_match(self, mock_distance, mock_compute_los):
        corridor = make_corridor(6)
        los_pairs = {
            ("cell_0", "cell_2"): True,
            ("cell_0", "cell_4"): True,
            ("cell_2", "cell_5"): True,
            ("cell_4", "cell_5"): True,
        }
        mock_distance.return_value = 1000.0
        mock_compute_los.side_effect = make_compute_los_func(
            los_pairs, default_visible=False, clearance=8.0
        )

        serial_surface = MeshSurface(make_cells(6), self.config)
        parallel_surface = MeshSurface(make_cells(6), self.config)

        serial_nodes = place_nodes_along_corridor(
            corridor, serial_surface, los_max_workers=1
        )
        parallel_nodes = place_nodes_along_corridor(
            corridor, parallel_surface, los_max_workers=4
        )

        self.assertEqual(serial_nodes, parallel_nodes)


class TestCorridorBatchAdoption(unittest.TestCase):
    """Corridor internals should route LOS work through compute_los_batch."""

    def setUp(self):
        self.config = MeshConfig(
            mast_height_m=10.0,
            max_towers_per_route=100,
        )

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.optimization.corridor.compute_los_batch')
    @patch('mesh_calculator.core.geometry.h3_distance')
    def test_dp_uses_batch_api(self, mock_distance, mock_batch, mock_compute_los):
        cells = make_cells(4)
        surface = MeshSurface(cells, self.config)
        corridor = make_corridor(4)

        mock_distance.return_value = 1000.0

        def _batch_side_effect(pairs, *_args, **_kwargs):
            out = {}
            for src, dst in pairs:
                visible = (src, dst) in {("cell_0", "cell_2"), ("cell_2", "cell_3")}
                out[(src, dst)] = LOSResult(
                    clearance_m=10.0 if visible else -1.0,
                    path_loss_db=100.0,
                    distance_m=1000.0,
                    is_visible=visible,
                )
            return out

        mock_batch.side_effect = _batch_side_effect
        result = corridor_mod._dp_place_towers_with_meta(
            corridor, surface, None, 4, los_max_workers=2
        )

        self.assertIsNotNone(result)
        self.assertEqual(result[0], ["cell_0", "cell_2", "cell_3"])
        self.assertGreater(mock_batch.call_count, 0)
        mock_compute_los.assert_not_called()

    @patch('mesh_calculator.optimization.corridor.compute_los')
    @patch('mesh_calculator.optimization.corridor.compute_los_batch')
    def test_find_broken_gaps_uses_batch_api(self, mock_batch, mock_compute_los):
        cells = make_cells(4)
        surface = MeshSurface(cells, self.config)
        chain = make_corridor(4)

        def _batch_side_effect(pairs, *_args, **_kwargs):
            out = {}
            for src, dst in pairs:
                visible = (src, dst) != ("cell_1", "cell_2")
                out[(src, dst)] = LOSResult(
                    clearance_m=5.0 if visible else -5.0,
                    path_loss_db=90.0,
                    distance_m=1000.0,
                    is_visible=visible,
                )
            return out

        mock_batch.side_effect = _batch_side_effect
        broken = corridor_mod._find_broken_gaps(
            chain, surface, None, los_max_workers=3
        )

        self.assertEqual(broken, [1])
        mock_batch.assert_called_once()
        mock_compute_los.assert_not_called()

    @patch('mesh_calculator.optimization.corridor.compute_los_batch')
    def test_wire_corridor_edges_uses_batch_api(self, mock_batch):
        cells = make_cells(3)
        surface = MeshSurface(cells, self.config)
        surface.place_tower("cell_0", source="test")
        surface.place_tower("cell_1", source="test")
        surface.place_tower("cell_2", source="test")

        def _batch_side_effect(pairs, *_args, **_kwargs):
            return {
                pair: LOSResult(
                    clearance_m=7.0,
                    path_loss_db=80.0,
                    distance_m=500.0,
                    is_visible=True,
                )
                for pair in pairs
            }

        mock_batch.side_effect = _batch_side_effect
        corridor_mod.wire_corridor_edges(
            ["cell_0", "cell_1", "cell_2"],
            make_corridor(3),
            surface,
            cache=None,
            los_max_workers=2,
        )

        self.assertEqual(mock_batch.call_count, 1)
        self.assertEqual(surface.visibility_graph.edge_count(), 2)


if __name__ == '__main__':
    unittest.main()
