"""
Regression tests for routing graph connectivity.

Verifies that build_routing_graph with routing_k_ring=2 bridges gaps between
road cell chains that are 2 H3 hops apart (which would be disconnected with k_ring=1).
"""
import networkx as nx
import h3

from mesh_calculator.network.routing import build_routing_graph
from mesh_calculator.core.grid import H3Cell
from mesh_calculator.core.config import MeshConfig


def _make_cell(h3_idx: str, elevation: float = 100.0) -> H3Cell:
    lat, lon = h3.cell_to_latlng(h3_idx)
    return H3Cell(
        h3_index=h3_idx,
        lat=lat,
        lon=lon,
        elevation=elevation,
        has_road=True,
        is_in_boundary=True,
    )


def _build_chain(start_cell: str, length: int) -> list[str]:
    """Build a chain of H3 cells by walking k=1 neighbors."""
    chain = [start_cell]
    seen = {start_cell}
    current = start_cell
    for _ in range(length - 1):
        for neighbor in h3.grid_ring(current, 1):
            if neighbor not in seen:
                chain.append(neighbor)
                seen.add(neighbor)
                current = neighbor
                break
    return chain


class TestRoutingGraphConnectivity:
    """Tests for routing graph k-ring gap bridging."""

    def _two_chains_with_gap(self, gap_k: int):
        """
        Build two chains of road cells separated by a gap of `gap_k` H3 hops.
        Returns (cells_dict, chain_a_start, chain_b_start).
        """
        # Chain A: 5 cells starting from a fixed cell
        seed = h3.latlng_to_cell(40.0, 44.0, 8)
        chain_a = _build_chain(seed, 5)

        # Find a cell that is exactly gap_k hops from the last cell of chain_a
        chain_a_end = chain_a[-1]
        gap_cells = h3.grid_ring(chain_a_end, gap_k)
        # Pick one that isn't already in chain_a
        bridge_cell = next(c for c in gap_cells if c not in chain_a)

        # Chain B: 5 cells starting from bridge_cell
        chain_b = _build_chain(bridge_cell, 5)

        cells = {}
        for idx in chain_a + chain_b:
            cells[idx] = _make_cell(idx)

        return cells, chain_a[0], chain_b[0]

    def test_k1_ring_disconnected_when_gap_is_2(self):
        """k_ring=1 cannot bridge a 2-hop gap — graph must be disconnected."""
        cells, start_a, start_b = self._two_chains_with_gap(gap_k=2)

        config = MeshConfig(h3_resolution=8, routing_k_ring=1)
        G = build_routing_graph(cells, None, config)

        components = nx.number_weakly_connected_components(G)
        assert components > 1, (
            "Expected disconnected graph with k_ring=1 and a 2-hop gap, "
            f"but got {components} component(s)"
        )
        assert not nx.has_path(G, start_a, start_b)

    def test_k2_ring_bridges_2hop_gap(self):
        """k_ring=2 bridges a 2-hop gap — graph must be connected."""
        cells, start_a, start_b = self._two_chains_with_gap(gap_k=2)

        config = MeshConfig(h3_resolution=8, routing_k_ring=2)
        G = build_routing_graph(cells, None, config)

        components = nx.number_weakly_connected_components(G)
        assert components == 1, (
            f"Expected 1 connected component with k_ring=2, got {components}"
        )
        assert nx.has_path(G, start_a, start_b)

    def test_k2_ring_path_found_across_gap(self):
        """Dijkstra finds a path across the 2-hop gap when k_ring=2."""
        from mesh_calculator.network.routing import find_road_corridor

        cells, start_a, start_b = self._two_chains_with_gap(gap_k=2)
        config = MeshConfig(h3_resolution=8, routing_k_ring=2)
        G = build_routing_graph(cells, None, config)

        corridor = find_road_corridor(start_a, start_b, G)
        assert corridor is not None, "Expected a corridor path but got None"
        assert corridor[0] == start_a
        assert corridor[-1] == start_b

    def test_default_k_ring_is_2(self):
        """MeshConfig default routing_k_ring is 2."""
        config = MeshConfig()
        assert config.routing_k_ring == 2
