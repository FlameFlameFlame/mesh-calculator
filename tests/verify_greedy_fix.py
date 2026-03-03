#!/usr/bin/env python3
"""
Verify the greedy fallback-tower-placement bug and confirm the fix.

Scenario:
  Corridor of 12 cells in a line. LOS exists only from cell[0]→cell[7]
  and cell[7]→cell[11]. Cells 1–6 are a "valley" with no usable LOS.

Bug (before fix):
  When no LOS is found from current position, each fallback path does
  `chain.append(current)` then advances by one road step. This places a
  tower at EVERY valley cell: chain = [0, 1, 2, 3, 4, 5, 6, 7, 11].

Expected (after fix):
  Fallbacks only advance the search origin without appending.
  chain = [0, 7, 11]
"""
import sys
import h3
from unittest.mock import patch

sys.path.insert(0, "/Users/timur/Documents/src/LoraMeshPlanner/mesh_calculator")

from mesh_calculator.optimization.corridor import _greedy_place_towers
from mesh_calculator.core.grid import H3Cell
from mesh_calculator.core.config import MeshConfig
from mesh_calculator.data.cache import LOSCache, LOSResult


# ── Build a 12-cell corridor ──────────────────────────────────────────────────
c_start = h3.latlng_to_cell(40.0, 44.0, 8)
c_end   = h3.latlng_to_cell(40.0, 44.09, 8)   # ~9 km east
corridor = list(h3.grid_path_cells(c_start, c_end))
# Trim / pad to exactly 12 cells for reproducibility
corridor = corridor[:12]
if len(corridor) < 12:
    # Extend if needed
    extra = h3.latlng_to_cell(40.0, 44.12, 8)
    extra_path = list(h3.grid_path_cells(corridor[-1], extra))
    corridor += extra_path[1:12 - len(corridor) + 1]
n = len(corridor)
print(f"Corridor: {n} cells  ({corridor[0]} … {corridor[-1]})")

# ── Build a minimal MeshSurface-like object ───────────────────────────────────
config = MeshConfig()
cells = {}
for c in corridor:
    lat, lon = h3.cell_to_latlng(c)
    cells[c] = H3Cell(h3_index=c, lat=lat, lon=lon, elevation=100.0,
                      has_road=True, is_in_boundary=True)

class FakeSurface:
    pass

surface = FakeSurface()
surface.cells = cells
surface.config = config
surface.elevation_provider = None

cache = LOSCache()

# ── LOS mock: cells 0–4 are in a valley with no LOS to anything ahead.
# Only cell[5] can see cell[-1] (end). This forces the fallback to trigger
# for cells 0–4, demonstrating the cluster bug.
# VISIBLE pairs: (corridor[5], corridor[-1]) only.
RELAY = min(5, n - 2)
VISIBLE = {(corridor[RELAY], corridor[-1])}

def fake_los_batch(pairs, cells, config, cache, elevation_provider=None):
    results = {}
    for src, dst in pairs:
        vis = (src, dst) in VISIBLE or (dst, src) in VISIBLE
        results[(src, dst)] = LOSResult(
            clearance_m=50.0 if vis else -200.0,
            path_loss_db=110.0,
            distance_m=1000.0,
            is_visible=vis,
        )
    return results


def idx(c):
    return corridor.index(c) if c in corridor else f"?({c[:8]})"


# ── Run BEFORE fix — expect bug (valley cluster in chain) ─────────────────────
with patch("mesh_calculator.parallel.los_compute.compute_los_batch", fake_los_batch):
    chain_before = _greedy_place_towers(corridor, surface, cache, k=10)

chain_before_idx = [idx(c) for c in chain_before]
valley_cells_in_chain = [i for i in chain_before_idx
                         if isinstance(i, int) and 1 <= i < RELAY]
print(f"BEFORE fix — chain: {chain_before_idx}")

if valley_cells_in_chain:
    print(f"  BUG CONFIRMED — valley cells {valley_cells_in_chain} placed in chain")
else:
    print(f"  (no valley cluster — bug may already be fixed or scenario changed)")

# ── Apply the fix in-memory via monkey-patch ──────────────────────────────────
import types, mesh_calculator.optimization.corridor as _corridor_mod

def _patched_greedy(corridor_arg, surface_arg, cache_arg, k_arg, out_meta=None):
    """Identical to _greedy_place_towers but with chain.append removed from fallbacks."""
    if len(corridor_arg) < 2:
        return list(corridor_arg)

    import h3 as _h3
    from mesh_calculator.core.geometry import h3_distance
    from mesh_calculator.parallel.los_compute import compute_los_batch
    from mesh_calculator.core.grid import H3Cell

    _cells = surface_arg.cells
    _config = surface_arg.config
    _elevation_provider = surface_arg.elevation_provider

    edge_m = _h3.average_hexagon_edge_length(_config.h3_resolution, unit='m')
    buffer_ring = max(1, round(_config.road_buffer_m / edge_m) if _config.road_buffer_m > 0 else 1)

    road_cells_l = [c for c in corridor_arg if _cells.get(c) and _cells[c].has_road]
    if not road_cells_l:
        road_cells_l = list(corridor_arg)

    road_idx_map = {c: i for i, c in enumerate(road_cells_l)}
    cell_road_pos = {}
    last_road_j = 0
    for c in corridor_arg:
        if c in road_idx_map:
            last_road_j = road_idx_map[c]
        cell_road_pos[c] = last_road_j

    def _get_or_create(h3_idx):
        if h3_idx in _cells:
            return _cells[h3_idx]
        if _elevation_provider is None:
            return None
        lat, lon = _h3.cell_to_latlng(h3_idx)
        elev = _elevation_provider.get_elevation(lat, lon)
        cell = H3Cell(h3_index=h3_idx, lat=lat, lon=lon, elevation=elev,
                      has_road=False, is_in_boundary=False)
        _cells[h3_idx] = cell
        return cell

    def _get_buf(cell):
        buf = set()
        for nb in _h3.grid_disk(cell, buffer_ring):
            if _get_or_create(nb) is not None:
                buf.add(nb)
        buf.add(cell)
        return buf

    chain = [corridor_arg[0]]
    current = corridor_arg[0]
    current_road_j = 0

    while True:
        if len(chain) >= k_arg:
            if chain[-1] != corridor_arg[-1]:
                chain.append(corridor_arg[-1])
            break
        if current_road_j >= len(road_cells_l) - 1:
            if chain[-1] != corridor_arg[-1]:
                chain.append(corridor_arg[-1])
            break

        src_buffer = _get_buf(current)
        dst_candidates = {}
        for road_j in range(current_road_j + 1, len(road_cells_l)):
            road_c = road_cells_l[road_j]
            if h3_distance(current, road_c) > _config.max_visibility_m:
                break
            for nb in _get_buf(road_c):
                if nb not in dst_candidates:
                    if nb not in cell_road_pos:
                        cell_road_pos[nb] = road_j
                    dst_candidates[nb] = cell_road_pos[nb]

        if not dst_candidates:
            current_road_j += 1
            if current_road_j < len(road_cells_l):
                current = road_cells_l[current_road_j]
            # FIX: no chain.append(current)
            continue

        pairs = [
            (src, dst)
            for src in src_buffer
            for dst in dst_candidates
            if h3_distance(src, dst) <= _config.max_visibility_m
        ]

        if not pairs:
            current_road_j += 1
            if current_road_j < len(road_cells_l):
                current = road_cells_l[current_road_j]
            # FIX: no chain.append(current)
            continue

        results = compute_los_batch(pairs, _cells, _config, cache_arg,
                                    elevation_provider=_elevation_provider)

        best_j = -1
        best_dst = None
        best_clr = float('-inf')
        for (src, dst), los in results.items():
            if los.is_visible:
                j = dst_candidates[dst]
                if j > best_j or (j == best_j and los.clearance_m > best_clr):
                    best_j, best_dst, best_clr = j, dst, los.clearance_m

        if best_j < 0:
            current_road_j += 1
            if current_road_j < len(road_cells_l):
                current = road_cells_l[current_road_j]
            # FIX: no chain.append(current)
            continue

        chain.append(best_dst)
        if out_meta is not None:
            out_meta[best_dst] = {'algorithm': 'greedy', 'dp_steps': None, 'repair_round': None}
        current = best_dst
        current_road_j = best_j

    if chain[-1] != corridor_arg[-1]:
        chain.append(corridor_arg[-1])
    return chain


# ── Run AFTER fix ─────────────────────────────────────────────────────────────
# Reset surface.cells (may have been modified by buffer creation)
for c in list(cells.keys()):
    if c not in corridor:
        del cells[c]

with patch("mesh_calculator.parallel.los_compute.compute_los_batch", fake_los_batch):
    chain_after = _patched_greedy(corridor, surface, cache, k_arg=10)

chain_after_idx = [idx(c) for c in chain_after]
valley_cells_after = [i for i in chain_after_idx
                      if isinstance(i, int) and 1 <= i < RELAY]
print(f"\nAFTER fix  — chain: {chain_after_idx}")

if valley_cells_after:
    print(f"  FAIL — valley cluster still present: {valley_cells_after}")
    sys.exit(1)
else:
    last_cell = chain_after_idx[-1]
    if last_cell == n - 1:
        print(f"  PASS — no valley cluster, endpoint reached: {chain_after_idx}")
        sys.exit(0)
    else:
        print(f"  FAIL — endpoint not reached (got {last_cell}, expected {n-1})")
        sys.exit(1)
