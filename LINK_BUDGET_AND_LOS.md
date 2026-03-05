# Link Budget and LOS in Mesh Calculator

This document explains how radio feasibility is computed and how that affects tower placement.

## 1) Core concepts

- `Fresnel clearance`: geometric clearance of the first Fresnel zone along the RF path.
- `Path loss`: total attenuation between two towers.
- `Link budget`: maximum allowed path loss based on TX power, antenna gains, and receiver sensitivity.
- `Visibility` (`is_visible`): whether a link is accepted by policy.

In this project, Fresnel clearance affects link quality through diffraction loss, and visibility is decided by policy in `compute_los()`.

## 2) Where calculations happen

- Fresnel clearance: `mesh_calculator/physics/fresnel.py` (`compute_fresnel_clearance`)
- Path loss: `mesh_calculator/physics/path_loss.py` (`compute_path_loss`)
- LOS policy decision: `mesh_calculator/physics/los.py` (`compute_los`)
- Link budget parameters: `mesh_calculator/core/config.py` (`MeshConfig`)

## 3) Fresnel clearance calculation

For a pair of H3 cells:

1. Build straight RF path samples with `h3.grid_path_cells`.
2. For each sample, compute:
   - line altitude between endpoints (endpoint elevation + mast height),
   - earth curvature term,
   - first Fresnel radius.
3. Clearance at sample:
   - `clearance = line_altitude - (terrain + earth_curvature + fresnel_radius)`
4. Keep worst (minimum) clearance along the path.

Interpretation:

- Positive clearance: no Fresnel intrusion.
- Negative clearance: obstacle intrudes into Fresnel zone.

## 4) Path loss calculation

`compute_path_loss(distance, frequency, clearance, d1, d2)` does:

1. Free-space path loss (FSPL):
   - `FSPL(dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.44`
2. If clearance is negative, add knife-edge diffraction loss (ITU-style) using `nu`.
3. Return:
   - `total_path_loss = FSPL + diffraction_loss`

So Fresnel clearance already influences path loss via diffraction.

## 5) Link budget calculation

From `MeshConfig`:

- `tx_power_dbm = 10*log10(tx_power_mw)`
- `link_budget_db = tx_power_dbm + 2*antenna_gain_dbi - receiver_sensitivity_dbm`

`max_visibility_m` is derived from FSPL-only budget limit and used as a fast distance filter.

## 6) LOS acceptance policy

In `compute_los()`:

1. Reject immediately if distance exceeds `max_visibility_m`.
2. Compute `clearance` and `path_loss_db`.
3. Evaluate policy:
   - `budget_ok = path_loss_db <= link_budget_db`
   - `clearance_ok = True` when `min_fresnel_clearance_m is None`
   - otherwise `clearance_ok = clearance_m >= min_fresnel_clearance_m`
4. Final result:
   - `is_visible = budget_ok and clearance_ok`

Default behavior:

- `min_fresnel_clearance_m = None` (no hard clearance gate).

Optional stricter behavior:

- Set `min_fresnel_clearance_m` (for example `0.0`) to require non-negative clearance in addition to budget.

## 7) How this affects tower placement

Tower placement consumes `compute_los()` everywhere:

- Corridor DP and greedy placement (`optimization/corridor.py`)
- Gap repair rounds (`optimization/corridor.py`)
- Edge wiring between placed towers (`wire_corridor_edges`)
- Global visibility graph updates (`network/graph.py`)
- Road-cell coverage (`network/graph.py`)

Practical effect:

- More permissive policy (`min_fresnel_clearance_m=None`) usually increases feasible links, improves route continuity, and reduces fragmented clusters.
- Stricter policy (`min_fresnel_clearance_m=0.0` or higher) reduces feasible links and can force fallback layouts with fewer connecting edges.

## 8) Cache behavior

LOS results are cached in `LOSCache`.

Cache keys include:

- source/destination H3
- mast heights
- frequency
- `tx_power_mw`
- `antenna_gain_dbi`
- `receiver_sensitivity_dbm`
- `min_fresnel_clearance_m`

This prevents stale visibility reuse when radio parameters or clearance policy change.

## 9) Tuning guidance

For better connectivity in difficult terrain:

- Keep `min_fresnel_clearance_m: null` (or omit it), and tune:
  - `mast_height_m`
  - `tx_power_mw`
  - `antenna_gain_dbi`
  - `receiver_sensitivity_dbm`

For stricter geometric LOS compliance:

- Set `min_fresnel_clearance_m` to `0.0` or higher.
- Expect fewer feasible links and potentially more disconnected routes unless compensated by higher towers or better radio budget.

## 10) Runtime tower coverage API

Tower radial coverage is now an explicit runtime calculation, not an automatic route-pipeline export.

- Standalone compute entrypoint: `mesh_calculator/network/tower_coverage.py`
  - `CoverageSource(source_id, h3_index, lat, lon)`
  - `compute_h3_tower_coverage(...)`
- The same LOS function (`compute_los`) is reused, so Fresnel, diffraction, link budget, and optional clearance threshold are identical to placement logic.
- Output is covered cells only, including the source H3 cell with:
  - `distance_m = 0.0`
  - `path_loss_db = 0.0`
- `mesh-generator` calls this on demand for selected/all displayed towers and random clicked map points (when elevation is available).
