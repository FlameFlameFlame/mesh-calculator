# Link Budget and LOS in Mesh Calculator

This document explains how radio feasibility is computed and how that affects tower placement.

## 1) Core concepts

- `Fresnel clearance`: geometric clearance of the first Fresnel zone along the RF path.
- `Path loss`: total attenuation between two towers.
- `Link budget`: maximum allowed path loss based on TX power, antenna gains, and receiver sensitivity.
- `Visibility` (`is_visible`): whether a link is accepted by policy.

In this project, Fresnel clearance affects link quality through diffraction loss, and route-planning visibility is decided by policy in `compute_los()`.

## 2) Where calculations happen

- Fresnel clearance: `mesh_calculator/physics/fresnel.py` (`compute_fresnel_clearance`)
- Path loss: `mesh_calculator/physics/path_loss.py` (`compute_path_loss`)
- LOS policy decision: `mesh_calculator/physics/los.py` (`compute_los`)
- Link budget parameters: `mesh_calculator/core/config.py` (`MeshConfig`)

## 3) Fresnel clearance calculation

For a pair of H3 cells:

1. Cell elevations are conservative: each H3 cell stores the **maximum DEM elevation** inside the cell polygon (not centroid sample).
2. For LOS/Fresnel, terrain along the RF line is sampled along the straight path.
3. At each sample, compute:
   - line altitude between endpoints (endpoint elevation + antenna height),
   - earth curvature term,
   - first Fresnel radius.
4. Clearance at sample:
   - `clearance = line_altitude - (terrain + earth_curvature + fresnel_radius)`
5. Final link clearance is the **minimum clearance across all samples** (`argmin clearance`), not the point of maximum terrain height.

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
3. Evaluate policy on coarse result:
   - `budget_ok = path_loss_db <= link_budget_db`
   - `fresnel_ok = max_obstruction_ratio <= 0.4`
4. If coarse policy passes and elevation data is available, run dense DEM profile verification and recompute policy on verified values.
5. Final result:
   - `is_visible = budget_ok and fresnel_ok`

Default behavior:

- hardcoded practical Fresnel policy: first-Fresnel obstruction must not exceed 40%.

Dense verification behavior:

- first dense pass samples the DEM along the straight RF line
- worst local intervals are then refined at higher density before final acceptance
- `los_dense_sample_step_m` controls the first dense pass step
- `los_dense_max_samples` caps the total adaptive sample count

## 7) How this affects tower placement

Tower placement consumes `compute_los()` everywhere:

- Corridor DP and greedy placement (`optimization/corridor.py`)
- Gap repair rounds (`optimization/corridor.py`)
- Edge wiring between placed towers (`wire_corridor_edges`)
- Global visibility graph updates (`network/graph.py`)
- Road-cell coverage (`network/graph.py`)

Practical effect:

- Compared with budget-only acceptance, the 40% Fresnel rule rejects more terrain-obstructed links while still allowing moderate Fresnel intrusion common in practical RF planning.
- Dense DEM verification reduces false positives from coarse H3-center sampling without paying the cost of ultra-dense sampling on every candidate link.

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
- `los_dense_sample_step_m`
- `los_dense_max_samples`
- verification mode sentinel (`hybrid_accept_verify`)

This prevents stale visibility reuse when radio parameters or clearance policy change.

## 9) Tuning guidance

For better connectivity or stricter acceptance in difficult terrain, tune:

- `mast_height_m`
- `tx_power_mw`
- `antenna_gain_dbi`
- `receiver_sensitivity_dbm`
- `los_dense_sample_step_m`
- `los_dense_max_samples`

`min_fresnel_clearance_m` remains in config for backward-compatible parsing and cache identity, but route-planning LOS now uses the hardcoded 40% Fresnel obstruction rule instead of a fixed clearance threshold.

## 10) Runtime tower coverage API

Tower radial coverage is now an explicit runtime calculation, not an automatic route-pipeline export.

- Standalone compute entrypoint: `mesh_calculator/network/tower_coverage.py`
  - `CoverageSource(source_id, h3_index, lat, lon)`
  - `compute_h3_tower_coverage(...)`
- Runtime tower coverage uses a strict terrain-shadow model:
  - hard geometric LOS (`clearance >= 0`) from tower top to coverage receiver height
  - no diffraction-based pass-through for blocked cells
  - FSPL-only budget check after LOS passes
- Terrain blocking uses sampled straight-line terrain and takes the minimum
  shadow clearance along the path.
- Coverage receiver endpoint height is controlled by `coverage_receiver_height_m` (default `1.5 m`), while source endpoint uses `mast_height_m`.
- Output includes **all cells in radius** (covered and uncovered), including source H3 cell with:
  - `distance_m = 0.0`
  - `path_loss_db = 0.0`
- Uncovered cells are returned with `is_covered=false` and serving metrics set to `null`.
- Source attribution is RF-serving based:
  - `serving_tower_id`: strongest received signal (minimum path loss)
  - `closest_tower_id`: nearest visible source (debug/backward compatibility)
- `mesh-generator` calls this on demand for selected/all displayed towers and random clicked map points (when elevation is available).
