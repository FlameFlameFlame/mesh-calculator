# Change Summary

- 2026-03-05: Added working-rule memory file under `.claude/agent-memory/git-commit-writer/MEMORY.md`.
- 2026-03-05: Fixed LOS visibility rule in `mesh_calculator/physics/los.py` to use link-budget feasibility (`path_loss_db <= link_budget_db`) instead of strict `clearance > 0`.
- 2026-03-05: Added regression tests in `mesh_calculator/tests/test_physics.py` for two cases: (1) negative-clearance links that are still budget-feasible, and (2) positive-clearance links that exceed link budget.
- 2026-03-05: Verified on examples (`gyumri-greedy-bad`, `gyumri-vanadzor-random`, `yerevan_gyumri`) that cluster connectivity improves (all become single-cluster in DP route pipeline runs).
- 2026-03-05: Added optional `MeshConfig.min_fresnel_clearance_m` policy threshold (default `None`) so clearance gating can be enabled explicitly without changing default behavior.
- 2026-03-05: Updated LOS acceptance to require link budget and, when configured, `clearance_m >= min_fresnel_clearance_m`.
- 2026-03-05: Expanded LOS cache key with radio/policy parameters (`tx_power_mw`, `antenna_gain_dbi`, `receiver_sensitivity_dbm`, `min_fresnel_clearance_m`) to prevent stale visibility reuse across policy changes.
- 2026-03-05: Added cache and config regression tests plus dual-mode scenario validation (`default_none` vs `strict_zero`) on `gyumri-greedy-bad`, `gyumri-vanadzor-random`, and `yerevan_gyumri`.
