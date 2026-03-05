# Change Summary

- 2026-03-05: Added working-rule memory file under `.claude/agent-memory/git-commit-writer/MEMORY.md`.
- 2026-03-05: Fixed LOS visibility rule in `mesh_calculator/physics/los.py` to use link-budget feasibility (`path_loss_db <= link_budget_db`) instead of strict `clearance > 0`.
- 2026-03-05: Added regression tests in `mesh_calculator/tests/test_physics.py` for two cases: (1) negative-clearance links that are still budget-feasible, and (2) positive-clearance links that exceed link budget.
- 2026-03-05: Verified on examples (`gyumri-greedy-bad`, `gyumri-vanadzor-random`, `yerevan_gyumri`) that cluster connectivity improves (all become single-cluster in DP route pipeline runs).
