Note: All of the code was written by LLMs: Claude Code and ChatGPT.

[![Unit Tests](https://github.com/FlameFlameFlame/mesh-calculator/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/FlameFlameFlame/mesh-calculator/actions/workflows/unit-tests.yml)
[![Smoke Tests](https://github.com/FlameFlameFlame/mesh-calculator/actions/workflows/smoke-tests.yml/badge.svg?branch=main)](https://github.com/FlameFlameFlame/mesh-calculator/actions/workflows/smoke-tests.yml)

# Project Description
mesh_calculator is a Python package that computes mesh-network tower placement and route planning using terrain, radio, and connectivity constraints.

# How to Run It
```bash
uv sync
```

Run the main optimizer CLI:

```bash
uv run mesh-calculator --config /path/to/config.yaml --output /path/to/output
```

Run the routes CLI:

```bash
uv run mesh-calculator-routes --config /path/to/config.yaml --output /path/to/output
```

# High-Level Implementation Details
The package is organized into modules for core geospatial processing, LOS and path-loss physics, graph/network construction, optimization, and data I/O. CLI entrypoints orchestrate these modules into full planning pipelines and export generated results for backend/frontend consumption.
