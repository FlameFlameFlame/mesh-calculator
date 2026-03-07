"""
Deterministic serial-vs-parallel regression probe for route pipeline outputs.

Usage:
    uv run python -m mesh_calculator.utils.parallel_probe \
      --project-dir /path/to/project
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import yaml

from ..core.config import MeshCalculatorConfig, RouteSpec
from ..core.grid_provider import GridProvider
from ..optimization.route_pipeline import run_route_pipeline


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _resolve_path(project_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return project_dir / p


def _artifact_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_route_specs(routes_json: dict) -> list[RouteSpec]:
    out: list[RouteSpec] = []
    for route in routes_json.get("routes", []):
        out.append(
            RouteSpec(
                route_id=route["route_id"],
                features=route.get("features", []),
                site1=route.get("site1", {}),
                site2=route.get("site2", {}),
                max_towers_per_route=int(route.get("max_towers_per_route", 10)),
            )
        )
    return out


def _parse_workers(values: str) -> list[int]:
    workers = []
    for part in values.split(","):
        part = part.strip()
        if not part:
            continue
        workers.append(max(1, int(part)))
    if not workers:
        workers = [1, 2, 4, 8]
    if 1 not in workers:
        workers.insert(0, 1)
    # preserve order but dedupe
    seen = set()
    ordered = []
    for w in workers:
        if w in seen:
            continue
        seen.add(w)
        ordered.append(w)
    return ordered


def _summary_fingerprint(summary: dict) -> dict:
    return {
        "total_towers": summary.get("total_towers"),
        "visibility_edges": summary.get("visibility_edges"),
        "num_clusters": summary.get("num_clusters"),
        "route_summaries": summary.get("route_summaries"),
        "los_batch": summary.get("los_batch"),
    }


def run_probe(
    project_dir: Path,
    workers: Iterable[int],
    output_dir: Path | None = None,
) -> dict:
    config_path = project_dir / "config.yaml"
    routes_path = project_dir / "routes.json"

    with config_path.open() as f:
        cfg_dict = yaml.safe_load(f) or {}
    cfg = MeshCalculatorConfig.from_dict(cfg_dict)
    mesh_config = cfg.parameters

    routes_json = _read_json(routes_path)
    routes = _build_route_specs(routes_json)
    if not routes:
        raise ValueError(f"No routes found in {routes_path}")

    inputs = cfg.inputs
    if inputs is None:
        raise ValueError("config.yaml missing inputs")

    boundary_path = _resolve_path(project_dir, inputs.boundary)
    city_path = _resolve_path(project_dir, inputs.city_boundaries)
    elev_path = _resolve_path(project_dir, inputs.elevation)
    bundle_path = _resolve_path(project_dir, inputs.grid_bundle or "grid_bundle.json")
    if boundary_path is None or elev_path is None or bundle_path is None:
        raise ValueError("Missing required inputs: boundary/elevation/grid_bundle")

    boundary_geojson = _read_json(boundary_path)
    city_geojson = _read_json(city_path) if city_path and city_path.exists() else None

    probe_root = output_dir or (project_dir / "parallel_probe")
    probe_root.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "project_dir": str(project_dir),
        "workers": list(workers),
        "runs": {},
        "diffs_vs_w1": {},
    }

    baseline_fp = None
    baseline_hashes = None

    for worker in workers:
        run_dir = probe_root / f"workers_{worker}"
        debug_dir = run_dir / "debug_snapshots"
        run_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        run_cfg = copy.deepcopy(mesh_config)
        run_cfg.los_parallel_workers = int(worker)

        provider = GridProvider.from_bundle(str(bundle_path), elevation_path=str(elev_path))
        try:
            summary = run_route_pipeline(
                routes=routes,
                mesh_config=run_cfg,
                grid_provider=provider,
                city_boundaries_geojson=city_geojson,
                boundary_geojson=boundary_geojson,
                output_dir=str(run_dir),
                debug_snapshot_dir=str(debug_dir),
            )
        finally:
            provider.close()

        artifact_hashes = {
            name: _artifact_hash(run_dir / name)
            for name in (
                "towers.geojson",
                "visibility_edges.geojson",
                "coverage.geojson",
                "report.json",
                "grid_cells.geojson",
                "gap_repair_hexes.geojson",
            )
        }
        fingerprint = _summary_fingerprint(summary)
        report["runs"][str(worker)] = {
            "summary": summary,
            "fingerprint": fingerprint,
            "artifact_hashes": artifact_hashes,
            "run_dir": str(run_dir),
        }

        if worker == 1:
            baseline_fp = fingerprint
            baseline_hashes = artifact_hashes
            continue

        diffs = {
            "summary_fingerprint_diff": fingerprint != baseline_fp,
            "artifact_hash_diff": artifact_hashes != baseline_hashes,
        }
        report["diffs_vs_w1"][str(worker)] = diffs

    report_path = probe_root / "parallel_probe_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run serial-vs-parallel reproducibility probe on a project",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Project directory containing config.yaml/routes.json/grid_bundle.json",
    )
    parser.add_argument(
        "--workers",
        default="1,2,4,8",
        help="Comma-separated worker counts (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for probe artifacts",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    workers = _parse_workers(args.workers)
    report = run_probe(project_dir, workers=workers, output_dir=output_dir)
    print(json.dumps(
        {
            "report_path": report.get("report_path"),
            "workers": report.get("workers"),
            "diffs_vs_w1": report.get("diffs_vs_w1", {}),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
