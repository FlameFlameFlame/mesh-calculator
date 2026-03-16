"""
Deterministic serial regression probe for route pipeline outputs.

Usage:
    uv run python -m mesh_calculator.utils.parallel_probe \
      --project-dir /path/to/project
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

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


def _artifact_hashes_for_dir(artifact_dir: Path) -> dict[str, str | None]:
    return {
        name: _artifact_hash(artifact_dir / name)
        for name in (
            "towers.geojson",
            "visibility_edges.geojson",
            "coverage.geojson",
            "report.json",
            "grid_cells.geojson",
            "gap_repair_hexes.geojson",
        )
    }


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
    runs: int,
    output_dir: Path | None = None,
    baseline_dir: Path | None = None,
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

    probe_root = output_dir or (project_dir / "serial_probe")
    probe_root.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "project_dir": str(project_dir),
        "runs": int(runs),
        "executions": {},
        "diffs_vs_run1": {},
        "diffs_vs_baseline": {},
    }
    if baseline_dir is not None:
        report["baseline_dir"] = str(baseline_dir)

    baseline_fp = None
    baseline_hashes = None
    baseline_artifact_hashes = (
        _artifact_hashes_for_dir(baseline_dir)
        if baseline_dir is not None
        else None
    )
    if baseline_artifact_hashes is not None:
        report["baseline_artifact_hashes"] = baseline_artifact_hashes

    for run_idx in range(1, int(runs) + 1):
        run_dir = probe_root / f"run_{run_idx}"
        debug_dir = run_dir / "debug_snapshots"
        run_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        run_cfg = copy.deepcopy(mesh_config)

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

        artifact_hashes = _artifact_hashes_for_dir(run_dir)
        fingerprint = _summary_fingerprint(summary)
        report["executions"][str(run_idx)] = {
            "summary": summary,
            "fingerprint": fingerprint,
            "artifact_hashes": artifact_hashes,
            "run_dir": str(run_dir),
        }

        if run_idx == 1:
            baseline_fp = fingerprint
            baseline_hashes = artifact_hashes
            continue

        diffs = {
            "summary_fingerprint_diff": fingerprint != baseline_fp,
            "artifact_hash_diff": artifact_hashes != baseline_hashes,
        }
        report["diffs_vs_run1"][str(run_idx)] = diffs
        if baseline_artifact_hashes is not None:
            report["diffs_vs_baseline"][str(run_idx)] = {
                "artifact_hash_diff": artifact_hashes != baseline_artifact_hashes,
                "differing_artifacts": sorted(
                    name
                    for name, digest in artifact_hashes.items()
                    if digest != baseline_artifact_hashes.get(name)
                ),
            }

    if baseline_artifact_hashes is not None and "1" not in report["diffs_vs_baseline"]:
        first_hashes = report["executions"].get("1", {}).get("artifact_hashes", {})
        report["diffs_vs_baseline"]["1"] = {
            "artifact_hash_diff": first_hashes != baseline_artifact_hashes,
            "differing_artifacts": sorted(
                name
                for name, digest in first_hashes.items()
                if digest != baseline_artifact_hashes.get(name)
            ),
        }

    report_path = probe_root / "serial_probe_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run serial reproducibility probe on a project",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Project directory containing config.yaml/routes.json/grid_bundle.json",
    )
    parser.add_argument(
        "--runs",
        default=3,
        type=int,
        help="Number of serial repetitions (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for probe artifacts",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Optional directory of existing artifacts to compare each run against",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    baseline_dir = Path(args.baseline_dir).resolve() if args.baseline_dir else None
    runs = max(1, int(args.runs))
    report = run_probe(
        project_dir,
        runs=runs,
        output_dir=output_dir,
        baseline_dir=baseline_dir,
    )
    print(json.dumps(
        {
            "report_path": report.get("report_path"),
            "runs": report.get("runs"),
            "diffs_vs_run1": report.get("diffs_vs_run1", {}),
            "diffs_vs_baseline": report.get("diffs_vs_baseline", {}),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
