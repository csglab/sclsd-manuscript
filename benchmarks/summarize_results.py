"""Combine benchmark JSON files into CSV and Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    return parser.parse_args()


def stage(result: Dict[str, Any], name: str) -> Dict[str, Any]:
    for value in result.get("stages", []):
        if value.get("name") == name:
            return value
    return {}


def nested(value: Dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def gib(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024**3)


def mib_to_gib(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / 1024


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_rows(root: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for result_path in sorted(root.glob("**/result.json")):
        result = json.loads(result_path.read_text())
        monitor_path = result_path.with_name("monitor.json")
        monitor = json.loads(monitor_path.read_text()) if monitor_path.exists() else {}
        config = result.get("effective_config", {})
        walks = config.get("walks", {})
        training = stage(result, "training")
        prior = stage(result, "prior_transition")
        walk_generation = stage(result, "walk_generation")
        error = result.get("error", {})
        rows.append(
            {
                "result_dir": str(result_path.parent.relative_to(root)),
                "run_kind": result.get("run_kind"),
                "dataset": result.get("display_name", result.get("dataset")),
                "implementation": result.get("implementation"),
                "status": result.get("status"),
                "commit": nested(result, "source", "git_commit"),
                "cells": (result.get("spec", {}).get("expected_shape") or [None, None])[0],
                "genes": (result.get("spec", {}).get("expected_shape") or [None, None])[1],
                "walks": walks.get("num_walks"),
                "walk_length": walks.get("path_len"),
                "batch_size": walks.get("batch_size"),
                "epochs": result.get("epochs"),
                "total_wall_seconds": result.get("total_wall_seconds"),
                "prior_wall_seconds": prior.get("wall_seconds"),
                "walk_wall_seconds": walk_generation.get("wall_seconds"),
                "training_wall_seconds": training.get("wall_seconds"),
                "training_cpu_seconds": training.get("cpu_seconds"),
                "peak_cpu_rss_gib": gib(
                    nested(monitor, "summary", "peak_rss_process_tree_bytes")
                ),
                "peak_torch_allocated_gib": gib(
                    max(
                        (
                            item.get("cuda_peak_allocated_bytes") or 0
                            for item in result.get("stages", [])
                        ),
                        default=0,
                    )
                ),
                "peak_torch_reserved_gib": gib(
                    max(
                        (
                            item.get("cuda_peak_reserved_bytes") or 0
                            for item in result.get("stages", [])
                        ),
                        default=0,
                    )
                ),
                "peak_nvml_process_gpu_gib": mib_to_gib(
                    nested(
                        monitor,
                        "summary",
                        "per_stage",
                        "training",
                        "gpu_process_memory_mib",
                        "peak",
                    )
                    or nested(monitor, "summary", "gpu_process_memory_mib", "peak")
                ),
                "mean_gpu_utilization_percent": nested(
                    monitor,
                    "summary",
                    "per_stage",
                    "training",
                    "gpu_utilization_percent",
                    "mean",
                ),
                "peak_gpu_utilization_percent": nested(
                    monitor,
                    "summary",
                    "per_stage",
                    "training",
                    "gpu_utilization_percent",
                    "peak",
                ),
                "mean_cpu_utilization_percent": nested(
                    monitor,
                    "summary",
                    "per_stage",
                    "training",
                    "cpu_normalized_percent",
                    "mean",
                ),
                "error": error.get("message"),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    root = args.result_root.resolve()
    rows = load_rows(root)
    if not rows:
        raise RuntimeError(f"No result.json files found under {root}")

    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = root / "summary.md"
    headers = (
        "Dataset",
        "Implementation",
        "Status",
        "Epochs",
        "Total wall (s)",
        "Training wall (s)",
        "Peak CPU RSS (GiB)",
        "Peak GPU process (GiB)",
        "Peak torch allocated (GiB)",
        "Mean GPU util (%)",
    )
    lines = [
        "# Benchmark summary",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    fmt(row["dataset"]),
                    fmt(row["implementation"]),
                    fmt(row["status"]),
                    fmt(row["epochs"]),
                    fmt(row["total_wall_seconds"]),
                    fmt(row["training_wall_seconds"]),
                    fmt(row["peak_cpu_rss_gib"]),
                    fmt(row["peak_nvml_process_gpu_gib"]),
                    fmt(row["peak_torch_allocated_gib"]),
                    fmt(row["mean_gpu_utilization_percent"]),
                )
            )
            + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
