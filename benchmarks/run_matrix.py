"""Run the dense/sparse benchmark matrix on one machine and GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from configs import SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SPECS),
        default=list(SPECS),
    )
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--mode", choices=("native", "controlled"), default="native")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path)
    parser.add_argument("--sparse-root", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser.parse_args()


def run_one(
    monitor: Path,
    runner: Path,
    manuscript_root: Path,
    output_dir: Path,
    dataset: str,
    implementation: str,
    source_root: Path,
    phase: str,
    mode: str,
    gpu_index: int,
    cpu_threads: int,
    walks_in: Path | None = None,
    walks_out: Path | None = None,
    skip_training: bool = False,
    allow_busy_gpu: bool = False,
) -> int:
    command = [
        sys.executable,
        str(monitor),
        "--dataset",
        dataset,
        "--implementation",
        implementation,
        "--source-root",
        str(source_root),
        "--manuscript-root",
        str(manuscript_root),
        "--output-dir",
        str(output_dir),
        "--runner",
        str(runner),
        "--gpu-index",
        str(gpu_index),
        "--cpu-threads",
        str(cpu_threads),
        "--run-kind",
        f"{mode}-{phase}",
    ]
    if phase == "smoke":
        command.extend(("--epochs", "1"))
    if walks_in is not None:
        command.extend(("--walks-in", str(walks_in)))
    if walks_out is not None:
        command.extend(("--walks-out", str(walks_out)))
    if skip_training:
        command.append("--skip-training")
    if allow_busy_gpu:
        command.append("--allow-busy-gpu")
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = parse_args()
    benchmark_root = Path(__file__).resolve().parent
    manuscript_root = benchmark_root.parent
    workspace_root = manuscript_root.parent
    dense_root = (args.dense_root or workspace_root / "sclsd-dense").resolve()
    sparse_root = (args.sparse_root or workspace_root / "sclsd").resolve()
    output_root = args.output_root.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    matrix_root = output_root / f"{args.mode}-{args.phase}-{timestamp}"
    matrix_root.mkdir(parents=True, exist_ok=True)
    monitor = benchmark_root / "monitor_run.py"
    runner = benchmark_root / "run_training.py"
    failures = 0

    for dataset in args.datasets:
        if args.mode == "native":
            for implementation, source_root in (
                ("dense", dense_root),
                ("sparse", sparse_root),
            ):
                return_code = run_one(
                    monitor,
                    runner,
                    manuscript_root,
                    matrix_root / dataset / implementation,
                    dataset,
                    implementation,
                    source_root,
                    args.phase,
                    args.mode,
                    args.gpu_index,
                    args.cpu_threads,
                    allow_busy_gpu=args.allow_busy_gpu,
                )
                failures += int(return_code != 0)
        else:
            walks = matrix_root / dataset / "fixed_dense_walks.npy"
            generation_code = run_one(
                monitor,
                runner,
                manuscript_root,
                matrix_root / dataset / "walk_generation_dense",
                dataset,
                "dense",
                dense_root,
                args.phase,
                args.mode,
                args.gpu_index,
                args.cpu_threads,
                walks_out=walks,
                skip_training=True,
                allow_busy_gpu=args.allow_busy_gpu,
            )
            failures += int(generation_code != 0)
            if generation_code != 0:
                continue
            for implementation, source_root in (
                ("dense", dense_root),
                ("sparse", sparse_root),
            ):
                return_code = run_one(
                    monitor,
                    runner,
                    manuscript_root,
                    matrix_root / dataset / implementation,
                    dataset,
                    implementation,
                    source_root,
                    args.phase,
                    args.mode,
                    args.gpu_index,
                    args.cpu_threads,
                    walks_in=walks,
                    allow_busy_gpu=args.allow_busy_gpu,
                )
                failures += int(return_code != 0)

    summarize = benchmark_root / "summarize_results.py"
    subprocess.run(
        [sys.executable, str(summarize), str(matrix_root)], check=False
    )
    print(f"Benchmark matrix: {matrix_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
