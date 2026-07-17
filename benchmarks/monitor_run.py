"""Launch and monitor one training benchmark in a fresh subprocess."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bone_marrow", "cancer"), required=True)
    parser.add_argument("--implementation", choices=("dense", "sparse"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--walks-in", type=Path)
    parser.add_argument("--walks-out", type=Path)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--run-kind", default="native")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nvidia_query(index: int) -> Dict[str, Optional[float]]:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    values = [value.strip() for value in output.strip().split(",")]

    def number(value: str) -> Optional[float]:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "gpu_utilization_percent": number(values[0]),
        "gpu_memory_used_mib": number(values[1]),
        "gpu_power_watts": number(values[2]),
    }


def nvidia_hardware(index: int) -> Dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [field.strip() for field in query.split(",")]
    header = subprocess.check_output(["nvidia-smi"], text=True)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "memory_total_mib": float(fields[2]),
        "driver_version": fields[3],
        "compute_capability": fields[4],
        "driver_cuda_compatibility": cuda_match.group(1) if cuda_match else None,
    }


def compute_processes(index: int) -> Dict[int, float]:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    processes: Dict[int, float] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            processes[int(fields[0])] = float(fields[1])
        except ValueError:
            continue
    return processes


def process_tree(process: psutil.Process) -> List[psutil.Process]:
    try:
        return [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def process_metrics(process: psutil.Process) -> Tuple[int, float, List[int]]:
    rss = 0
    cpu_seconds = 0.0
    pids: List[int] = []
    for member in process_tree(process):
        try:
            pids.append(member.pid)
            rss += member.memory_info().rss
            times = member.cpu_times()
            cpu_seconds += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss, cpu_seconds, pids


def numeric_summary(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"mean": None, "median": None, "peak": None}
    return {
        "mean": statistics.fmean(present),
        "median": statistics.median(present),
        "peak": max(present),
    }


def stream_output(
    pipe,
    log_path: Path,
    stage_state: Dict[str, Optional[str]],
    stage_lock: threading.Lock,
    events: List[Dict[str, Any]],
) -> None:
    with log_path.open("w") as log:
        for line in iter(pipe.readline, ""):
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            stripped = line.strip()
            if stripped.startswith("SCLSD_BENCHMARK_EVENT "):
                try:
                    event = json.loads(stripped.split(" ", 1)[1])
                except json.JSONDecodeError:
                    continue
                events.append(event)
                with stage_lock:
                    if event.get("event") == "stage_start":
                        stage_state["current"] = event.get("stage")
                    elif event.get("event") == "stage_end":
                        stage_state["current"] = None


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    monitor_path = output_dir / "monitor.json"
    log_path = output_dir / "stdout.log"

    hardware = nvidia_hardware(args.gpu_index)
    busy_before = compute_processes(args.gpu_index)
    if busy_before and not args.allow_busy_gpu:
        raise RuntimeError(
            "GPU already has compute processes: "
            + ", ".join(str(pid) for pid in sorted(busy_before))
        )

    command = [
        str(args.python),
        str(args.runner.resolve()),
        "--dataset",
        args.dataset,
        "--implementation",
        args.implementation,
        "--source-root",
        str(args.source_root.resolve()),
        "--manuscript-root",
        str(args.manuscript_root.resolve()),
        "--output",
        str(result_path),
        "--device",
        "cuda:0",
        "--run-kind",
        args.run_kind,
    ]
    if args.epochs is not None:
        command.extend(("--epochs", str(args.epochs)))
    if args.walks_in is not None:
        command.extend(("--walks-in", str(args.walks_in.resolve())))
    if args.walks_out is not None:
        command.extend(("--walks-out", str(args.walks_out.resolve())))
    if args.skip_training:
        command.append("--skip-training")

    env = os.environ.copy()
    source_path = str((args.source_root / "src").resolve())
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output_dir / "matplotlib")
    env["PYTHONUNBUFFERED"] = "1"
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        env[variable] = str(args.cpu_threads)

    started_at = utc_now()
    start_wall = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    stage_state: Dict[str, Optional[str]] = {"current": None}
    stage_lock = threading.Lock()
    events: List[Dict[str, Any]] = []
    output_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, log_path, stage_state, stage_lock, events),
        daemon=True,
    )
    output_thread.start()
    ps_process = psutil.Process(process.pid)

    samples: List[Dict[str, Any]] = []
    prior_wall: Optional[float] = None
    prior_cpu: Optional[float] = None
    while process.poll() is None:
        sample_wall = time.perf_counter()
        rss, cpu_seconds, pids = process_metrics(ps_process)
        try:
            gpu = nvidia_query(args.gpu_index)
            gpu_processes = compute_processes(args.gpu_index)
        except (OSError, subprocess.SubprocessError, ValueError):
            gpu = {
                "gpu_utilization_percent": None,
                "gpu_memory_used_mib": None,
                "gpu_power_watts": None,
            }
            gpu_processes = {}
        process_gpu_mib = sum(gpu_processes.get(pid, 0.0) for pid in pids)
        aggregate_cpu_percent = None
        normalized_cpu_percent = None
        if prior_wall is not None and prior_cpu is not None:
            elapsed = sample_wall - prior_wall
            if elapsed > 0:
                aggregate_cpu_percent = 100.0 * (cpu_seconds - prior_cpu) / elapsed
                normalized_cpu_percent = aggregate_cpu_percent / args.cpu_threads
        with stage_lock:
            current_stage = stage_state["current"]
        samples.append(
            {
                "elapsed_seconds": sample_wall - start_wall,
                "stage": current_stage,
                "rss_process_tree_bytes": rss,
                "cpu_seconds_process_tree": cpu_seconds,
                "aggregate_cpu_percent": aggregate_cpu_percent,
                "normalized_cpu_percent": normalized_cpu_percent,
                "process_gpu_memory_mib": process_gpu_mib,
                **gpu,
            }
        )
        prior_wall = sample_wall
        prior_cpu = cpu_seconds
        time.sleep(max(0.05, args.sample_interval))

    return_code = process.wait()
    output_thread.join(timeout=10)
    finished_at = utc_now()
    elapsed = time.perf_counter() - start_wall

    def sample_summary(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "peak_rss_process_tree_bytes": max(
                (sample["rss_process_tree_bytes"] for sample in selected), default=0
            ),
            "cpu_aggregate_percent": numeric_summary(
                sample["aggregate_cpu_percent"] for sample in selected
            ),
            "cpu_normalized_percent": numeric_summary(
                sample["normalized_cpu_percent"] for sample in selected
            ),
            "gpu_utilization_percent": numeric_summary(
                sample["gpu_utilization_percent"] for sample in selected
            ),
            "gpu_device_memory_used_mib": numeric_summary(
                sample["gpu_memory_used_mib"] for sample in selected
            ),
            "gpu_process_memory_mib": numeric_summary(
                sample["process_gpu_memory_mib"] for sample in selected
            ),
            "gpu_power_watts": numeric_summary(
                sample["gpu_power_watts"] for sample in selected
            ),
        }

    stage_names = sorted(
        {sample["stage"] for sample in samples if sample.get("stage") is not None}
    )
    summary = {
        "peak_rss_process_tree_bytes": max(
            (sample["rss_process_tree_bytes"] for sample in samples), default=0
        ),
        "cpu_aggregate_percent": numeric_summary(
            sample["aggregate_cpu_percent"] for sample in samples
        ),
        "cpu_normalized_percent": numeric_summary(
            sample["normalized_cpu_percent"] for sample in samples
        ),
        "gpu_utilization_percent": numeric_summary(
            sample["gpu_utilization_percent"] for sample in samples
        ),
        "gpu_device_memory_used_mib": numeric_summary(
            sample["gpu_memory_used_mib"] for sample in samples
        ),
        "gpu_process_memory_mib": numeric_summary(
            sample["process_gpu_memory_mib"] for sample in samples
        ),
        "gpu_power_watts": numeric_summary(
            sample["gpu_power_watts"] for sample in samples
        ),
        "per_stage": {
            name: sample_summary(
                [sample for sample in samples if sample.get("stage") == name]
            )
            for name in stage_names
        },
    }
    monitor = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": elapsed,
        "return_code": return_code,
        "command": command,
        "gpu_index": args.gpu_index,
        "hardware": hardware,
        "cpu_threads": args.cpu_threads,
        "sample_interval_seconds": args.sample_interval,
        "gpu_processes_before_run": busy_before,
        "summary": summary,
        "events": events,
        "samples": samples,
    }
    write_json(monitor_path, monitor)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
