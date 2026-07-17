"""Run one exact manuscript training benchmark and emit structured metrics."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from configs import build_lsd_config, get_spec, resolved_paths


EVENT_PREFIX = "SCLSD_BENCHMARK_EVENT "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bone_marrow", "cancer"), required=True)
    parser.add_argument("--implementation", choices=("dense", "sparse"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--walks-in", type=Path)
    parser.add_argument("--walks-out", type=Path)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--run-kind", default="native")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def package_versions() -> Dict[str, str]:
    names = (
        "sclsd",
        "torch",
        "scanpy",
        "anndata",
        "pyro-ppl",
        "numpy",
        "scipy",
        "pandas",
        "torchdiffeq",
        "cellrank",
        "scikit-learn",
    )
    versions: Dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def emit_event(event: Dict[str, Any]) -> None:
    print(EVENT_PREFIX + json.dumps(event, sort_keys=True), flush=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_stats(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


@contextmanager
def measured_stage(
    name: str,
    device: torch.device,
    stages: list[Dict[str, Any]],
) -> Iterator[None]:
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_cuda = cuda_stats(device)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    emit_event({"event": "stage_start", "stage": name, "time": utc_now()})
    status = "ok"
    error: Optional[str] = None
    try:
        yield
    except BaseException as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            synchronize(device)
        except BaseException as sync_exc:
            if error is None:
                status = "error"
                error = f"{type(sync_exc).__name__}: {sync_exc}"
        end_cuda = cuda_stats(device)
        stage = {
            "name": name,
            "status": status,
            "wall_seconds": time.perf_counter() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
            "cuda_start_allocated_bytes": start_cuda.get("allocated_bytes"),
            "cuda_start_reserved_bytes": start_cuda.get("reserved_bytes"),
            "cuda_peak_allocated_bytes": end_cuda.get("peak_allocated_bytes"),
            "cuda_peak_reserved_bytes": end_cuda.get("peak_reserved_bytes"),
            "error": error,
        }
        stages.append(stage)
        emit_event({"event": "stage_end", "time": utc_now(), **stage})


def config_record(cfg: Any) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def write_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_source(source_root: Path, implementation: str) -> Dict[str, Any]:
    import sclsd
    from sclsd.train import walks

    expected_src = (source_root / "src").resolve()
    imported = Path(sclsd.__file__).resolve()
    if not imported.is_relative_to(expected_src):
        raise RuntimeError(
            f"Imported sclsd from {imported}, expected a module below {expected_src}"
        )
    has_sparse_sampler = hasattr(walks, "random_walks_sparse")
    if implementation == "sparse" and not has_sparse_sampler:
        raise RuntimeError("Sparse implementation does not expose random_walks_sparse")
    if implementation == "dense" and has_sparse_sampler:
        raise RuntimeError("Dense implementation unexpectedly exposes random_walks_sparse")
    return {
        "source_root": str(source_root.resolve()),
        "imported_module": str(imported),
        "git_commit": git_value(source_root, "rev-parse", "HEAD"),
        "git_branch": git_value(source_root, "branch", "--show-current"),
        "git_status_porcelain": git_value(source_root, "status", "--porcelain"),
        "has_sparse_sampler": has_sparse_sampler,
    }


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    source_root = args.source_root.resolve()
    manuscript_root = args.manuscript_root.resolve()
    spec = get_spec(args.dataset)
    paths = resolved_paths(manuscript_root, spec)
    epochs = spec.epochs if args.epochs is None else args.epochs
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    result: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "run_kind": args.run_kind,
        "dataset": spec.name,
        "display_name": spec.display_name,
        "implementation": args.implementation,
        "epochs": epochs,
        "skip_training": args.skip_training,
        "stages": [],
        "spec": dataclasses.asdict(spec),
    }
    write_result(args.output, result)

    try:
        source = validate_source(source_root, args.implementation)
        notebook_hash = sha256(paths["notebook"])
        data_hash = sha256(paths["data"])
        if notebook_hash != spec.notebook_sha256:
            raise RuntimeError(
                f"Notebook hash mismatch: {notebook_hash} != {spec.notebook_sha256}"
            )
        if data_hash != spec.data_sha256:
            raise RuntimeError(f"Dataset hash mismatch: {data_hash} != {spec.data_sha256}")

        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
            torch.cuda.set_device(device)
            torch.empty(1, device=device)
            synchronize(device)
            torch.cuda.empty_cache()

        from sclsd.train import LSD
        from sclsd.utils import set_all_seeds
        from sclsd.utils.seed import clear_pyro_state
        import scanpy as sc

        result.update(
            {
                "source": source,
                "notebook_sha256": notebook_hash,
                "data_sha256": data_hash,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": package_versions(),
                    "torch_cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "cpu_model": cpu_model(),
                    "accessible_cpu_threads": len(os.sched_getaffinity(0)),
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "device": str(device),
                    "gpu_name": (
                        torch.cuda.get_device_name(device) if device.type == "cuda" else None
                    ),
                    "gpu_total_memory_bytes": (
                        torch.cuda.get_device_properties(device).total_memory
                        if device.type == "cuda"
                        else None
                    ),
                },
            }
        )

        total_wall = time.perf_counter()
        total_cpu = time.process_time()
        stages = result["stages"]

        with measured_stage("data_load", device, stages):
            adata = sc.read(paths["data"])
            if adata.shape != spec.expected_shape:
                raise RuntimeError(
                    f"Dataset shape mismatch: {adata.shape} != {spec.expected_shape}"
                )

        with measured_stage("model_initialization", device, stages):
            set_all_seeds(spec.seed)
            cfg = build_lsd_config(spec.name)
            effective_walk_config = (
                cfg.walks.num_walks,
                cfg.walks.path_len,
                cfg.walks.batch_size,
            )
            expected_walk_config = (spec.num_walks, spec.path_len, spec.batch_size)
            if effective_walk_config != expected_walk_config:
                raise RuntimeError(
                    "Walk configuration mismatch: "
                    f"{effective_walk_config} != {expected_walk_config}"
                )
            lsd = LSD(adata=adata, config=cfg, device=device)
            result["effective_config"] = config_record(cfg)

        if args.walks_in is None:
            with measured_stage("prior_transition", device, stages):
                lsd.set_prior_transition(prior_time_key=spec.prior_time_key)
            with measured_stage("walk_generation", device, stages):
                lsd.prepare_walks()
        else:
            with measured_stage("fixed_walk_load", device, stages):
                walks_array = np.load(args.walks_in)
                if walks_array.ndim != 2:
                    raise RuntimeError("Fixed walks must be a two-dimensional array")
                if walks_array.shape[1] != cfg.walks.path_len:
                    raise RuntimeError(
                        f"Fixed walk length {walks_array.shape[1]} != {cfg.walks.path_len}"
                    )
                if walks_array.size and (
                    walks_array.min() < 0 or walks_array.max() >= lsd.adata.n_obs
                ):
                    raise RuntimeError("Fixed walks contain an out-of-range cell index")
                lsd.walks = torch.from_numpy(
                    np.asarray(walks_array, dtype=np.int32)
                )

        result["walks"] = {
            "shape": list(lsd.walks.shape),
            "dtype": str(lsd.walks.dtype),
            "sha256": hashlib.sha256(lsd.walks.cpu().numpy().tobytes()).hexdigest(),
            "source": "fixed" if args.walks_in is not None else "native",
        }
        if args.walks_out is not None:
            args.walks_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.walks_out, lsd.walks.cpu().numpy())

        if not args.skip_training:
            with measured_stage("training", device, stages):
                clear_pyro_state()
                lsd.train(
                    num_epochs=epochs,
                    save_dir=None,
                    plot_loss=False,
                    random_state=spec.seed,
                )

        synchronize(device)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(
            {
                "status": "ok",
                "finished_at": utc_now(),
                "total_wall_seconds": time.perf_counter() - total_wall,
                "total_cpu_seconds": time.process_time() - total_cpu,
                "peak_rss_self_bytes": int(usage.ru_maxrss * 1024),
            }
        )
        write_result(args.output, result)
        return 0
    except BaseException as exc:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(
            {
                "status": "error",
                "finished_at": utc_now(),
                "peak_rss_self_bytes": int(usage.ru_maxrss * 1024),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_result(args.output, result)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
