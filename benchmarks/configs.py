"""Exact manuscript configurations used by the end-to-end benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class BenchmarkSpec:
    """Immutable provenance and training settings for one manuscript workflow."""

    name: str
    display_name: str
    notebook_path: str
    notebook_sha256: str
    data_path: str
    data_sha256: str
    expected_shape: Tuple[int, int]
    prior_time_key: str
    seed: int
    epochs: int
    num_walks: int
    path_len: int
    batch_size: int
    config_overrides: Dict[str, Any]


SPECS: Dict[str, BenchmarkSpec] = {
    "bone_marrow": BenchmarkSpec(
        name="bone_marrow",
        display_name="Bone Marrow",
        notebook_path="Notebooks/BoneMarrow/train.ipynb",
        notebook_sha256=(
            "1299802b4e316014a5dbdf9a1de51c229d962efc06e2549f37beb5e111506c51"
        ),
        data_path="Zenodo/BoneMarrow/preprocessed_adata.h5ad",
        data_sha256=(
            "97e9e740362ae7d38a145a0a5970734f20b61effce27d9ff4782132196135773"
        ),
        expected_shape=(5292, 2000),
        prior_time_key="palantir_pseudotime",
        seed=42,
        epochs=200,
        num_walks=4096,
        path_len=10,
        batch_size=256,
        config_overrides={"optimizer.kl_schedule.af": 2},
    ),
    "cancer": BenchmarkSpec(
        name="cancer",
        display_name="Cancer",
        notebook_path="Notebooks/Cancer/train.ipynb",
        notebook_sha256=(
            "a62f7c23c706a0a110105a0ce347fd8c98ee7a54043acdeb4b489105ec0935b4"
        ),
        data_path="Zenodo/KP_tracer/preprocessed_adata.h5ad",
        data_sha256=(
            "86272debc454c59a54fe323c4a0534acea1f99ea5f170fe660b3e019dc1c7c91"
        ),
        expected_shape=(33179, 1999),
        prior_time_key="transferred_phylo_pseudotime",
        seed=42,
        epochs=100,
        num_walks=8192,
        path_len=8,
        batch_size=256,
        config_overrides={
            "model.V_coeff": 1e-4,
            "model.layer_dims.x_encoder": [512, 256, 128],
            "model.layer_dims.potential": [16, 16],
            "optimizer.adam.lr": 2e-3,
            "optimizer.wasserstein_schedule.max_W": 1e-2,
            "optimizer.kl_schedule.af": 2,
            "walks.path_len": 8,
            "walks.num_walks": 8192,
        },
    ),
}


def get_spec(name: str) -> BenchmarkSpec:
    """Return a benchmark specification by its command-line name."""
    try:
        return SPECS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(SPECS))
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {choices}") from exc


def build_lsd_config(name: str):
    """Construct the exact ``LSDConfig`` encoded in the audited notebook."""
    from sclsd.core import LSDConfig

    cfg = LSDConfig()
    if name == "bone_marrow":
        cfg.optimizer.kl_schedule.af = 2
        cfg.walks.random_state = 42
    elif name == "cancer":
        cfg.model.V_coeff = 1e-4
        cfg.model.layer_dims.x_encoder = [512, 256, 128]
        cfg.model.layer_dims.potential = [16, 16]
        cfg.optimizer.adam.lr = 2e-3
        cfg.optimizer.wasserstein_schedule.max_W = 1e-2
        cfg.optimizer.kl_schedule.af = 2
        cfg.walks.path_len = 8
        cfg.walks.num_walks = 8192
        cfg.walks.random_state = 42
    else:
        get_spec(name)
        raise AssertionError("Unreachable")
    return cfg


def resolved_paths(manuscript_root: Path, spec: BenchmarkSpec) -> Dict[str, Path]:
    """Resolve the notebook and data paths relative to the repository root."""
    return {
        "notebook": manuscript_root / spec.notebook_path,
        "data": manuscript_root / spec.data_path,
    }
