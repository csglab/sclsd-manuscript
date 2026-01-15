"""Dataclass-based configuration objects for the LSD model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class LogCoshActivation(nn.Module):
    """Smooth activation that mirrors the notebook definition."""

    def forward(self, x):  # type: ignore[override]
        return torch.log(torch.cosh(x))


ActivationSpec = Union[str, nn.Module, type, None]


def resolve_activation(spec: ActivationSpec) -> nn.Module:
    """Map a string/Module/class into a concrete activation instance."""

    if isinstance(spec, nn.Module):
        return spec
    if spec is None:
        return nn.Identity()
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec()

    if isinstance(spec, str):
        key = spec.lower()
        if key == "logcosh":
            return LogCoshActivation()
        if key == "relu":
            return nn.ReLU()
        if key in {"softplus", "sp"}:
            return nn.Softplus()
        if key in {"identity", "id"}:
            return nn.Identity()
        raise ValueError(f"Unknown activation keyword: {spec}")

    raise TypeError(f"Unsupported activation spec: {spec}")


@dataclass
class LayerDims:
    B_decoder: List[int] = field(default_factory=lambda: [32, 64])
    z_decoder: List[int] = field(default_factory=lambda: [128, 256])
    x_encoder: List[int] = field(default_factory=lambda: [512, 256])
    z_encoder: List[int] = field(default_factory=lambda: [64, 32])
    xl_encoder: List[int] = field(default_factory=lambda: [64, 32])
    potential: List[int] = field(default_factory=lambda: [32, 32])
    potential_af: ActivationSpec = "logcosh"

    def as_dict(self) -> Dict[str, Any]:
        dims = {
            "B_decoder": list(self.B_decoder),
            "z_decoder": list(self.z_decoder),
            "x_encoder": list(self.x_encoder),
            "z_encoder": list(self.z_encoder),
            "xl_encoder": list(self.xl_encoder),
            "potential": list(self.potential),
        }
        dims["potential_af"] = resolve_activation(self.potential_af)
        return dims


@dataclass
class AdamConfig:
    lr: float = 1e-3
    eta_min: float = 1e-5
    T_0: int = 40
    T_mult: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "eta_min": self.eta_min,
            "T_0": self.T_0,
            "T_mult": self.T_mult,
        }


@dataclass
class KLScheduleConfig:
    af: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_af": self.af,
            "max_af": self.af,
            "max_epoch": 100,
        }


@dataclass
class WassersteinScheduleConfig:
    min_W: float = 1e-4
    max_W: float = 1e-3
    max_epoch: int = 50

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_W": self.min_W,
            "max_W": self.max_W,
            "max_epoch": self.max_epoch,
        }


@dataclass
class OptimizerConfig:
    adam: AdamConfig = field(default_factory=AdamConfig)
    kl_schedule: KLScheduleConfig = field(default_factory=KLScheduleConfig)
    wasserstein_schedule: WassersteinScheduleConfig = field(default_factory=WassersteinScheduleConfig)


@dataclass
class WalkConfig:
    batch_size: int = 256
    path_len: int = 10
    num_walks: int = 4096
    random_state: int = 42


@dataclass
class ModelConfig:
    z_dim: int = 10
    B_dim: int = 2
    V_coeff: float = 0.01
    layer_dims: LayerDims = field(default_factory=LayerDims)


@dataclass
class LSDConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    walks: WalkConfig = field(default_factory=WalkConfig)


def default_config() -> LSDConfig:
    """Convenience helper mirroring the prior defaults."""
    return LSDConfig()
