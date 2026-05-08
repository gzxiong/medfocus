"""YAML config loaders.

Plain dataclasses (no pydantic dependency) for portability — the schema is
small enough that a hand-written ingestion is simpler than pulling in a
validation library.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ----------------------------- schema models --------------------------------

@dataclass
class ModelSpec:
    hf_id: str
    family: str
    dtype: str = "bfloat16"
    image_token: Optional[str] = None


@dataclass
class ModelsConfig:
    models: dict[str, ModelSpec]


@dataclass
class DatasetSpec:
    annotations: Optional[str] = None
    images_glob: Optional[str] = None
    images_dir: Optional[str] = None
    studies_json: Optional[str] = None
    direct_json: Optional[str] = None
    reasoning_json: Optional[str] = None


@dataclass
class DatasetsConfig:
    data_root: str
    datasets: dict[str, DatasetSpec]
    image: dict[str, Any] = field(default_factory=lambda: {"resize_width": 224})
    question_suffixes: dict[str, str] = field(default_factory=dict)


@dataclass
class OTConfig:
    epsilon: float = 0.05
    lambda_marginal: float = 0.1
    selection_grid: int = 14
    transfer_grid: int = 56
    sinkhorn_max_iter: int = 500
    sinkhorn_tol: float = 1e-6
    mass_quantile: float = 0.75


@dataclass
class InterventionConfig:
    baseline: str = "zero"
    tau: float = 0.75


@dataclass
class MedSAMConfig:
    model_id: str = "flaviagiammarino/medsam-vit-base"


@dataclass
class MedFocusConfig:
    ot: OTConfig
    medsam: MedSAMConfig
    intervention: InterventionConfig
    concepts: list[str]
    composites: dict[str, list[str]]


@dataclass
class RadEditConfig:
    radedit: dict[str, Any]
    prompts: dict[str, str]


@dataclass
class ReferencePoolConfig:
    images_dir: str
    masks_dir: str
    candidates: list[Any] = field(default_factory=list)


@dataclass
class AppConfig:
    models: ModelsConfig
    datasets: DatasetsConfig
    medfocus: MedFocusConfig
    radedit: Optional[RadEditConfig] = None
    reference_pool: Optional[ReferencePoolConfig] = None


# ------------------------------- helpers ------------------------------------

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(obj: Any) -> Any:
    """Substitute ${ENV_VAR} (with optional ${VAR:-default}) inside any nested object."""
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) or m.group(0))
        return _ENV_RE.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return _expand_env(yaml.safe_load(f) or {})


# ------------------------------- loaders ------------------------------------

def load_models(path: Path) -> ModelsConfig:
    raw = _read_yaml(path)
    models = {k: ModelSpec(**v) for k, v in (raw.get("models") or {}).items()}
    return ModelsConfig(models=models)


def load_datasets(path: Path) -> DatasetsConfig:
    raw = _read_yaml(path)
    datasets = {k: DatasetSpec(**(v or {})) for k, v in (raw.get("datasets") or {}).items()}
    return DatasetsConfig(
        data_root=raw.get("data_root", ""),
        datasets=datasets,
        image=raw.get("image") or {"resize_width": 224},
        question_suffixes=raw.get("question_suffixes") or {},
    )


def load_medfocus(path: Path) -> MedFocusConfig:
    raw = _read_yaml(path)
    return MedFocusConfig(
        ot=OTConfig(**(raw.get("ot") or {})),
        medsam=MedSAMConfig(**(raw.get("medsam") or {})),
        intervention=InterventionConfig(**(raw.get("intervention") or {})),
        concepts=list(raw.get("concepts") or []),
        composites={k: list(v) for k, v in (raw.get("composites") or {}).items()},
    )


def load_radedit(path: Path) -> RadEditConfig:
    raw = _read_yaml(path)
    return RadEditConfig(
        radedit=raw.get("radedit") or {},
        prompts=raw.get("prompts") or {},
    )


def load_reference_pool(path: Path) -> ReferencePoolConfig:
    raw = _read_yaml(path)
    return ReferencePoolConfig(
        images_dir=raw.get("images_dir", ""),
        masks_dir=raw.get("masks_dir", ""),
        candidates=list(raw.get("candidates") or []),
    )


def load_config(configs_dir: str | Path | None = None) -> AppConfig:
    """Merge all five YAMLs in `configs_dir` (default: ./configs) into AppConfig."""
    configs_dir = Path(configs_dir) if configs_dir else Path("configs")
    return AppConfig(
        models=load_models(configs_dir / "models.yaml"),
        datasets=load_datasets(configs_dir / "datasets.yaml"),
        medfocus=load_medfocus(configs_dir / "medfocus.yaml"),
        radedit=load_radedit(configs_dir / "radedit.yaml")
        if (configs_dir / "radedit.yaml").exists() else None,
        reference_pool=load_reference_pool(configs_dir / "reference_pool.yaml")
        if (configs_dir / "reference_pool.yaml").exists() else None,
    )
