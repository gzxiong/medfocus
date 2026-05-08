"""Verify the YAML configs parse into the Pydantic schemas."""

from __future__ import annotations

import os
from pathlib import Path

from medfocus.config import (
    load_datasets,
    load_medfocus,
    load_models,
    load_radedit,
    load_reference_pool,
)


def test_models_yaml_parses(repo_root: Path):
    cfg = load_models(repo_root / "configs" / "models.yaml")
    assert "qwen2_5_vl_3b" in cfg.models
    assert cfg.models["medgemma1_5_4b"].family == "gemma"


def test_medfocus_yaml_concepts(repo_root: Path):
    cfg = load_medfocus(repo_root / "configs" / "medfocus.yaml")
    assert len(cfg.concepts) == 11
    assert "cardiac silhouette" in cfg.concepts
    assert set(cfg.composites["bilateral_lungs"]) == {"left lung", "right lung"}
    assert cfg.intervention.tau == 0.75
    assert cfg.ot.transfer_grid == 56


def test_datasets_yaml_parses(repo_root: Path, monkeypatch):
    monkeypatch.setenv("MEDFOCUS_DATA_ROOT", "/tmp/fake")
    cfg = load_datasets(repo_root / "configs" / "datasets.yaml")
    assert cfg.data_root == "/tmp/fake"
    assert "imagenome" in cfg.datasets


def test_radedit_yaml_parses(repo_root: Path):
    cfg = load_radedit(repo_root / "configs" / "radedit.yaml")
    assert "radedit" in cfg.radedit
    assert "foreground" in cfg.prompts


def test_reference_pool_yaml_parses(repo_root: Path):
    cfg = load_reference_pool(repo_root / "configs" / "reference_pool.yaml")
    assert cfg.images_dir.endswith("images")
