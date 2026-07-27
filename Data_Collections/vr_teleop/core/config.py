from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_project_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    base = Path(config_dir) if config_dir else PROJECT_ROOT / "config"
    merged: dict[str, Any] = {}
    for name in (
        "network.yaml",
        "host_network.yaml",
        "robot.yaml",
        "gripper.yaml",
        "camera.yaml",
        "safety.yaml",
        "vr_mapping.yaml",
        "data_collection.yaml",
    ):
        path = base / name
        if path.exists():
            merged.update(load_yaml(path))
    return merged
