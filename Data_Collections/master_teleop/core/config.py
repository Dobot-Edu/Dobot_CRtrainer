from __future__ import annotations

from pathlib import Path

from vr_teleop.core.config import load_project_config, load_yaml


MASTER_TELEOP_ROOT = Path(__file__).resolve().parents[1]
MASTER_RUNTIME_CONFIG = MASTER_TELEOP_ROOT / "config" / "master_runtime.yaml"


def load_master_project_config(config_dir: str | Path | None = None) -> dict:
    """Load shared CR5A config, then overlay master-only configuration."""

    config = load_project_config(config_dir)
    master_path = MASTER_TELEOP_ROOT / "config" / "master_hand.yaml"
    config.update(load_yaml(master_path))
    if MASTER_RUNTIME_CONFIG.exists():
        config = _deep_merge(config, load_yaml(MASTER_RUNTIME_CONFIG))
    master_safety = config.get("master_hand", {}).get("safety", {})
    if master_safety:
        if not isinstance(master_safety, dict):
            raise ValueError("master_hand.safety must be a mapping")
        config["safety"] = _deep_merge(config["safety"], master_safety)
    return config


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
