from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from vr_teleop.core.config import load_project_config
from vr_teleop.collection.vr_teleop import DEFAULT_DATASET_DIR


def main() -> None:
    config = load_project_config(ROOT / "config")
    collection = config.get("data_collection", {})

    parser = argparse.ArgumentParser(description="Inspect Dobot_CRtrainer episodes.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--validate-npz", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir is not None else Path(DEFAULT_DATASET_DIR)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    base = dataset_dir
    if args.task:
        base = base / args.task

    metadata_files = sorted(base.glob("**/metadata.json")) if base.exists() else []
    if not metadata_files:
        print(f"No episodes found under: {base}")
        return

    total_frames = 0
    total_duration = 0.0
    valid_count = 0
    print(f"Dataset: {base}")
    for metadata_path in metadata_files:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = int(metadata.get("frame_count", 0))
        duration = float(metadata.get("duration_s", 0.0))
        valid = bool(metadata.get("valid", False))
        total_frames += frames
        total_duration += duration
        valid_count += int(valid)
        npz_status = _validate_npz(metadata_path.parent) if args.validate_npz else "not_checked"
        print(
            f"- {metadata_path.parent.name}: task={metadata.get('task')} "
            f"frames={frames} duration={duration:.2f}s valid={valid} "
            f"status={metadata.get('status')} npz={npz_status}"
        )

    print(
        f"Summary: episodes={len(metadata_files)} valid={valid_count} "
        f"frames={total_frames} duration={total_duration:.2f}s"
    )


def _validate_npz(episode_dir: Path) -> str:
    path = episode_dir / "data.npz"
    if not path.exists():
        return "missing"
    try:
        import numpy as np

        with np.load(path) as data:
            frame_count = len(data["frame_index"])
            if data["action_target_tcp_pose_mm_deg"].shape != (frame_count, 6):
                return "invalid_shape"
        return "ok"
    except Exception as exc:
        return f"error:{exc}"


if __name__ == "__main__":
    main()
