from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from master_teleop.collection.launch_nodes import DATASET_TASK_NAME

# =============================================================================
# 用户配置：修改后直接运行 `python Dobot_Train/inspect_dataset.py`。
# =============================================================================

# 待检查的原始采集数据集目录。相对路径以项目根目录为基准。
# 目录内应包含 episodes/episode_<时间戳>/metadata.json 和 data.npz。
RAW_DATASET_PATH = f"datasets/{DATASET_TASK_NAME}"

# True：同时读取每个 episode 的 data.npz 并检查帧数和 action 形状。
# False：只汇总 metadata.json，检查速度更快。
VALIDATE_NPZ_BY_DEFAULT = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Dobot_CRtrainer episodes.")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Temporarily override RAW_DATASET_PATH for this run.",
    )
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument(
        "--validate-npz",
        dest="validate_npz",
        action="store_true",
        help="Validate every episode data.npz file.",
    )
    validation.add_argument(
        "--no-validate-npz",
        dest="validate_npz",
        action="store_false",
        help="Only summarize episode metadata.json files.",
    )
    parser.set_defaults(validate_npz=VALIDATE_NPZ_BY_DEFAULT)
    args = parser.parse_args()

    base = Path(args.dataset_path or RAW_DATASET_PATH).expanduser()
    if not base.is_absolute():
        base = ROOT / base
    base = base.resolve()

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
