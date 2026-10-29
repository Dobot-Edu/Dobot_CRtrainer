from __future__ import annotations

from pathlib import Path
from typing import Any

if __name__ == "__main__" and not __package__:
    raise SystemExit(
        "This is an internal collection module, not a command-line entry point.\n"
        "Run data collection from the project root with:\n"
        "  python Data_Collections/vr_teleop/collection/start_vr_teleop.py"
    )

from ..recording.dataset_recorder import EpisodeRecorder


# ============================================================
# 默认数据集输出目录
#
# 日常采集只需要修改这里，不需要修改 YAML。
# 相对路径以 project_root（Dobot_CRtrainer 项目根目录）为基准；
# 也可以直接填写绝对路径，例如：
# DEFAULT_DATASET_DIR = r"D:\\RobotData\\Dobot_CRtrainer"
# ============================================================
DEFAULT_DATASET_DIR: str | Path = "datasets"

# 默认任务名称。日常采集只需要修改这里，不需要在命令行添加 --task。
DEFAULT_TASK = "test123"


def create_vr_teleop_collection(
    config: dict[str, Any],
    project_root: str | Path,
    control_period_s: float,
    dataset_dir: str | Path | None = None,
    task: str | None = None,
) -> EpisodeRecorder:
    """Build the VR teleoperation collection module.

    The default output directory is intentionally defined in this module so
    data location changes do not require editing the project YAML files.
    """
    if control_period_s <= 0.0:
        raise ValueError("control_period_s must be positive")

    root = Path(project_root).resolve()
    collection = config.get("data_collection", {})
    output_dir = Path(dataset_dir) if dataset_dir is not None else Path(DEFAULT_DATASET_DIR)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    robot = config.get("robot", {})

    return EpisodeRecorder(
        output_dir=output_dir,
        project_name=str(collection.get("project_name", "Dobot_CRtrainer")),
        task=str(task if task is not None else DEFAULT_TASK),
        control_hz=1.0 / control_period_s,
        record_button=str(collection.get("record_button", "button_b")),
        flush_every_frames=int(collection.get("flush_every_frames", 10)),
        min_episode_frames=int(collection.get("min_episode_frames", 10)),
        save_npz=bool(collection.get("save_npz", True)),
        config_snapshot=config,
        robot_name=str(robot.get("name", "dobot_cr5a")),
        image_jpeg_quality=int(collection.get("image_jpeg_quality", 90)),
        image_queue_size=int(collection.get("image_queue_size", 120)),
    )
