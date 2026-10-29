#!/usr/bin/env python3
"""Diagnose TCP angle branches and gripper state in raw and LeRobot data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONVERSION_SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(CONVERSION_SCRIPT_DIR))

from convert_to_lerobot import (  # noqa: E402
    DEFAULT_MAX_ANGLE_LEAD_DEG,
    DEFAULT_MAX_CAMERA_AGE_MS,
    DEFAULT_MAX_CAMERA_REPEAT_RATE,
    DEFAULT_MAX_CAMERA_SKEW_MS,
    DEFAULT_MAX_POSITION_LEAD_MM,
    DEFAULT_MAX_RESAMPLE_ERROR_MS,
    DEFAULT_MIN_EXECUTED_RATIO,
    DEFAULT_MIN_FEEDBACK_COVERAGE,
    DEFAULT_MIN_SAFETY_CLEAN_RATIO,
    TASK_NAME,
    alignment_violations,
    build_alignment_quality,
    validate_and_prepare_segments,
)

# =============================================================================
# USER SETTINGS
#
# Edit this section, then run:
#     python Dobot_Train/diagnose_training_data.py
#
# Command-line options remain available and temporarily override these values.
# =============================================================================

# 原始采集任务目录，其中应包含 episodes/episode_xxx/frames.jsonl。
RAW_TASK_DIR = f"datasets/{TASK_NAME}"

# convert_to_lerobot.py 生成的 LeRobot 数据集目录。
LEROBOT_DIR = f"datasets/{TASK_NAME}/{TASK_NAME}_lerobot"

# True 表示发现任何训练数据质量警告时以状态码 1 退出。
# 这样可以防止有问题的数据继续进入训练；命令行可用 --no-strict 临时关闭。
STRICT_MODE = True

# 数据采集和转换使用的目标帧率。
DIAGNOSIS_FPS = 30

# 必须存在并参与时间对齐检查的相机名称，多个名称使用英文逗号分隔。
CAMERA_NAMES = "top,wrist"

# 以下阈值与转换脚本保持一致，通常不需要单独修改。
MAX_CAMERA_SKEW_MS = DEFAULT_MAX_CAMERA_SKEW_MS
MAX_CAMERA_AGE_MS = DEFAULT_MAX_CAMERA_AGE_MS
MAX_CAMERA_REPEAT_RATE = DEFAULT_MAX_CAMERA_REPEAT_RATE
MAX_RESAMPLE_ERROR_MS = DEFAULT_MAX_RESAMPLE_ERROR_MS
MAX_POSITION_LEAD_MM = DEFAULT_MAX_POSITION_LEAD_MM
MAX_ANGLE_LEAD_DEG = DEFAULT_MAX_ANGLE_LEAD_DEG
MIN_FEEDBACK_COVERAGE = DEFAULT_MIN_FEEDBACK_COVERAGE
MIN_EXECUTED_RATIO = DEFAULT_MIN_EXECUTED_RATIO
MIN_SAFETY_CLEAN_RATIO = DEFAULT_MIN_SAFETY_CLEAN_RATIO


@dataclass
class RawStats:
    episodes: int = 0
    frames: int = 0
    missing_observation_gripper: int = 0
    legacy_action_actual_gripper: int = 0
    angle_branch_rows: int = 0
    max_raw_angle_delta: float = 0.0
    max_wrapped_angle_delta: float = 0.0
    gripper_state_values: list[float] | None = None
    gripper_action_values: list[float] | None = None


@dataclass
class ConvertedStats:
    episodes: int = 0
    frames: int = 0
    angle_branch_rows: int = 0
    max_raw_angle_delta: float = 0.0
    max_wrapped_angle_delta: float = 0.0
    gripper_state_values: list[float] | None = None
    gripper_action_values: list[float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check equivalent Euler-angle branches and gripper observation/action "
            "values before training."
        )
    )
    parser.add_argument(
        "--raw-task-dir",
        default=RAW_TASK_DIR,
        help="Raw task directory containing episodes/.",
    )
    parser.add_argument(
        "--lerobot-dir",
        default=LEROBOT_DIR,
        help="Converted LeRobot dataset directory.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=STRICT_MODE,
        help="Exit with status 1 when a training-data warning is found.",
    )
    parser.add_argument("--fps", type=int, default=DIAGNOSIS_FPS)
    parser.add_argument(
        "--camera-names",
        default=CAMERA_NAMES,
        help="Comma-separated raw cameras expected in every selected sample.",
    )
    parser.add_argument(
        "--max-camera-skew-ms", type=float, default=MAX_CAMERA_SKEW_MS
    )
    parser.add_argument(
        "--max-camera-age-ms", type=float, default=MAX_CAMERA_AGE_MS
    )
    parser.add_argument(
        "--max-camera-repeat-rate",
        type=float,
        default=MAX_CAMERA_REPEAT_RATE,
    )
    parser.add_argument(
        "--max-resample-error-ms",
        type=float,
        default=MAX_RESAMPLE_ERROR_MS,
    )
    parser.add_argument(
        "--max-position-lead-mm",
        type=float,
        default=MAX_POSITION_LEAD_MM,
    )
    parser.add_argument(
        "--max-angle-lead-deg",
        type=float,
        default=MAX_ANGLE_LEAD_DEG,
    )
    parser.add_argument(
        "--min-feedback-coverage",
        type=float,
        default=MIN_FEEDBACK_COVERAGE,
    )
    parser.add_argument(
        "--min-executed-ratio",
        type=float,
        default=MIN_EXECUTED_RATIO,
    )
    parser.add_argument(
        "--min-safety-clean-ratio",
        type=float,
        default=MIN_SAFETY_CLEAN_RATIO,
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def wrapped_angle_delta(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    return (action - state + 180.0) % 360.0 - 180.0


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def normalized_gripper(value: Any) -> float | None:
    result = finite_float(value)
    if result is None:
        return None
    return float(np.clip(result / 1000.0, 0.0, 1.0))


def read_raw_stats(raw_task_dir: Path) -> RawStats:
    episode_root = raw_task_dir / "episodes"
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Raw episode directory not found: {episode_root}")

    stats = RawStats(gripper_state_values=[], gripper_action_values=[])
    for frames_path in sorted(episode_root.glob("episode_*/frames.jsonl")):
        if frames_path.stat().st_size == 0:
            continue
        episode_used = False
        with frames_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                    observation = frame["observation"]
                    action = frame["action"]
                    state_angles = np.asarray(
                        observation["robot_tcp_pose_mm_deg"][3:6],
                        dtype=np.float64,
                    )
                    action_angles = np.asarray(
                        action["target_tcp_pose_mm_deg"][3:6],
                        dtype=np.float64,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid raw frame: {frames_path}:{line_number}: {exc}"
                    ) from exc

                raw_delta = action_angles - state_angles
                wrapped_delta = wrapped_angle_delta(action_angles, state_angles)
                stats.angle_branch_rows += int(np.any(np.abs(raw_delta) > 180.0))
                stats.max_raw_angle_delta = max(
                    stats.max_raw_angle_delta,
                    float(np.max(np.abs(raw_delta))),
                )
                stats.max_wrapped_angle_delta = max(
                    stats.max_wrapped_angle_delta,
                    float(np.max(np.abs(wrapped_delta))),
                )

                observation_gripper = observation.get("gripper_actual_position")
                legacy_gripper = action.get("gripper_actual_position")
                if observation_gripper is None:
                    stats.missing_observation_gripper += 1
                    observation_gripper = legacy_gripper
                if legacy_gripper is not None:
                    stats.legacy_action_actual_gripper += 1

                state_value = normalized_gripper(observation_gripper)
                action_value = normalized_gripper(action.get("gripper_target_position"))
                if state_value is not None:
                    stats.gripper_state_values.append(state_value)
                if action_value is not None:
                    stats.gripper_action_values.append(action_value)
                stats.frames += 1
                episode_used = True
        stats.episodes += int(episode_used)
    if stats.frames == 0:
        raise ValueError(f"No non-empty raw frames found under: {episode_root}")
    return stats


def read_converted_stats(lerobot_dir: Path) -> ConvertedStats:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to inspect converted Parquet data; "
            "activate the dobot-lerobot environment"
        ) from exc

    parquet_files = sorted((lerobot_dir / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot Parquet files found: {lerobot_dir / 'data'}")

    stats = ConvertedStats(gripper_state_values=[], gripper_action_values=[])
    episode_ids: set[int] = set()
    for path in parquet_files:
        table = pq.read_table(
            path,
            columns=["observation.state", "action", "episode_index"],
        )
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        episode = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)
        if state.ndim != 2 or action.ndim != 2 or state.shape[1] < 7 or action.shape[1] < 7:
            raise ValueError(
                f"Expected state/action arrays with at least 7 values in {path}"
            )

        raw_delta = action[:, 3:6] - state[:, 3:6]
        wrapped_delta = wrapped_angle_delta(action[:, 3:6], state[:, 3:6])
        stats.angle_branch_rows += int(
            np.any(np.abs(raw_delta) > 180.0, axis=1).sum()
        )
        stats.max_raw_angle_delta = max(
            stats.max_raw_angle_delta,
            float(np.max(np.abs(raw_delta))),
        )
        stats.max_wrapped_angle_delta = max(
            stats.max_wrapped_angle_delta,
            float(np.max(np.abs(wrapped_delta))),
        )
        stats.gripper_state_values.extend(state[:, 6].tolist())
        stats.gripper_action_values.extend(action[:, 6].tolist())
        stats.frames += len(state)
        episode_ids.update(int(value) for value in episode)
    stats.episodes = len(episode_ids)
    return stats


def describe_gripper(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return "missing"
    closed = int((array <= 0.35).sum())
    opened = int((array >= 0.65).sum())
    middle = int(array.size - closed - opened)
    return (
        f"min={array.min():.3f} max={array.max():.3f} "
        f"closed={closed} middle={middle} open={opened}"
    )


def main() -> int:
    args = parse_args()
    raw_task_dir = resolve(args.raw_task_dir)
    lerobot_dir = resolve(args.lerobot_dir)
    camera_names = tuple(
        name.strip() for name in args.camera_names.split(",") if name.strip()
    )
    if not camera_names:
        print("Diagnosis failed: --camera-names cannot be empty", file=sys.stderr)
        return 2

    try:
        raw = read_raw_stats(raw_task_dir)
        converted = read_converted_stats(lerobot_dir)
        segments, _ = validate_and_prepare_segments(
            episode_root=raw_task_dir / "episodes",
            task=raw_task_dir.name,
            fps=args.fps,
            max_gap_s=0.25,
            min_frames=20,
            include_invalid=False,
            no_images=True,
            camera_names=camera_names,
        )
        alignment = build_alignment_quality(
            segments,
            fps=args.fps,
            camera_names=camera_names,
            check_cameras=True,
        )
        args.no_images = False
        timing_warnings = alignment_violations(alignment, args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Diagnosis failed: {exc}", file=sys.stderr)
        return 2

    print(f"Raw task: {raw_task_dir}")
    print(f"  episodes={raw.episodes} frames={raw.frames}")
    print(
        "  angle action-state: "
        f"branch_rows={raw.angle_branch_rows} "
        f"max_raw={raw.max_raw_angle_delta:.3f}deg "
        f"max_wrapped={raw.max_wrapped_angle_delta:.3f}deg"
    )
    print(
        "  observation gripper field: "
        f"missing={raw.missing_observation_gripper}/{raw.frames} "
        f"legacy_action_feedback={raw.legacy_action_actual_gripper}/{raw.frames}"
    )
    print(f"  gripper state:  {describe_gripper(raw.gripper_state_values or [])}")
    print(f"  gripper action: {describe_gripper(raw.gripper_action_values or [])}")
    print("  selected-sample alignment:")
    print(
        "    camera skew ms: "
        f"p95={alignment['camera_skew_ms']['p95']:.3f} "
        f"max={alignment['camera_skew_ms']['max']:.3f}"
    )
    for camera_name in camera_names:
        age = alignment["camera_age_ms"][camera_name]
        repeat = alignment["camera_adjacent_repeats"][camera_name]
        coverage = alignment["camera_timing_coverage"][camera_name]
        print(
            f"    {camera_name}: age_p95={age['p95']:.3f}ms "
            f"age_max={age['max']:.3f}ms "
            f"repeat_rate={repeat['rate']:.3%} "
            f"timing_coverage="
            f"{min(float(value) for value in coverage.values()):.3%}"
        )
    print(
        "    resample error ms: "
        f"p95={alignment['resample_error_ms']['p95']:.3f} "
        f"max={alignment['resample_error_ms']['max']:.3f}"
    )
    print(
        "    TCP lead: "
        f"position_max={alignment['tcp_position_lead_mm']['max']:.3f}mm "
        f"wrapped_angle_max="
        f"{alignment['tcp_wrapped_angle_lead_deg']['max']:.3f}deg"
    )
    print(
        "    coverage: "
        f"gripper={alignment['gripper_feedback_coverage']:.3%} "
        f"executed={alignment['executed_action_ratio']:.3%} "
        f"safety_clean={alignment['safety_clean_ratio']:.3%}"
    )

    print(f"LeRobot dataset: {lerobot_dir}")
    print(f"  episodes={converted.episodes} frames={converted.frames}")
    print(
        "  angle action-state: "
        f"branch_rows={converted.angle_branch_rows} "
        f"max_raw={converted.max_raw_angle_delta:.3f}deg "
        f"max_wrapped={converted.max_wrapped_angle_delta:.3f}deg"
    )
    print(
        f"  gripper state:  {describe_gripper(converted.gripper_state_values or [])}"
    )
    print(
        f"  gripper action: {describe_gripper(converted.gripper_action_values or [])}"
    )

    warnings: list[str] = []
    if converted.angle_branch_rows:
        warnings.append(
            "Converted action/state contains equivalent Euler branches separated "
            "by more than 180 degrees."
        )
    state_values = np.asarray(converted.gripper_state_values, dtype=np.float64)
    action_values = np.asarray(converted.gripper_action_values, dtype=np.float64)
    state_constant = state_values.size > 0 and float(np.ptp(state_values)) < 1e-6
    action_changes = action_values.size > 0 and float(np.ptp(action_values)) > 0.5
    if state_constant and action_changes:
        warnings.append(
            "Converted gripper observation is constant while gripper actions "
            "contain both open and closed commands."
        )
    warnings.extend(timing_warnings)

    if warnings:
        print("Result: WARNING")
        for warning in warnings:
            print(f"  - {warning}")
        return 1 if args.strict else 0

    print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
