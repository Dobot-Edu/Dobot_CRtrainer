from __future__ import annotations

"""Convert Dobot_CRtrainer raw episodes to the LeRobot dataset format.

The recorded control space is TCP pose, not joint pose.  Therefore the
standard LeRobot features produced here are:

    observation.state = [tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper]
    action           = [target_x, target_y, target_z, target_rx, target_ry,
                        target_rz, target_gripper]

The first six values use mm and degrees.  Gripper values are normalized to
the interval [0, 1], where 1.0 means fully open and 0.0 means fully closed.
This is a TCP-action dataset and needs a CR5A TCP controller at inference
time; it is not pretending that a TCP command is a joint command.
"""

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# USER SETTINGS: edit this section, then run `python scripts/convert_to_lerobot.py`.
# Command-line options remain available and temporarily override these values.
# =============================================================================

# 原始采集数据的根目录。脚本会在其中查找
# <DATASET_DIR>/<TASK_NAME>/episodes/episode_xxx 等采集回合。
DATASET_DIR = "datasets"

# 本次转换的任务目录名，必须与 DATASET_DIR 下的实际文件夹名称一致。
TASK_NAME = "test123"

# 转换后的 LeRobot 数据集保存目录，包含 data、meta、videos 等内容。
OUTPUT_DIR = "datasets/test123/test123_lerobot"

# 写入 LeRobot 元数据的数据集标识。使用本地数据时可采用
# local/<名称>；训练脚本中的 dataset.repo_id 必须与这里保持一致。
REPO_ID = "local/dobot_cr5a_test123"

# 每一帧关联的自然语言任务描述，供 Pi0、Pi0.5、VLA 等语言条件策略使用。
# 同一任务的转换、训练和推理阶段应使用语义一致的提示词。
TASK_PROMPT = "Pick up the gear and place it at the target position."

# True：若 OUTPUT_DIR 已存在，则允许删除并重新生成；旧结果会被覆盖。
# False：若输出目录已存在则停止转换，可用于防止误删已有数据。
OVERWRITE_EXISTING = True

# =============================================================================
# CONVERSION DEFAULTS
# =============================================================================

DEFAULT_FPS = 30
DEFAULT_MAX_GAP_S = 0.25
DEFAULT_MIN_FRAMES = 20
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_MAX_CAMERA_SKEW_MS = 40.0
DEFAULT_MAX_CAMERA_AGE_MS = 60.0
DEFAULT_MAX_CAMERA_REPEAT_RATE = 0.20
DEFAULT_MAX_RESAMPLE_ERROR_MS = 15.0
DEFAULT_MAX_POSITION_LEAD_MM = 40.0
DEFAULT_MAX_ANGLE_LEAD_DEG = 10.0
DEFAULT_MIN_FEEDBACK_COVERAGE = 1.0
DEFAULT_MIN_EXECUTED_RATIO = 1.0
DEFAULT_MIN_SAFETY_CLEAN_RATIO = 1.0
# All demonstrations start near this calibrated TCP orientation. Anchor every
# episode to the same equivalent Euler branch before computing dataset stats.
DEFAULT_TCP_EULER_REFERENCE_DEG = np.asarray([180.0, 0.0, -90.0], dtype=np.float64)


@dataclass
class PreparedSegment:
    episode_dir: Path
    source_episode: str
    segment_index: int
    task: str
    frames: list[dict[str, Any]]
    resampled_indices: list[int]
    max_source_gap_s: float
    timestamp_repairs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Dobot_CRtrainer episodes into a LeRobot dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR,
        help="Raw Dobot_CRtrainer dataset root containing <task>/episodes.",
    )
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument(
        "--task-prompt",
        default=TASK_PROMPT,
        help=(
            "Optional natural-language task stored in every LeRobot frame. "
            "Use this for VLA training, for example 'Pick up the gear and place it in the tray.'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="LeRobot dataset root. Defaults to datasets/<task>/<task>_lerobot.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="LeRobot repo id stored in metadata, for example local/dobot_cr5a_pick_and_place.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=DEFAULT_MAX_GAP_S,
        help="Split an episode when adjacent raw frames exceed this gap.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="Discard segments shorter than this after resampling.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include raw episodes whose metadata valid flag is false.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Create a state/action-only dataset without RealSense RGB images.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Encode image features as LeRobot videos instead of individual images.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help=(
            "Output image width for training data. Defaults to 640. "
            "Images are letterboxed to preserve the source aspect ratio."
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help=(
            "Output image height for training data. Defaults to 480. "
            "Images are letterboxed to preserve the source aspect ratio."
        ),
    )
    parser.add_argument(
        "--camera-names",
        default="top,wrist",
        help=(
            "Comma-separated raw camera names to convert. Defaults to 'top,wrist', "
            "which produces observation.images.top and observation.images.wrist. "
            "Use 'wrist' for legacy single-camera episodes."
        ),
    )
    parser.add_argument(
        "--include-vr-inputs",
        action="store_true",
        help=(
            "Include raw VR controller position and rotation as policy inputs. "
            "Disabled by default because autonomous Dobot inference does not have live VR inputs."
        ),
    )
    parser.add_argument(
        "--max-camera-skew-ms",
        type=float,
        default=DEFAULT_MAX_CAMERA_SKEW_MS,
        help="Reject conversion when selected camera timestamps differ by more than this.",
    )
    parser.add_argument(
        "--max-camera-age-ms",
        type=float,
        default=DEFAULT_MAX_CAMERA_AGE_MS,
        help="Reject conversion when a selected RGB frame is older than this.",
    )
    parser.add_argument(
        "--max-camera-repeat-rate",
        type=float,
        default=DEFAULT_MAX_CAMERA_REPEAT_RATE,
        help="Maximum adjacent repeated-frame ratio allowed for each camera.",
    )
    parser.add_argument(
        "--max-resample-error-ms",
        type=float,
        default=DEFAULT_MAX_RESAMPLE_ERROR_MS,
        help="Maximum distance between a 30 FPS target time and its selected control frame.",
    )
    parser.add_argument(
        "--max-position-lead-mm",
        type=float,
        default=DEFAULT_MAX_POSITION_LEAD_MM,
        help="Maximum Euclidean TCP action-state position difference.",
    )
    parser.add_argument(
        "--max-angle-lead-deg",
        type=float,
        default=DEFAULT_MAX_ANGLE_LEAD_DEG,
        help="Maximum wrapped TCP action-state angle difference on any axis.",
    )
    parser.add_argument(
        "--min-feedback-coverage",
        type=float,
        default=DEFAULT_MIN_FEEDBACK_COVERAGE,
        help="Minimum fraction of selected frames containing actual gripper feedback.",
    )
    parser.add_argument(
        "--min-executed-ratio",
        type=float,
        default=DEFAULT_MIN_EXECUTED_RATIO,
        help="Minimum fraction of selected frames marked action.executed=true.",
    )
    parser.add_argument(
        "--min-safety-clean-ratio",
        type=float,
        default=DEFAULT_MIN_SAFETY_CLEAN_RATIO,
        help="Minimum fraction of selected frames with no robot error/collision and ready mode.",
    )
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="Print alignment warnings without blocking conversion. For diagnostics only.",
    )
    parser.add_argument(
        "--lerobot-src",
        default=str(ROOT / "lerobot-main" / "src"),
        help="Local LeRobot source tree. Used when the package is not installed.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=OVERWRITE_EXISTING,
        help=(
            "Delete an existing output directory before conversion. "
            "Use --no-overwrite to protect it for one run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the conversion plan without importing LeRobot or writing output.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else ROOT / resolved


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_frames(episode_dir: Path) -> list[dict[str, Any]]:
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.exists():
        raise ValueError(f"Missing frames.jsonl: {episode_dir}")
    frames: list[dict[str, Any]] = []
    with frames_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {frames_path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Frame is not an object at {frames_path}:{line_number}")
            frames.append(value)
    if not frames:
        raise ValueError(f"No frames in {episode_dir}")
    return frames


def split_on_gaps(
    frames: list[dict[str, Any]],
    max_gap_s: float,
) -> list[tuple[list[dict[str, Any]], float, int]]:
    elapsed = [float(frame.get("elapsed_s", index / 50.0)) for index, frame in enumerate(frames)]
    segments: list[list[dict[str, Any]]] = [[]]
    segment_gaps: list[float] = [0.0]
    timestamp_repairs: list[int] = [0]
    for index, frame in enumerate(frames):
        if index > 0:
            gap = elapsed[index] - elapsed[index - 1]
            if gap < -1e-6:
                raise ValueError("Frame elapsed_s moved backwards")
            if gap <= 1e-6:
                # The recorder can emit two valid samples in the same clock
                # tick. Keep both samples; LeRobot will assign uniform time.
                timestamp_repairs[-1] += 1
            if gap > max_gap_s:
                segments.append([])
                segment_gaps.append(0.0)
                timestamp_repairs.append(0)
            segment_gaps[-1] = max(segment_gaps[-1], gap)
        segments[-1].append(frame)
    return list(zip(segments, segment_gaps, timestamp_repairs, strict=True))


def resample_indices(frames: list[dict[str, Any]], fps: int) -> list[int]:
    if len(frames) == 1:
        return [0]
    elapsed = np.asarray([float(frame["elapsed_s"]) for frame in frames], dtype=np.float64)
    relative = elapsed - elapsed[0]
    target_times = np.arange(0.0, float(relative[-1]) + 1e-9, 1.0 / fps)
    indices = [int(np.argmin(np.abs(relative - target))) for target in target_times]
    # np.argmin can choose the same source frame for adjacent targets near a
    # boundary.  Keep the sequence monotonic; duplicates are valid samples.
    return indices


def unwrap_angles(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape[0] < 2:
        return result
    for index in range(1, result.shape[0]):
        delta = (result[index] - result[index - 1] + 180.0) % 360.0 - 180.0
        result[index] = result[index - 1] + delta
    return result


def anchor_angle_sequence(
    values: np.ndarray,
    reference_deg: np.ndarray = DEFAULT_TCP_EULER_REFERENCE_DEG,
) -> np.ndarray:
    """Unwrap one episode, then shift it to the shared dataset Euler branch."""
    result = unwrap_angles(values)
    reference = np.asarray(reference_deg, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != reference.shape:
        raise ValueError(
            f"Angle sequence/reference shapes do not match: {result.shape} and {reference.shape}"
        )
    if result.shape[0] == 0:
        return result
    branch_offset = 360.0 * np.round((reference - result[0]) / 360.0)
    return result + branch_offset


def align_action_angles_to_state(
    state_angles: np.ndarray,
    action_angles: np.ndarray,
) -> np.ndarray:
    """Choose the action Euler branch nearest to the same-frame robot state."""
    state = np.asarray(state_angles, dtype=np.float64)
    action = np.asarray(action_angles, dtype=np.float64)
    if state.shape != action.shape:
        raise ValueError(
            f"State/action angle shapes do not match: {state.shape} != {action.shape}"
        )
    wrapped_delta = (action - state + 180.0) % 360.0 - 180.0
    return state + wrapped_delta


def vector(value: Any, length: int, label: str) -> np.ndarray:
    if value is None or len(value) < length:
        raise ValueError(f"Missing {label} with length {length}")
    result = np.asarray([float(item) for item in value[:length]], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite values in {label}")
    return result


def gripper_value(value: Any, fallback: float = 1000.0) -> float:
    if value is None:
        value = fallback
    result = float(value)
    if not math.isfinite(result):
        result = fallback
    return float(np.clip(result / 1000.0, 0.0, 1.0))


def observed_gripper_position(
    observation: dict[str, Any],
    action: dict[str, Any],
) -> Any:
    """Read current gripper feedback, including legacy v4 recorder fields."""
    candidates = (
        observation.get("gripper_actual_position"),
        action.get("gripper_actual_position"),
        observation.get("gripper_target_position"),
        action.get("gripper_target_position"),
    )
    return next((value for value in candidates if value is not None), None)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def camera_timing(
    frame: dict[str, Any],
    camera_name: str,
) -> tuple[int | None, float | None, float | None]:
    observation = frame.get("observation", {})
    camera = observation.get("cameras", {}).get(camera_name, {})
    if isinstance(camera, dict) and camera:
        sequence = camera.get("sequence")
        timestamp = camera.get("timestamp_unix_s")
        age = camera.get("age_s")
    elif camera_name == "wrist":
        sequence = observation.get("camera_frame_sequence")
        timestamp = observation.get("camera_timestamp_unix_s")
        age = observation.get("camera_age_s")
    else:
        return None, None, None

    try:
        sequence_value = int(sequence) if sequence is not None else None
        timestamp_value = float(timestamp) if timestamp is not None else None
        age_value = float(age) if age is not None else None
    except (TypeError, ValueError):
        return None, None, None
    if timestamp_value is not None and not math.isfinite(timestamp_value):
        timestamp_value = None
    if age_value is not None and not math.isfinite(age_value):
        age_value = None
    return sequence_value, timestamp_value, age_value


def build_alignment_quality(
    segments: list[PreparedSegment],
    fps: int,
    camera_names: tuple[str, ...],
    check_cameras: bool,
) -> dict[str, Any]:
    camera_skew_ms: list[float] = []
    camera_age_ms: dict[str, list[float]] = {name: [] for name in camera_names}
    camera_timestamp_frames: dict[str, int] = {name: 0 for name in camera_names}
    camera_sequence_frames: dict[str, int] = {name: 0 for name in camera_names}
    camera_repeats: dict[str, int] = {name: 0 for name in camera_names}
    camera_transitions: dict[str, int] = {name: 0 for name in camera_names}
    resample_error_ms: list[float] = []
    position_lead_mm: list[float] = []
    angle_lead_deg: list[float] = []
    feedback_frames = 0
    executed_frames = 0
    safety_clean_frames = 0
    total_frames = 0

    for segment in segments:
        source_elapsed = np.asarray(
            [float(frame["elapsed_s"]) for frame in segment.frames],
            dtype=np.float64,
        )
        relative_elapsed = source_elapsed - source_elapsed[0]
        previous_sequences: dict[str, int | None] = {
            name: None for name in camera_names
        }
        for output_index, source_index in enumerate(segment.resampled_indices):
            frame = segment.frames[source_index]
            observation = frame.get("observation", {})
            action = frame.get("action", {})
            state_pose = vector(
                observation.get("robot_tcp_pose_mm_deg"),
                6,
                "robot TCP state",
            ).astype(np.float64)
            action_pose = vector(
                action.get("target_tcp_pose_mm_deg"),
                6,
                "target TCP action",
            ).astype(np.float64)

            target_time = output_index / fps
            resample_error_ms.append(
                abs(float(relative_elapsed[source_index]) - target_time) * 1000.0
            )
            position_lead_mm.append(
                float(np.linalg.norm(action_pose[:3] - state_pose[:3]))
            )
            angle_delta = (
                action_pose[3:6] - state_pose[3:6] + 180.0
            ) % 360.0 - 180.0
            angle_lead_deg.append(float(np.max(np.abs(angle_delta))))

            feedback = observed_gripper_position(observation, action)
            try:
                feedback_valid = feedback is not None and math.isfinite(float(feedback))
            except (TypeError, ValueError):
                feedback_valid = False
            feedback_frames += int(feedback_valid)
            executed_frames += int(bool(action.get("executed", False)))
            mode = observation.get("robot_mode")
            safety_clean_frames += int(
                int(observation.get("error_status", 0)) == 0
                and int(observation.get("collision_state", 0)) == 0
                and (mode is None or int(mode) in (5, 7))
            )

            if check_cameras:
                timestamps: list[float] = []
                for camera_name in camera_names:
                    sequence, timestamp, age = camera_timing(frame, camera_name)
                    if timestamp is not None:
                        timestamps.append(timestamp)
                        camera_timestamp_frames[camera_name] += 1
                    if sequence is not None:
                        camera_sequence_frames[camera_name] += 1
                    if age is not None:
                        camera_age_ms[camera_name].append(age * 1000.0)
                    previous = previous_sequences[camera_name]
                    if previous is not None:
                        camera_transitions[camera_name] += 1
                        camera_repeats[camera_name] += int(sequence == previous)
                    previous_sequences[camera_name] = sequence
                if len(timestamps) >= 2:
                    camera_skew_ms.append(
                        (max(timestamps) - min(timestamps)) * 1000.0
                    )
            total_frames += 1

    repeated = {}
    for camera_name in camera_names:
        transitions = camera_transitions[camera_name]
        repeated[camera_name] = {
            "repeated_transitions": camera_repeats[camera_name],
            "transitions": transitions,
            "rate": camera_repeats[camera_name] / transitions if transitions else 0.0,
        }
    return {
        "selected_frames": total_frames,
        "camera_skew_ms": numeric_summary(camera_skew_ms),
        "camera_age_ms": {
            name: numeric_summary(values) for name, values in camera_age_ms.items()
        },
        "camera_timing_coverage": {
            name: {
                "sequence": camera_sequence_frames[name] / total_frames,
                "timestamp": camera_timestamp_frames[name] / total_frames,
                "age": len(camera_age_ms[name]) / total_frames,
            }
            for name in camera_names
        },
        "camera_skew_coverage": (
            len(camera_skew_ms) / total_frames if len(camera_names) >= 2 else 1.0
        ),
        "camera_adjacent_repeats": repeated,
        "resample_error_ms": numeric_summary(resample_error_ms),
        "tcp_position_lead_mm": numeric_summary(position_lead_mm),
        "tcp_wrapped_angle_lead_deg": numeric_summary(angle_lead_deg),
        "gripper_feedback_coverage": feedback_frames / total_frames,
        "executed_action_ratio": executed_frames / total_frames,
        "safety_clean_ratio": safety_clean_frames / total_frames,
    }


def alignment_violations(
    quality: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    violations: list[str] = []

    def maximum(metric: dict[str, Any]) -> float:
        value = metric.get("max")
        return float(value) if value is not None else 0.0

    if not args.no_images:
        for camera_name, coverage in quality["camera_timing_coverage"].items():
            for field_name, ratio in coverage.items():
                if float(ratio) < 1.0:
                    violations.append(
                        f"camera '{camera_name}' {field_name} coverage "
                        f"{float(ratio):.3%} is below 100.000%"
                    )
        if float(quality["camera_skew_coverage"]) < 1.0:
            violations.append(
                "camera skew coverage "
                f"{float(quality['camera_skew_coverage']):.3%} is below 100.000%"
            )
        skew = maximum(quality["camera_skew_ms"])
        if skew > args.max_camera_skew_ms:
            violations.append(
                f"camera skew {skew:.3f}ms exceeds {args.max_camera_skew_ms:.3f}ms"
            )
        for camera_name, metric in quality["camera_age_ms"].items():
            age = maximum(metric)
            if age > args.max_camera_age_ms:
                violations.append(
                    f"camera '{camera_name}' age {age:.3f}ms exceeds "
                    f"{args.max_camera_age_ms:.3f}ms"
                )
        for camera_name, metric in quality["camera_adjacent_repeats"].items():
            rate = float(metric["rate"])
            if rate > args.max_camera_repeat_rate:
                violations.append(
                    f"camera '{camera_name}' repeat rate {rate:.3%} exceeds "
                    f"{args.max_camera_repeat_rate:.3%}"
                )

    resample_error = maximum(quality["resample_error_ms"])
    if resample_error > args.max_resample_error_ms:
        violations.append(
            f"resample error {resample_error:.3f}ms exceeds "
            f"{args.max_resample_error_ms:.3f}ms"
        )
    position_lead = maximum(quality["tcp_position_lead_mm"])
    if position_lead > args.max_position_lead_mm:
        violations.append(
            f"TCP position lead {position_lead:.3f}mm exceeds "
            f"{args.max_position_lead_mm:.3f}mm"
        )
    angle_lead = maximum(quality["tcp_wrapped_angle_lead_deg"])
    if angle_lead > args.max_angle_lead_deg:
        violations.append(
            f"TCP wrapped angle lead {angle_lead:.3f}deg exceeds "
            f"{args.max_angle_lead_deg:.3f}deg"
        )
    if quality["gripper_feedback_coverage"] < args.min_feedback_coverage:
        violations.append(
            "gripper feedback coverage "
            f"{quality['gripper_feedback_coverage']:.3%} is below "
            f"{args.min_feedback_coverage:.3%}"
        )
    if quality["executed_action_ratio"] < args.min_executed_ratio:
        violations.append(
            f"executed action ratio {quality['executed_action_ratio']:.3%} is below "
            f"{args.min_executed_ratio:.3%}"
        )
    if quality["safety_clean_ratio"] < args.min_safety_clean_ratio:
        violations.append(
            f"safety-clean ratio {quality['safety_clean_ratio']:.3%} is below "
            f"{args.min_safety_clean_ratio:.3%}"
        )
    return violations


def parse_camera_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("--camera-names must contain at least one camera name")
    if len(set(names)) != len(names):
        raise ValueError("--camera-names must not contain duplicate names")
    return names


def image_path(
    frame: dict[str, Any],
    camera_name: str,
    episode_dir: Path | None = None,
) -> Path | None:
    observation = frame.get("observation", {})
    camera = observation.get("cameras", {}).get(camera_name, {})
    raw = camera.get("rgb_path") if isinstance(camera, dict) else None
    # v3 episodes contained one primary stream only, stored under the legacy
    # fields. Retain conversion support when explicitly requested as wrist.
    if not raw and camera_name == "wrist":
        raw = observation.get("camera_rgb_path")
    if not raw:
        return None
    path = Path(str(raw))
    if episode_dir is not None and not path.is_absolute():
        path = episode_dir / path
    return path


def validate_and_prepare_segments(
    episode_root: Path,
    task: str,
    fps: int,
    max_gap_s: float,
    min_frames: int,
    include_invalid: bool,
    no_images: bool,
    camera_names: tuple[str, ...],
) -> tuple[list[PreparedSegment], dict[str, int]]:
    prepared: list[PreparedSegment] = []
    report = {
        "episodes_seen": 0,
        "episodes_used": 0,
        "episodes_skipped_invalid": 0,
        "segments_split_by_gap": 0,
        "segments_skipped_short": 0,
        "timestamp_repairs": 0,
        "frames_output": 0,
    }
    episode_dirs = sorted(path.parent for path in episode_root.glob("episode_*/metadata.json"))
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found: {episode_root}")

    for episode_dir in episode_dirs:
        report["episodes_seen"] += 1
        metadata = load_json(episode_dir / "metadata.json")
        if not include_invalid and not bool(metadata.get("valid", False)):
            report["episodes_skipped_invalid"] += 1
            continue
        frames = load_frames(episode_dir)
        for segment_index, (segment_frames, max_source_gap, timestamp_repairs) in enumerate(
            split_on_gaps(frames, max_gap_s)
        ):
            if segment_index > 0:
                report["segments_split_by_gap"] += 1
            indices = resample_indices(segment_frames, fps)
            if len(indices) < min_frames:
                report["segments_skipped_short"] += 1
                continue
            if not no_images:
                for camera_name in camera_names:
                    missing = [
                        frame_index
                        for frame_index in indices
                        if image_path(segment_frames[frame_index], camera_name, episode_dir) is None
                        or not image_path(
                            segment_frames[frame_index], camera_name, episode_dir
                        ).exists()
                    ]
                    if missing:
                        raise FileNotFoundError(
                            f"Missing RGB image for camera '{camera_name}' in {episode_dir}; "
                            f"first frame index {missing[0]}. Use --camera-names wrist for "
                            "legacy single-camera episodes."
                        )
            prepared.append(
                PreparedSegment(
                    episode_dir=episode_dir,
                    source_episode=episode_dir.name,
                    segment_index=segment_index,
                    task=str(metadata.get("task", task)),
                    frames=segment_frames,
                    resampled_indices=indices,
                    max_source_gap_s=max_source_gap,
                    timestamp_repairs=timestamp_repairs,
                )
            )
            report["timestamp_repairs"] += timestamp_repairs
            report["frames_output"] += len(indices)
        if any(item.source_episode == episode_dir.name for item in prepared):
            report["episodes_used"] += 1
    return prepared, report


def build_numeric_arrays(
    segment: PreparedSegment,
    include_vr_inputs: bool,
) -> dict[str, np.ndarray]:
    source = segment.frames
    tcp_state = []
    tcp_action = []
    joints = []
    gripper_state = []
    gripper_action = []
    vr_position = []
    vr_rotation = []
    for frame in source:
        observation = frame.get("observation", {})
        action = frame.get("action", {})
        tcp_state.append(vector(observation.get("robot_tcp_pose_mm_deg"), 6, "robot TCP state"))
        tcp_action.append(vector(action.get("target_tcp_pose_mm_deg"), 6, "target TCP action"))
        joints.append(vector(observation.get("robot_joint_positions_deg"), 6, "robot joints"))
        gripper_state.append(
            gripper_value(
                observed_gripper_position(observation, action),
                fallback=1000.0,
            )
        )
        gripper_action.append(
            gripper_value(action.get("gripper_target_position"), fallback=gripper_state[-1] * 1000.0)
        )
        if include_vr_inputs:
            vr_position.append(vector(observation.get("vr_right_position_m"), 3, "VR position"))
            vr_rotation.append(vector(observation.get("vr_right_rotation_xyzw"), 4, "VR rotation"))

    state_tcp = np.asarray(tcp_state, dtype=np.float64)
    action_tcp = np.asarray(tcp_action, dtype=np.float64)
    state_tcp[:, 3:6] = anchor_angle_sequence(state_tcp[:, 3:6])
    action_tcp[:, 3:6] = unwrap_angles(action_tcp[:, 3:6])
    action_tcp[:, 3:6] = align_action_angles_to_state(
        state_tcp[:, 3:6],
        action_tcp[:, 3:6],
    )
    arrays = {
        "state": np.concatenate(
            [state_tcp, np.asarray(gripper_state, dtype=np.float64)[:, None]], axis=1
        ).astype(np.float32),
        "action": np.concatenate(
            [action_tcp, np.asarray(gripper_action, dtype=np.float64)[:, None]], axis=1
        ).astype(np.float32),
        "joints": np.asarray(joints, dtype=np.float32),
        "gripper_state": np.asarray(gripper_state, dtype=np.float32)[:, None],
    }
    if include_vr_inputs:
        arrays["vr_position"] = np.asarray(vr_position, dtype=np.float32)
        arrays["vr_rotation"] = np.asarray(vr_rotation, dtype=np.float32)
    return arrays


def feature_schema(
    image_shapes: dict[str, tuple[int, int, int]],
    use_videos: bool,
    include_vr_inputs: bool,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["target_x_mm", "target_y_mm", "target_z_mm", "target_rx_deg", "target_ry_deg", "target_rz_deg", "target_gripper"],
        },
        "observation.robot_joints": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg"],
        },
    }
    if include_vr_inputs:
        features["observation.vr_position"] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x_m", "y_m", "z_m"],
        }
        features["observation.vr_rotation"] = {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "w"],
        }
    for camera_name, image_shape in image_shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
    return features


def load_image(path: Path, target_width: int, target_height: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to convert RGB images") from exc
    with Image.open(path) as image:
        image = image.convert("RGB")
        source_width, source_height = image.size
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        if (resized_width, resized_height) != image.size:
            image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

        # Preserve the camera geometry when converting 16:9 recordings to the
        # requested 4:3 training size. Centered black padding avoids cropping
        # objects at the workspace edges or stretching robot motion.
        canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        canvas.paste(image, (offset_x, offset_y))
        return np.asarray(canvas, dtype=np.uint8)


def import_lerobot(lerobot_src: Path):
    if lerobot_src.exists():
        sys.path.insert(0, str(lerobot_src))
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            "Cannot import LeRobotDataset. Install the LeRobot dataset dependencies "
            "in the active conda environment, or pass --lerobot-src to a compatible "
            "lerobot/src directory. Example: pip install -e lerobot-main[dataset]"
        ) from exc
    return LeRobotDataset


def convert(args: argparse.Namespace, segments: list[PreparedSegment], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    # Check the complete LeRobot stack before opening images. This avoids a
    # misleading Pillow-only error when the active environment is Python 3.11
    # or does not have torch/datasets installed yet.
    LeRobotDataset = import_lerobot(resolve_path(args.lerobot_src))

    image_shapes: dict[str, tuple[int, int, int]] = {}
    if not args.no_images:
        first_frame = segments[0].frames[segments[0].resampled_indices[0]]
        for camera_name in args.camera_names:
            first_image = load_image(
                image_path(first_frame, camera_name, segments[0].episode_dir),  # type: ignore[arg-type]
                args.image_width,
                args.image_height,
            )
            if first_image.ndim != 3 or first_image.shape[2] != 3:
                raise ValueError(f"Expected RGB HWC image, got shape {first_image.shape}")
            image_shapes[camera_name] = tuple(int(item) for item in first_image.shape)

    repo_id = args.repo_id or f"local/dobot_cr5a_{args.task}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=feature_schema(image_shapes, args.use_videos, args.include_vr_inputs),
        root=output_dir,
        robot_type="dobot_cr5a",
        use_videos=args.use_videos,
        image_writer_threads=2 if image_shapes else 0,
    )

    converted_episodes: list[dict[str, Any]] = []
    try:
        for output_episode_index, segment in enumerate(segments):
            arrays = build_numeric_arrays(segment, args.include_vr_inputs)
            for source_index in segment.resampled_indices:
                frame: dict[str, Any] = {
                    "observation.state": arrays["state"][source_index],
                    "action": arrays["action"][source_index],
                    "observation.robot_joints": arrays["joints"][source_index],
                    "task": args.task_prompt or segment.task,
                }
                if args.include_vr_inputs:
                    frame["observation.vr_position"] = arrays["vr_position"][source_index]
                    frame["observation.vr_rotation"] = arrays["vr_rotation"][source_index]
                for camera_name, image_shape in image_shapes.items():
                    raw_image = image_path(
                        segment.frames[source_index],
                        camera_name,
                        segment.episode_dir,
                    )
                    assert raw_image is not None
                    image = load_image(raw_image, args.image_width, args.image_height)
                    if tuple(image.shape) != image_shape:
                        raise ValueError(
                            f"Image shape changed for camera '{camera_name}' in "
                            f"{segment.source_episode}: expected {image_shape}, got {image.shape}"
                        )
                    frame[f"observation.images.{camera_name}"] = image
                dataset.add_frame(frame)
            dataset.save_episode()
            converted_episodes.append(
                {
                    "output_episode_index": output_episode_index,
                    "source_episode": segment.source_episode,
                    "source_segment_index": segment.segment_index,
                    "frames": len(segment.resampled_indices),
                    "max_source_gap_s": segment.max_source_gap_s,
                    "timestamp_repairs": segment.timestamp_repairs,
                }
            )
            print(
                f"[convert] episode {output_episode_index}: "
                f"{segment.source_episode} segment={segment.segment_index} "
                f"frames={len(segment.resampled_indices)}"
            )
        dataset.finalize()
    except Exception:
        # finalize closes parquet/image writers so a failed conversion does not
        # leave file handles open on Windows. The output is not reported valid.
        dataset.finalize()
        raise

    report = {
        "converter": "Dobot_CRtrainer/scripts/convert_to_lerobot.py",
        "format": "LeRobot",
        "robot_type": "dobot_cr5a",
        "task": args.task,
        "task_prompt": args.task_prompt,
        "fps": args.fps,
        "output_dir": str(output_dir),
        "repo_id": repo_id,
        "state_definition": "TCP pose [mm, deg] + normalized gripper [0=closed, 1=open]",
        "action_definition": "target TCP pose [mm, deg] + normalized gripper [0=closed, 1=open]",
        "use_videos": bool(args.use_videos),
        "image_size": {
            "width": args.image_width,
            "height": args.image_height,
            "resize_mode": "letterbox",
        },
        "camera_names": list(args.camera_names),
        "include_vr_inputs": bool(args.include_vr_inputs),
        "alignment_quality": args.alignment_quality,
        "quality_gate": {
            "passed": not args.alignment_violations,
            "violations": args.alignment_violations,
        },
        "episodes": converted_episodes,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    if args.max_gap_s <= 0.0:
        raise SystemExit("--max-gap-s must be positive")
    if args.min_frames < 1:
        raise SystemExit("--min-frames must be positive")
    if args.image_width < 1 or args.image_height < 1:
        raise SystemExit("--image-width and --image-height must be positive")
    if args.use_videos and args.no_images:
        raise SystemExit("--use-videos cannot be combined with --no-images")
    ratio_values = {
        "--max-camera-repeat-rate": args.max_camera_repeat_rate,
        "--min-feedback-coverage": args.min_feedback_coverage,
        "--min-executed-ratio": args.min_executed_ratio,
        "--min-safety-clean-ratio": args.min_safety_clean_ratio,
    }
    for name, value in ratio_values.items():
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")
    positive_thresholds = {
        "--max-camera-skew-ms": args.max_camera_skew_ms,
        "--max-camera-age-ms": args.max_camera_age_ms,
        "--max-resample-error-ms": args.max_resample_error_ms,
        "--max-position-lead-mm": args.max_position_lead_mm,
        "--max-angle-lead-deg": args.max_angle_lead_deg,
    }
    for name, value in positive_thresholds.items():
        if value <= 0.0:
            raise SystemExit(f"{name} must be positive")

    try:
        args.camera_names = parse_camera_names(args.camera_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dataset_root = resolve_path(args.dataset_dir)
    episode_root = dataset_root / args.task / "episodes"
    segments, report = validate_and_prepare_segments(
        episode_root=episode_root,
        task=args.task,
        fps=args.fps,
        max_gap_s=args.max_gap_s,
        min_frames=args.min_frames,
        include_invalid=args.include_invalid,
        no_images=args.no_images,
        camera_names=args.camera_names,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not segments:
        raise SystemExit("No usable segments found")
    args.alignment_quality = build_alignment_quality(
        segments,
        fps=args.fps,
        camera_names=args.camera_names,
        check_cameras=not args.no_images,
    )
    args.alignment_violations = alignment_violations(args.alignment_quality, args)
    print(
        "[quality] alignment="
        + json.dumps(args.alignment_quality, ensure_ascii=False)
    )
    if args.alignment_violations:
        for violation in args.alignment_violations:
            print(f"[quality WARNING] {violation}", file=sys.stderr)
        if not args.no_quality_gate:
            raise SystemExit(
                "Alignment quality gate failed. Fix the raw data or use "
                "--no-quality-gate for diagnostics only."
            )
    else:
        print("[quality] PASS")

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else dataset_root / args.task / f"{args.task}_lerobot"
    )
    print(f"[convert] output={output_dir}")
    print(
        f"[convert] planned_episodes={len(segments)} fps={args.fps} "
        f"image_size={args.image_width}x{args.image_height} "
        f"cameras={','.join(args.camera_names)} "
        f"include_vr_inputs={args.include_vr_inputs}"
    )
    if args.dry_run:
        for index, segment in enumerate(segments):
            print(
                f"[dry-run] {index}: source={segment.source_episode} "
                f"segment={segment.segment_index} frames={len(segment.resampled_indices)} "
                f"max_gap={segment.max_source_gap_s:.3f}s"
            )
        return 0

    result = convert(args, segments, output_dir)
    print(
        f"[convert] complete episodes={len(result['episodes'])} "
        f"path={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
