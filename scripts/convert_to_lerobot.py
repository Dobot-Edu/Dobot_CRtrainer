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
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter
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

# 本次转换读取的原始数据任务名。该模块保持独立，拆分使用时可单独修改。
TASK_NAME = "cosmetics_quality"

# 转换后的 LeRobot 数据集保存目录，包含 data、meta、videos 等内容。
OUTPUT_DIR = f"datasets/{TASK_NAME}/{TASK_NAME}_lerobot"

# 写入 LeRobot 元数据的数据集标识。使用本地数据时可采用
# local/<名称>；训练脚本中的 dataset.repo_id 必须与这里保持一致。
REPO_ID = f"local/dobot_cr5a_{TASK_NAME}"

# 每一帧关联的自然语言任务描述，供 Pi0、Pi0.5、VLA 等语言条件策略使用。
# 同一任务的转换、训练和推理阶段应使用语义一致的提示词。
TASK_PROMPT = (
    "Place one cosmetic jar from the tray on the right onto the electronic scale "
    "for weighing. Wait until the weight reading stabilizes, then sort the jar "
    "according to the indicator light: if the light is green, place the jar in "
    "the lower-left tray; if the light is yellow, place it in the upper-left tray."
)

# True：裸命令运行时，根据夹爪事件和最终放置位置自动写入下方五条阶段标签。
# 这些提示词必须与 scripts/Dobot_inference.py 的分阶段提示词逐字一致。
ENABLE_COSMETICS_STAGE_LABELS = True

COSMETICS_STAGE_1_TASK = (
    "Pick up one cosmetic jar from the tray on the right and place it on the "
    "electronic scale. The destination for this stage is the electronic scale."
)
COSMETICS_STAGE_2_TASKS = {
    "GREEN": (
        "The indicator light is GREEN. GREEN means the destination is the LOWER-LEFT tray. "
        "Pick up the cosmetic jar from the electronic scale and prepare to move it toward "
        "the LOWER-LEFT tray. Do not move it to the UPPER-LEFT tray."
    ),
    "YELLOW": (
        "The indicator light is YELLOW. YELLOW means the destination is the UPPER-LEFT tray. "
        "Pick up the cosmetic jar from the electronic scale and prepare to move it toward "
        "the UPPER-LEFT tray. Do not move it to the LOWER-LEFT tray."
    ),
}
COSMETICS_STAGE_3_TASKS = {
    "GREEN": (
        "The indicator light is GREEN. GREEN always corresponds to the LOWER-LEFT tray. "
        "Move the cosmetic jar from the electronic scale to the LOWER-LEFT tray and release "
        "it in the LOWER-LEFT tray. Do not place it in the UPPER-LEFT tray."
    ),
    "YELLOW": (
        "The indicator light is YELLOW. YELLOW always corresponds to the UPPER-LEFT tray. "
        "Move the cosmetic jar from the electronic scale to the UPPER-LEFT tray and release "
        "it in the UPPER-LEFT tray. Do not place it in the LOWER-LEFT tray."
    ),
}

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
DEFAULT_VIDEO_ENCODING_BATCH_SIZE = 3
DEFAULT_MAX_CAMERA_SKEW_MS = 40.0
DEFAULT_MAX_CAMERA_AGE_MS = 60.0
DEFAULT_MAX_CAMERA_REPEAT_RATE = 0.20
DEFAULT_MAX_RESAMPLE_ERROR_MS = 15.0
DEFAULT_MAX_POSITION_LEAD_MM = 40.0
DEFAULT_MAX_ANGLE_LEAD_DEG = 10.0
DEFAULT_MIN_FEEDBACK_COVERAGE = 1.0
DEFAULT_MIN_EXECUTED_RATIO = 1.0
DEFAULT_MIN_SAFETY_CLEAN_RATIO = 1.0
DEFAULT_GRIPPER_OPEN_THRESHOLD = 0.5
DEFAULT_GRIPPER_DEBOUNCE_FRAMES = 5
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 20260903
DEFAULT_AUDIT_EPISODES = 10
DEFAULT_MIN_DESTINATION_CLUSTER_GAP_MM = 100.0
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


@dataclass
class CosmeticsSegmentLabel:
    source_episode: str
    segment_index: int
    color: str
    destination: str
    split: str
    first_grasp_source_frame: int
    scale_release_source_frame: int
    second_grasp_source_frame: int
    final_release_source_frame: int
    destination_pose_mm_deg: list[float]
    destination_cluster_y_mm: float
    output_episode_index: int = -1


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
        "--cosmetics-stage-labels",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_COSMETICS_STAGE_LABELS,
        help=(
            "Assign one of five cosmetics task prompts per frame from debounced gripper "
            "events and the demonstrated final tray destination. Keeps each trajectory "
            "as one continuous episode. Use --no-cosmetics-stage-labels for a fixed prompt."
        ),
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=DEFAULT_GRIPPER_OPEN_THRESHOLD,
        help="Normalized gripper target at or above this value is treated as open.",
    )
    parser.add_argument(
        "--gripper-debounce-frames",
        type=int,
        default=DEFAULT_GRIPPER_DEBOUNCE_FRAMES,
        help="Number of consecutive raw control frames required to confirm a gripper transition.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Per-color fraction reserved at the end of the output episode order for evaluation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Reproducible seed for the color-stratified train/eval episode manifest.",
    )
    parser.add_argument(
        "--audit-episodes",
        type=int,
        default=DEFAULT_AUDIT_EPISODES,
        help="Number of randomly selected episodes exported as stage-boundary audit tables/images.",
    )
    parser.add_argument(
        "--min-destination-cluster-gap-mm",
        type=float,
        default=DEFAULT_MIN_DESTINATION_CLUSTER_GAP_MM,
        help="Minimum separation in robot Y between demonstrated upper/lower final tray clusters.",
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
        "--video-encoding-batch-size",
        type=int,
        default=DEFAULT_VIDEO_ENCODING_BATCH_SIZE,
        help=(
            "Number of episodes whose camera videos are encoded concurrently. "
            "With two cameras, the default 3 runs up to 6 FFmpeg workers. "
            "Use 1 for the original per-episode behavior."
        ),
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


def commanded_gripper_value(frame: dict[str, Any]) -> float:
    observation = frame.get("observation", {})
    action = frame.get("action", {})
    value = action.get("gripper_target_position")
    if value is None:
        value = observation.get("gripper_target_position")
    if value is None and action.get("gripper_desired_closed") is not None:
        value = 0.0 if bool(action["gripper_desired_closed"]) else 1000.0
    return gripper_value(value)


def debounced_gripper_events(
    frames: list[dict[str, Any]],
    open_threshold: float,
    debounce_frames: int,
) -> list[tuple[int, bool, bool]]:
    """Return (run_start, old_open, new_open) after persistence filtering."""
    states = [commanded_gripper_value(frame) >= open_threshold for frame in frames]
    if not states:
        return []
    stable = states[0]
    candidate: bool | None = None
    candidate_start = 0
    candidate_count = 0
    events: list[tuple[int, bool, bool]] = []
    for index, state in enumerate(states[1:], start=1):
        if state == stable:
            candidate = None
            candidate_count = 0
            continue
        if candidate is None or candidate != state:
            candidate = state
            candidate_start = index
            candidate_count = 1
        else:
            candidate_count += 1
        if candidate_count >= debounce_frames:
            events.append((candidate_start, stable, state))
            stable = state
            candidate = None
            candidate_count = 0
    return events


def cosmetics_stage_and_task(
    label: CosmeticsSegmentLabel,
    source_frame_index: int,
) -> tuple[str, str]:
    if source_frame_index <= label.scale_release_source_frame:
        return "stage_1_right_tray_to_scale", COSMETICS_STAGE_1_TASK
    if source_frame_index < label.second_grasp_source_frame:
        return "stage_2_observe_and_regrasp", COSMETICS_STAGE_2_TASKS[label.color]
    return "stage_3_scale_to_target_tray", COSMETICS_STAGE_3_TASKS[label.color]


def nearest_resampled_frame(segment: PreparedSegment, source_frame_index: int) -> int:
    values = np.asarray(segment.resampled_indices, dtype=np.int64)
    return int(np.argmin(np.abs(values - source_frame_index)))


def prepare_cosmetics_stage_labels(
    segments: list[PreparedSegment],
    args: argparse.Namespace,
) -> tuple[list[PreparedSegment], dict[tuple[str, int], CosmeticsSegmentLabel], dict[str, Any]]:
    source_counts = Counter(segment.source_episode for segment in segments)
    split_sources = sorted(source for source, count in source_counts.items() if count != 1)
    if split_sources:
        raise ValueError(
            "Cosmetics stage labeling requires one continuous segment per source episode; "
            f"found split episodes: {split_sources[:5]}"
        )

    records: list[tuple[PreparedSegment, list[tuple[int, bool, bool]], list[float]]] = []
    expected = [(True, False), (False, True), (True, False), (False, True)]
    for segment in segments:
        events = debounced_gripper_events(
            segment.frames,
            open_threshold=args.gripper_open_threshold,
            debounce_frames=args.gripper_debounce_frames,
        )
        pattern = [(old_open, new_open) for _, old_open, new_open in events]
        if pattern != expected:
            readable = [
                f"{'open' if old else 'closed'}->{'open' if new else 'closed'}@{index}"
                for index, old, new in events
            ]
            raise ValueError(
                f"Expected exactly four debounced gripper transitions in "
                f"{segment.source_episode}, got {readable}"
            )
        final_release = events[3][0]
        pose = vector(
            segment.frames[final_release].get("action", {}).get("target_tcp_pose_mm_deg"),
            6,
            f"final release TCP target in {segment.source_episode}",
        ).astype(float).tolist()
        records.append((segment, events, pose))

    destination_y = np.asarray([pose[1] for _, _, pose in records], dtype=np.float64)
    order = np.argsort(destination_y)
    ordered_y = destination_y[order]
    gaps = np.diff(ordered_y)
    if gaps.size == 0:
        raise ValueError("At least two cosmetics episodes are required to infer tray destinations")
    gap_index = int(np.argmax(gaps))
    cluster_gap = float(gaps[gap_index])
    if cluster_gap < args.min_destination_cluster_gap_mm:
        raise ValueError(
            f"Final tray Y clusters are separated by only {cluster_gap:.3f} mm; "
            f"expected at least {args.min_destination_cluster_gap_mm:.3f} mm"
        )
    threshold_y = float((ordered_y[gap_index] + ordered_y[gap_index + 1]) / 2.0)

    labels: dict[tuple[str, int], CosmeticsSegmentLabel] = {}
    by_color: dict[str, list[PreparedSegment]] = {"GREEN": [], "YELLOW": []}
    for segment, events, pose in records:
        # Site calibration verified from top-camera final-placement frames:
        # higher robot-base Y is the LOWER-LEFT tray; lower Y is UPPER-LEFT.
        color = "GREEN" if pose[1] > threshold_y else "YELLOW"
        destination = "LOWER-LEFT" if color == "GREEN" else "UPPER-LEFT"
        label = CosmeticsSegmentLabel(
            source_episode=segment.source_episode,
            segment_index=segment.segment_index,
            color=color,
            destination=destination,
            split="",
            first_grasp_source_frame=events[0][0],
            scale_release_source_frame=events[1][0],
            second_grasp_source_frame=events[2][0],
            final_release_source_frame=events[3][0],
            destination_pose_mm_deg=pose,
            destination_cluster_y_mm=float(pose[1]),
        )
        labels[(segment.source_episode, segment.segment_index)] = label
        by_color[color].append(segment)

    green_count = len(by_color["GREEN"])
    yellow_count = len(by_color["YELLOW"])
    if green_count != yellow_count:
        raise ValueError(
            f"Color balance requirement failed: GREEN={green_count}, YELLOW={yellow_count}"
        )

    rng = random.Random(args.split_seed)
    for values in by_color.values():
        rng.shuffle(values)
    eval_per_color = max(1, round(green_count * args.validation_fraction))
    if eval_per_color >= green_count:
        raise ValueError("Validation fraction leaves no training episodes per color")
    train_segments: list[PreparedSegment] = []
    eval_segments: list[PreparedSegment] = []
    for color in ("GREEN", "YELLOW"):
        eval_segments.extend(by_color[color][:eval_per_color])
        train_segments.extend(by_color[color][eval_per_color:])
    rng.shuffle(train_segments)
    rng.shuffle(eval_segments)
    ordered_segments = train_segments + eval_segments
    for output_index, segment in enumerate(ordered_segments):
        label = labels[(segment.source_episode, segment.segment_index)]
        label.output_episode_index = output_index
        label.split = "validation" if output_index >= len(train_segments) else "train"

    cluster_counts = {
        "GREEN": green_count,
        "YELLOW": yellow_count,
    }
    summary = {
        "enabled": True,
        "method": "debounced_gripper_events+final_destination_y_cluster+camera_audit",
        "gripper_open_threshold": args.gripper_open_threshold,
        "gripper_debounce_frames": args.gripper_debounce_frames,
        "destination_y_threshold_mm": threshold_y,
        "destination_cluster_gap_mm": cluster_gap,
        "color_episode_counts": cluster_counts,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "train_episode_counts": {
            color: sum(
                labels[(segment.source_episode, segment.segment_index)].split == "train"
                for segment in by_color[color]
            )
            for color in ("GREEN", "YELLOW")
        },
        "validation_episode_counts": {
            color: sum(
                labels[(segment.source_episode, segment.segment_index)].split == "validation"
                for segment in by_color[color]
            )
            for color in ("GREEN", "YELLOW")
        },
        "task_texts": [
            COSMETICS_STAGE_1_TASK,
            COSMETICS_STAGE_2_TASKS["GREEN"],
            COSMETICS_STAGE_2_TASKS["YELLOW"],
            COSMETICS_STAGE_3_TASKS["GREEN"],
            COSMETICS_STAGE_3_TASKS["YELLOW"],
        ],
    }
    return ordered_segments, labels, summary


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


def write_cosmetics_label_artifacts(
    output_dir: Path,
    segments: list[PreparedSegment],
    labels: dict[tuple[str, int], CosmeticsSegmentLabel],
    args: argparse.Namespace,
    task_frame_counts: Counter[str],
    task_episode_counts: Counter[str],
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    total_frames = sum(task_frame_counts.values())
    stats = {
        "total_frames": total_frames,
        "total_episodes": len(segments),
        "tasks": [
            {
                "task": task,
                "frames": frames,
                "frame_fraction": frames / total_frames if total_frames else 0.0,
                "episodes": task_episode_counts[task],
                "episode_fraction": task_episode_counts[task] / len(segments),
            }
            for task, frames in task_frame_counts.items()
        ],
        "stage_frame_counts": {
            "stage_1_right_tray_to_scale": task_frame_counts[COSMETICS_STAGE_1_TASK],
            "stage_2_observe_and_regrasp": sum(
                task_frame_counts[task] for task in COSMETICS_STAGE_2_TASKS.values()
            ),
            "stage_3_scale_to_target_tray": sum(
                task_frame_counts[task] for task in COSMETICS_STAGE_3_TASKS.values()
            ),
        },
        "color_episode_counts": {
            color: sum(label.color == color for label in labels.values())
            for color in ("GREEN", "YELLOW")
        },
        "color_frame_counts": {
            color: sum(
                len(segment.resampled_indices)
                for segment in segments
                if labels[(segment.source_episode, segment.segment_index)].color == color
            )
            for color in ("GREEN", "YELLOW")
        },
    }
    (meta_dir / "segment_label_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_manifest: dict[str, Any] = {
        "seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "train": {},
        "validation": {},
    }
    for split in ("train", "validation"):
        for color in ("GREEN", "YELLOW"):
            split_manifest[split][color.lower()] = [
                {
                    "episode_index": label.output_episode_index,
                    "source_episode": label.source_episode,
                }
                for label in sorted(labels.values(), key=lambda value: value.output_episode_index)
                if label.split == split and label.color == color
            ]
    split_manifest["train_episode_indices"] = sorted(
        label.output_episode_index for label in labels.values() if label.split == "train"
    )
    split_manifest["validation_episode_indices"] = sorted(
        label.output_episode_index for label in labels.values() if label.split == "validation"
    )
    (meta_dir / "train_eval_split.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    label_path = meta_dir / "segment_labels.jsonl"
    with label_path.open("w", encoding="utf-8") as handle:
        for label in sorted(labels.values(), key=lambda value: value.output_episode_index):
            handle.write(json.dumps(label.__dict__, ensure_ascii=False) + "\n")

    audit_count = min(args.audit_episodes, len(segments))
    rng = random.Random(args.split_seed + 1)
    by_color = {
        color: [
            segment
            for segment in segments
            if labels[(segment.source_episode, segment.segment_index)].color == color
        ]
        for color in ("GREEN", "YELLOW")
    }
    green_audit = audit_count // 2
    yellow_audit = audit_count - green_audit
    selected = rng.sample(by_color["GREEN"], green_audit) + rng.sample(
        by_color["YELLOW"], yellow_audit
    )
    rng.shuffle(selected)

    audit_dir = meta_dir / "segment_label_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    for segment in selected:
        label = labels[(segment.source_episode, segment.segment_index)]
        stage_2_start = next(
            index
            for index, source_index in enumerate(segment.resampled_indices)
            if source_index > label.scale_release_source_frame
        )
        stage_3_start = next(
            index
            for index, source_index in enumerate(segment.resampled_indices)
            if source_index >= label.second_grasp_source_frame
        )
        interesting = {
            0,
            nearest_resampled_frame(segment, label.first_grasp_source_frame),
            max(0, stage_2_start - 1),
            stage_2_start,
            (stage_2_start + stage_3_start) // 2,
            stage_3_start,
            nearest_resampled_frame(segment, label.final_release_source_frame),
            len(segment.resampled_indices) - 1,
        }
        for output_frame_index in sorted(interesting):
            source_frame_index = segment.resampled_indices[output_frame_index]
            frame = segment.frames[source_frame_index]
            stage, task = cosmetics_stage_and_task(label, source_frame_index)
            observed = observed_gripper_position(
                frame.get("observation", {}), frame.get("action", {})
            )
            audit_rows.append(
                {
                    "episode_index": label.output_episode_index,
                    "source_episode": label.source_episode,
                    "frame_index": output_frame_index,
                    "source_frame_index": source_frame_index,
                    "stage": stage,
                    "color": label.color,
                    "task": task,
                    "commanded_gripper": commanded_gripper_value(frame),
                    "observed_gripper": (
                        gripper_value(observed) if observed is not None else None
                    ),
                }
            )

        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("Pillow is required to write cosmetics audit images") from exc
        moments = [
            ("scale_release", nearest_resampled_frame(segment, label.scale_release_source_frame)),
            ("indicator", (stage_2_start + stage_3_start) // 2),
            ("second_grasp", stage_3_start),
            ("final_release", nearest_resampled_frame(segment, label.final_release_source_frame)),
        ]
        tile_width, tile_height, header_height = 240, 180, 44
        canvas = Image.new("RGB", (tile_width * len(moments), header_height + tile_height * 2), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (6, 4),
            f"episode={label.output_episode_index} {label.source_episode} "
            f"color={label.color} target={label.destination}",
            fill="black",
        )
        for column, (moment, output_frame_index) in enumerate(moments):
            source_frame_index = segment.resampled_indices[output_frame_index]
            frame = segment.frames[source_frame_index]
            draw.text((column * tile_width + 5, 24), f"{moment} f={output_frame_index}", fill="black")
            for row, camera_name in enumerate(("top", "wrist")):
                raw_path = image_path(frame, camera_name, segment.episode_dir)
                if raw_path is None or not raw_path.exists():
                    continue
                with Image.open(raw_path) as raw_image:
                    tile = raw_image.convert("RGB")
                    tile.thumbnail((tile_width, tile_height))
                    x = column * tile_width + (tile_width - tile.width) // 2
                    y = header_height + row * tile_height + (tile_height - tile.height) // 2
                    canvas.paste(tile, (x, y))
        canvas.save(
            audit_dir / f"episode-{label.output_episode_index:06d}_{label.color.lower()}.jpg",
            quality=90,
        )

    audit_csv = audit_dir / "frame_stage_audit.csv"
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    print(
        "[labels] artifacts="
        f"{meta_dir / 'segment_labels.jsonl'} split={meta_dir / 'train_eval_split.json'} "
        f"audits={audit_dir}"
    )


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
        batch_encoding_size=args.video_encoding_batch_size if args.use_videos else 1,
        # Video metadata is filled after each parallel batch. Keep every
        # episode record buffered until finalize() so it can be updated safely.
        metadata_buffer_size=max(10, len(segments) + 1),
    )

    converted_episodes: list[dict[str, Any]] = []
    task_frame_counts: Counter[str] = Counter()
    task_episode_counts: Counter[str] = Counter()
    try:
        for output_episode_index, segment in enumerate(segments):
            arrays = build_numeric_arrays(segment, args.include_vr_inputs)
            label = args.cosmetics_labels.get(
                (segment.source_episode, segment.segment_index)
            )
            episode_tasks: set[str] = set()
            for source_index in segment.resampled_indices:
                if label is not None:
                    _, frame_task = cosmetics_stage_and_task(label, source_index)
                else:
                    frame_task = args.task_prompt or segment.task
                frame: dict[str, Any] = {
                    "observation.state": arrays["state"][source_index],
                    "action": arrays["action"][source_index],
                    "observation.robot_joints": arrays["joints"][source_index],
                    "task": frame_task,
                }
                task_frame_counts[frame_task] += 1
                episode_tasks.add(frame_task)
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
            for episode_task in episode_tasks:
                task_episode_counts[episode_task] += 1
            label_report = None
            if label is not None:
                label_report = {
                    "color": label.color,
                    "destination": label.destination,
                    "split": label.split,
                    "first_grasp_source_frame": label.first_grasp_source_frame,
                    "scale_release_source_frame": label.scale_release_source_frame,
                    "second_grasp_source_frame": label.second_grasp_source_frame,
                    "final_release_source_frame": label.final_release_source_frame,
                }
            converted_episodes.append(
                {
                    "output_episode_index": output_episode_index,
                    "source_episode": segment.source_episode,
                    "source_segment_index": segment.segment_index,
                    "frames": len(segment.resampled_indices),
                    "max_source_gap_s": segment.max_source_gap_s,
                    "timestamp_repairs": segment.timestamp_repairs,
                    "segment_label": label_report,
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

    if args.cosmetics_stage_labels:
        write_cosmetics_label_artifacts(
            output_dir,
            segments,
            args.cosmetics_labels,
            args,
            task_frame_counts,
            task_episode_counts,
        )

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
        "video_encoding_batch_size": (
            args.video_encoding_batch_size if args.use_videos else 1
        ),
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
        "cosmetics_stage_labeling": args.cosmetics_label_summary,
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
    if args.video_encoding_batch_size < 1:
        raise SystemExit("--video-encoding-batch-size must be positive")
    if not 0.0 < args.gripper_open_threshold < 1.0:
        raise SystemExit("--gripper-open-threshold must be in (0, 1)")
    if args.gripper_debounce_frames < 1:
        raise SystemExit("--gripper-debounce-frames must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be in (0, 0.5)")
    if args.audit_episodes < 10 and args.cosmetics_stage_labels:
        raise SystemExit("--audit-episodes must be at least 10 for cosmetics stage labels")
    if args.min_destination_cluster_gap_mm <= 0.0:
        raise SystemExit("--min-destination-cluster-gap-mm must be positive")
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
    args.cosmetics_labels = {}
    args.cosmetics_label_summary = {"enabled": False}
    if args.cosmetics_stage_labels:
        try:
            segments, args.cosmetics_labels, args.cosmetics_label_summary = (
                prepare_cosmetics_stage_labels(segments, args)
            )
        except ValueError as exc:
            raise SystemExit(f"Cosmetics stage labeling failed: {exc}") from exc
        print(
            "[labels] summary="
            + json.dumps(args.cosmetics_label_summary, ensure_ascii=False)
        )
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
    max_ffmpeg_workers = (
        args.video_encoding_batch_size * len(args.camera_names) if args.use_videos else 0
    )
    print(
        f"[convert] planned_episodes={len(segments)} fps={args.fps} "
        f"image_size={args.image_width}x{args.image_height} "
        f"cameras={','.join(args.camera_names)} "
        f"include_vr_inputs={args.include_vr_inputs} "
        f"video_batch={args.video_encoding_batch_size if args.use_videos else 1} "
        f"max_ffmpeg_workers={max_ffmpeg_workers}"
    )
    if args.dry_run:
        for index, segment in enumerate(segments):
            label = args.cosmetics_labels.get((segment.source_episode, segment.segment_index))
            label_text = ""
            if label is not None:
                label_text = (
                    f" color={label.color} split={label.split} "
                    f"bounds={label.first_grasp_source_frame},"
                    f"{label.scale_release_source_frame},"
                    f"{label.second_grasp_source_frame},"
                    f"{label.final_release_source_frame}"
                )
            print(
                f"[dry-run] {index}: source={segment.source_episode} "
                f"segment={segment.segment_index} frames={len(segment.resampled_indices)} "
                f"max_gap={segment.max_source_gap_s:.3f}s{label_text}"
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
