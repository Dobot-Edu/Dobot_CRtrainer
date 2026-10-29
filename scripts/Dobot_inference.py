#!/usr/bin/env python3
"""Run a selected LeRobot checkpoint on the Dobot CR5A.

Edit SELECTED_MODEL and TASK_MODE below. Fixed mode uses TASK_PROMPT; segmented
cosmetics mode selects one of five training prompts at runtime. The checkpoint
config determines the actual policy class, required cameras and preprocessing.
Only checkpoints producing this project's 7-D TCP action contract may command
the real robot.
"""

from __future__ import annotations

import math
import json
import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import ActionQueue, LatencyTracker, RTCConfig

from vr_teleop.control.safety_supervisor import SafetySupervisor
from vr_teleop.core.config import load_project_config
from vr_teleop.devices.camera import CameraConfig, CameraGroup, create_camera
from vr_teleop.devices.dobot_adapter import DobotTcpAdapter, MockRobotAdapter
from vr_teleop.devices.gripper_controller import GripperConfig, TriggerGripperController


# =============================================================================
# 用户配置：修改本区域后运行 `python scripts/Dobot_inference.py`
# =============================================================================

# 选择 MODEL_REGISTRY 中注册的模型名称。该名称决定加载哪个策略类型、
# checkpoint 和相机映射，不是 checkpoint 文件夹本身。
SELECTED_MODEL = "PI0.5"

# 本次推理加载的训练任务名。该模块保持独立，拆分使用时可单独修改。
INFERENCE_TASK_NAME = "cosmetics_quality"

# 单张 RTX 4090 默认只使用 GPU 0。INFERENCE_DEVICE 可填写 "cuda"、
# "cuda:0" 或 "cpu"；正式 Pi0.5 推理建议保持 CUDA。
CUDA_VISIBLE_DEVICES = "0"
INFERENCE_DEVICE = "cuda"

# Pi0.5 checkpoint 会从 outputs/train/<INFERENCE_TASK_NAME>/ 中读取。

# "fixed" 始终使用 TASK_PROMPT，适用于普通单提示词任务。
# "segmented_cosmetics" 使用下方 5 条训练提示词和阶段状态机。
TASK_MODE = "segmented_cosmetics"

# TASK_MODE="fixed" 时使用的单条任务提示词。
TASK_PROMPT = (
    "Place one cosmetic jar from the tray on the right onto the electronic scale "
    "for weighing. Wait until the weight reading stabilizes, then sort the jar "
    "according to the indicator light: if the light is green, place the jar in "
    "the lower-left tray; if the light is yellow, place it in the upper-left tray."
)

# TASK_MODE="segmented_cosmetics" 时的任务提示词。必须与分阶段数据集完全一致。
STAGE_1_PROMPT = (
    "Pick up one cosmetic jar from the tray on the right and place it on the "
    "electronic scale. The destination for this stage is the electronic scale."
)
GREEN_STAGE_2_PROMPT = (
    "The indicator light is GREEN. GREEN means the destination is the LOWER-LEFT tray. "
    "Pick up the cosmetic jar from the electronic scale and prepare to move it toward "
    "the LOWER-LEFT tray. Do not move it to the UPPER-LEFT tray."
)
YELLOW_STAGE_2_PROMPT = (
    "The indicator light is YELLOW. YELLOW means the destination is the UPPER-LEFT tray. "
    "Pick up the cosmetic jar from the electronic scale and prepare to move it toward "
    "the UPPER-LEFT tray. Do not move it to the LOWER-LEFT tray."
)
GREEN_STAGE_3_PROMPT = (
    "The indicator light is GREEN. GREEN always corresponds to the LOWER-LEFT tray. "
    "Move the cosmetic jar from the electronic scale to the LOWER-LEFT tray and release "
    "it in the LOWER-LEFT tray. Do not place it in the UPPER-LEFT tray."
)
YELLOW_STAGE_3_PROMPT = (
    "The indicator light is YELLOW. YELLOW always corresponds to the UPPER-LEFT tray. "
    "Move the cosmetic jar from the electronic scale to the UPPER-LEFT tray and release "
    "it in the UPPER-LEFT tray. Do not place it in the LOWER-LEFT tray."
)

# 是否把模型输出发送给真实机械臂。False 使用初始位姿模拟状态，只检查
# 相机、模型输出、实时预览和视频录制；True 会连接并控制真实机械臂。
EXECUTE = True

# EXECUTE=True 时是否主动发送机械臂使能命令。机械臂已经使能时也可保留 True。
ENABLE_ROBOT = True

# 是否根据模型输出控制真实夹爪。只有 EXECUTE 和本参数都为 True，且项目
# 配置中启用了夹爪，才会实际执行夹爪动作。
EXECUTE_GRIPPER = True

# 推理循环总步数。实际运行时长约为 STEPS × 机械臂伺服周期。
STEPS = 15000

# True：完成一罐后回到现有推理初始位姿、重置分阶段状态并继续下一罐，
# 不再受 STEPS 限制；按 Ctrl+C 停止。False：完成一罐后正常退出。
CONTINUOUS_TASKS = True

# 最终松爪后等待夹爪动作稳定，再返回推理初始位姿。仅影响任务衔接，
# 不改变机械臂、夹爪或安全限幅参数。
CONTINUOUS_RESTART_DELAY_S = 0.5

# Dobot 全局运动速度百分比，范围为 1～100；数值越大，机械臂动作越快。
SPEED_FACTOR_PERCENT = 80

# 保留 restore_initial_pose.py 使用的标准初始关节不变。真机推理启动时，
# 在创建 RTC 动作队列之前先用 MovJ 移动到训练数据 100 条轨迹的
# 代表性起点。该关节组合是数据集中真实记录过的可达点。
MOVE_TO_INFERENCE_START = True
INFERENCE_START_JOINTS_DEG = (84.700, -3.868, 92.203, -0.170, -92.352, -105.913)
INFERENCE_START_POSE_MM_DEG = (88.876, -458.845, 480.432, 176.223, -0.783, -79.901)
INFERENCE_START_VELOCITY_PERCENT = 30
INFERENCE_START_ACCELERATION_PERCENT = 30
INFERENCE_START_TOLERANCE_DEG = 1.0
INFERENCE_START_TIMEOUT_S = 60.0

# 单步位置/姿态限制统一读取 config/safety.yaml 中的 max_step_mm 和
# max_step_deg，同时受该文件中的每秒速度上限约束。

# Real-Time Chunking 在同一进程的后台线程预测新 action chunk；主线程继续
# 按 robot.yaml 的 ServoP 周期执行队列动作，无需启动第二个推理进程。
RTC_ENABLED = True
RTC_INITIAL_EXECUTION_HORIZON = 24
RTC_BLEND_STEPS_AFTER_DELAY = 8
RTC_MAX_GUIDANCE_WEIGHT = 10.0
RTC_QUEUE_THRESHOLD = 30
RTC_WARMUP_INFERENCES = 2
RTC_PREFILL_TIMEOUT_S = 120.0
RTC_QUEUE_EMPTY_ABORT_S = 0.5

# 分阶段模式：连续多少个推理步命令相同，才确认一次夹爪开/合转换。
PHASE_GRIPPER_DEBOUNCE_STEPS = 5

# 秤上第一次释放后先等待灯色稳定，再连续确认多帧。
INDICATOR_SETTLE_S = 0.50
INDICATOR_STABLE_FRAMES = 5
INDICATOR_TIMEOUT_S = 10.0
INDICATOR_MIN_COLOR_FRACTION = 0.004
INDICATOR_MIN_DOMINANCE = 1.8

# (x1, y1, x2, y2) 为相对图像尺寸的 ROI。这两个区域由当前数据集的
# 10 条分阶段审计图确认：顶部相机为主，腕部相机为辅。真机相机移位后需重新标定。
INDICATOR_ROIS = {
    "top": (0.28, 0.00, 0.52, 0.25),
    "wrist": (0.00, 0.42, 0.42, 1.00),
}

# TCP 绝对姿态相对于标定初始姿态允许偏离的最大范围，依次对应
# (Rx, Ry, Rz)，单位为度。它用于阻止欧拉角分支错误导致腕部持续旋转，
# 同时保留训练示范中正常的姿态变化。
ORIENTATION_LIMIT_DEG = (8.0, 10.0, 12.0)

# 是否实时显示顶部和腕部相机的拼接推理画面。
SHOW_CAMERA = True

# 是否把相机画面和当前预测动作录制为 MP4，输出到 outputs/inference_videos。
RECORD_VIDEO = True

# 实时预览模式：
# "auto"：检测到本地/X11 显示环境时创建窗口；
# "window"：强制要求创建 GUI 窗口，显示不可用时会报错；
# "off"：不创建预览窗口。RECORD_VIDEO=True 时仍可继续录制视频。
PREVIEW_MODE = "auto"

# 预览画面的显示目标：
# "host"：始终显示在机械臂主机连接的显示器上，包括通过 SSH 启动时；
# "session"：使用当前终端继承的 DISPLAY/WAYLAND_DISPLAY，适合 X11 转发。
PREVIEW_TARGET = "host"

# PREVIEW_TARGET="host" 时使用的主机 X11 显示编号，可通过 `echo $DISPLAY`
# 或查看主机图形会话确认；当前 ":1" 表示使用 X Server 1。
HOST_DESKTOP_DISPLAY = ":1"

# 归一化夹爪预测值小于或等于此阈值时执行关闭。数据约定为 0=关闭、1=打开。
GRIPPER_CLOSE_THRESHOLD = 0.35

# 归一化夹爪预测值大于或等于此阈值时执行打开；处于两个阈值之间时保持
# 当前状态，形成迟滞区，避免预测值在边界附近波动造成夹爪频繁开合。
GRIPPER_OPEN_THRESHOLD = 0.65

# Model name -> local checkpoint. `policy_type` is optional validation only;
# the actual class is always read from the checkpoint config.json.
# `camera_bindings` maps a checkpoint image suffix to a physical camera name.
# The legacy ACT checkpoints used top-camera data under the old "wrist" key.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "ACT": {
        "policy_type": "act",
        "checkpoint": "outputs/train/act_pick_gears_gpu_20000/checkpoints/020000/pretrained_model",
        "camera_bindings": {"wrist": "top"},
    },
    "Pi0": {
        "policy_type": "pi0",
        "checkpoint": "outputs/train/pi0_test_pi0_gpu_bs1_w8_3000/checkpoints/003000/pretrained_model",
        "camera_bindings": {"top": "top", "wrist": "wrist"},
    },
    "PI0.5": {
        "policy_type": "pi05",
        "checkpoint": (
            f"outputs/train/{INFERENCE_TASK_NAME}/"
            "checkpoints/030000/pretrained_model"
        ),
        "camera_bindings": {"top": "top", "wrist": "wrist"},
    },
    # Add future LeRobot checkpoints here. Examples:
    # "act_dual_camera": {
    #     "checkpoint": "outputs/train/.../checkpoints/020000/pretrained_model",
    #     "camera_bindings": {"top": "top", "wrist": "wrist"},
    # },
    # "pi0_dual_camera": {
    #     "checkpoint": "outputs/train/.../checkpoints/020000/pretrained_model",
    #     "camera_bindings": {"top": "top", "wrist": "wrist"},
    # },
}


def die(message: str) -> None:
    raise SystemExit(f"[Dobot_inference] {message}")


def wrap_delta_deg(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def normalize_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def move_to_inference_start(robot: DobotTcpAdapter, safety: SafetySupervisor) -> None:
    target = [float(value) for value in INFERENCE_START_JOINTS_DEG]
    if len(target) != 6:
        die("INFERENCE_START_JOINTS_DEG must contain six joint values")
    if not 1 <= INFERENCE_START_VELOCITY_PERCENT <= 100:
        die("INFERENCE_START_VELOCITY_PERCENT must be in [1, 100]")
    if not 1 <= INFERENCE_START_ACCELERATION_PERCENT <= 100:
        die("INFERENCE_START_ACCELERATION_PERCENT must be in [1, 100]")
    if INFERENCE_START_TOLERANCE_DEG <= 0 or INFERENCE_START_TIMEOUT_S <= 0:
        die("Inference-start tolerance and timeout must be positive")

    state = robot.state()
    if (
        not state.connected
        or state.feedback_age_s > safety.feedback_timeout_s
        or len(state.q_actual_deg) < 6
    ):
        die("Robot feedback is unavailable before inference-start MovJ")
    if state.error_status or state.collision_state:
        die(
            "Robot safety state blocks inference-start MovJ: "
            f"error={state.error_status} collision={state.collision_state}"
        )

    def joint_error() -> float:
        return max(
            abs(float(state.q_actual_deg[index]) - target[index])
            for index in range(6)
        )

    error = joint_error()
    print(
        "inference_start: "
        f"current_joints={[round(float(v), 3) for v in state.q_actual_deg[:6]]} "
        f"target_joints={target} max_error_deg={error:.3f}",
        flush=True,
    )
    if error <= INFERENCE_START_TOLERANCE_DEG:
        print("inference_start_ready: already at target", flush=True)
        return

    print(
        "inference_start_movj: "
        f"acceleration={INFERENCE_START_ACCELERATION_PERCENT}% "
        f"velocity={INFERENCE_START_VELOCITY_PERCENT}%",
        flush=True,
    )
    robot.move_j_joint(
        target,
        acceleration=INFERENCE_START_ACCELERATION_PERCENT,
        velocity=INFERENCE_START_VELOCITY_PERCENT,
    )
    deadline = time.monotonic() + INFERENCE_START_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(0.2)
        state = robot.state()
        if (
            not state.connected
            or state.feedback_age_s > safety.feedback_timeout_s
            or len(state.q_actual_deg) < 6
        ):
            die("Robot feedback was lost during inference-start MovJ")
        if state.error_status or state.collision_state:
            die(
                "Inference-start MovJ stopped by robot state: "
                f"error={state.error_status} collision={state.collision_state}"
            )
        error = joint_error()
        if error <= INFERENCE_START_TOLERANCE_DEG:
            print(
                "inference_start_ready: "
                f"joints={[round(float(v), 3) for v in state.q_actual_deg[:6]]} "
                f"pose={[round(float(v), 3) for v in state.pose_mm_deg]} "
                f"max_error_deg={error:.3f}",
                flush=True,
            )
            time.sleep(0.3)
            return
    robot.stop()
    die(
        "Timed out while moving to inference start: "
        f"max_error_deg={error:.3f} timeout_s={INFERENCE_START_TIMEOUT_S:.1f}"
    )


def guard_orientation(
    action: np.ndarray,
    reference_pose: list[float],
) -> tuple[np.ndarray, list[str]]:
    guarded = action.copy()
    corrections: list[str] = []
    for offset, limit in enumerate(ORIENTATION_LIMIT_DEG):
        index = 3 + offset
        reference = float(reference_pose[index])
        raw_delta = wrap_delta_deg(float(action[index]), reference)
        safe_delta = float(np.clip(raw_delta, -limit, limit))
        guarded[index] = normalize_deg(reference + safe_delta)
        if not math.isclose(raw_delta, safe_delta, abs_tol=1e-6):
            axis = ("Rx", "Ry", "Rz")[offset]
            corrections.append(
                f"{axis}:{float(action[index]):.2f}->{guarded[index]:.2f}"
            )
    return guarded, corrections


def image_tensor(frame: Any, expected_shape: tuple[int, int, int]) -> torch.Tensor:
    channels, target_h, target_w = expected_shape
    if channels != 3:
        die(f"Only three-channel camera features are supported, got {expected_shape}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for live camera inference") from exc
    bgr = frame.color_bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source_h, source_w = rgb.shape[:2]
    scale = min(target_w / source_w, target_h / source_h)
    resized_w, resized_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    resized = cv2.resize(
        rgb,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x, y = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float() / 255.0


def load_policy(model: dict[str, Any]) -> tuple[Any, Any, Any, Path]:
    path = (ROOT / str(model["checkpoint"])).resolve()
    if not path.is_dir():
        die(f"Checkpoint not found: {path}")
    config = PreTrainedConfig.from_pretrained(path)
    expected_type = model.get("policy_type")
    if expected_type and config.type != expected_type:
        die(f"Registry says policy_type={expected_type}, checkpoint says {config.type}")
    if RTC_ENABLED:
        if config.type not in {"pi0", "pi0_fast", "pi05", "smolvla"}:
            die(f"RTC is not supported for policy type {config.type!r}")
        chunk_size = int(config.chunk_size)
        if chunk_size < 2:
            die(f"RTC requires chunk_size >= 2, got {chunk_size}")
        config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=min(RTC_INITIAL_EXECUTION_HORIZON, chunk_size - 1),
            max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    requested_device = INFERENCE_DEVICE.strip().lower()
    if not requested_device:
        die("INFERENCE_DEVICE cannot be empty")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        die(
            "INFERENCE_DEVICE requests CUDA, but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver and CUDA-enabled PyTorch installation."
        )
    config.device = requested_device
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    policy.eval()
    policy.reset()

    # The training checkpoint records the tokenizer's cache path from the
    # training server. Prefer the tokenizer bundled beside the model so a
    # copied project remains self-contained and works from any install path.
    tokenizer_candidates = [
        path / "tokenizer",
        path.parents[2] / "tokenizer",
    ]
    bundled_tokenizer = next(
        (candidate for candidate in tokenizer_candidates if candidate.is_dir()),
        tokenizer_candidates[0],
    )
    processor_config_path = path / "policy_preprocessor.json"
    if bundled_tokenizer.is_dir() and processor_config_path.is_file():
        processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
        changed = False

        def use_bundled_tokenizer(value: Any) -> None:
            nonlocal changed
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "tokenizer_name" and item != str(bundled_tokenizer):
                        value[key] = str(bundled_tokenizer)
                        changed = True
                    else:
                        use_bundled_tokenizer(item)
            elif isinstance(value, list):
                for item in value:
                    use_bundled_tokenizer(item)

        use_bundled_tokenizer(processor_config)
        if changed:
            processor_config_path.write_text(
                json.dumps(processor_config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": str(config.device)}},
    )
    return policy, preprocessor, postprocessor, path


def required_camera_bindings(input_features: dict[str, Any], model: dict[str, Any]) -> dict[str, str]:
    bindings = dict(model.get("camera_bindings", {}))
    required: dict[str, str] = {}
    for name in input_features:
        if not name.startswith("observation.images."):
            continue
        suffix = name.rsplit(".", 1)[1]
        required[name] = str(bindings.get(suffix, suffix))
    return required


class TaskPhase(str, Enum):
    STAGE_1 = "stage_1_right_tray_to_scale"
    WAIT_FOR_COLOR = "wait_for_indicator_color"
    STAGE_2 = "stage_2_pick_from_scale"
    STAGE_3 = "stage_3_place_in_target_tray"
    DONE = "done"


@dataclass(frozen=True)
class IndicatorReading:
    color: str | None
    confidence: float
    green_fraction: float
    yellow_fraction: float


class IndicatorColorDetector:
    """Detect the external GREEN/YELLOW lamp using configured camera ROIs.

    A decision is returned only after the settle interval and several identical
    confident frames.  Unknown/ambiguous images never default to either color.
    """

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.votes: deque[str] = deque(maxlen=INDICATOR_STABLE_FRAMES)
        self.last_reading = IndicatorReading(None, 0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.started_at = time.monotonic()
        self.votes.clear()
        self.last_reading = IndicatorReading(None, 0.0, 0.0, 0.0)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def update(self, frames: dict[str, Any]) -> str | None:
        self.last_reading = self._read(frames)
        if self.elapsed_s < INDICATOR_SETTLE_S:
            self.votes.clear()
            return None
        if self.last_reading.color is None:
            self.votes.clear()
            return None
        self.votes.append(self.last_reading.color)
        if len(self.votes) == INDICATOR_STABLE_FRAMES and len(set(self.votes)) == 1:
            return self.votes[0]
        return None

    @staticmethod
    def _read(frames: dict[str, Any]) -> IndicatorReading:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for indicator detection") from exc

        # Top is the stable, primary view. Wrist contributes as an auxiliary
        # view with lower weight because its ROI changes with arm pose.
        weights = {"top": 1.0, "wrist": 0.35}
        green_score = 0.0
        yellow_score = 0.0
        total_weight = 0.0
        for name, roi in INDICATOR_ROIS.items():
            frame = frames.get(name)
            if frame is None:
                continue
            image = frame.color_bgr
            height, width = image.shape[:2]
            x1, y1, x2, y2 = roi
            left = int(np.clip(round(x1 * width), 0, width - 1))
            right = int(np.clip(round(x2 * width), left + 1, width))
            top = int(np.clip(round(y1 * height), 0, height - 1))
            bottom = int(np.clip(round(y2 * height), top + 1, height))
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            saturation = hsv[..., 1]
            value = hsv[..., 2]
            vivid = (saturation >= 90) & (value >= 140)
            hue = hsv[..., 0]
            green = vivid & (hue >= 35) & (hue <= 95)
            yellow = vivid & (hue >= 5) & (hue < 35)
            area = float(crop.shape[0] * crop.shape[1])
            weight = weights.get(name, 0.25)
            green_score += weight * float(np.count_nonzero(green)) / area
            yellow_score += weight * float(np.count_nonzero(yellow)) / area
            total_weight += weight
        if total_weight <= 0:
            return IndicatorReading(None, 0.0, 0.0, 0.0)
        green_fraction = green_score / total_weight
        yellow_fraction = yellow_score / total_weight
        strongest = max(green_fraction, yellow_fraction)
        weakest = min(green_fraction, yellow_fraction)
        dominance = strongest / max(weakest, 1e-6)
        confidence = min(1.0, dominance / INDICATOR_MIN_DOMINANCE)
        if strongest < INDICATOR_MIN_COLOR_FRACTION or dominance < INDICATOR_MIN_DOMINANCE:
            return IndicatorReading(None, confidence, green_fraction, yellow_fraction)
        color = "GREEN" if green_fraction > yellow_fraction else "YELLOW"
        return IndicatorReading(color, confidence, green_fraction, yellow_fraction)


class SegmentedCosmeticsTask:
    """Map debounced gripper events and detected lamp color to training prompts."""

    def __init__(self) -> None:
        self.phase = TaskPhase.STAGE_1
        self.color: str | None = None
        self.prompt: str | None = STAGE_1_PROMPT
        self._stable_gripper = "open"
        self._candidate_gripper: str | None = None
        self._candidate_count = 0
        self._close_count = 0
        self._open_after_close_count = 0

    def update_gripper(self, command: str) -> str | None:
        if command not in {"open", "close"}:
            return None
        if command == self._stable_gripper:
            self._candidate_gripper = None
            self._candidate_count = 0
            return None
        if command != self._candidate_gripper:
            self._candidate_gripper = command
            self._candidate_count = 1
            return None
        self._candidate_count += 1
        if self._candidate_count < PHASE_GRIPPER_DEBOUNCE_STEPS:
            return None
        previous = self._stable_gripper
        self._stable_gripper = command
        self._candidate_gripper = None
        self._candidate_count = 0

        if previous == "open" and command == "close":
            self._close_count += 1
            if self._close_count == 2 and self.phase == TaskPhase.STAGE_2:
                self.phase = TaskPhase.STAGE_3
                self.prompt = self._prompt_for_stage(3)
                return "second_grasp"
            return "first_grasp" if self._close_count == 1 else None

        if previous == "close" and command == "open":
            self._open_after_close_count += 1
            if self._open_after_close_count == 1 and self.phase == TaskPhase.STAGE_1:
                self.phase = TaskPhase.WAIT_FOR_COLOR
                self.prompt = None
                return "scale_release"
            if self._open_after_close_count == 2 and self.phase == TaskPhase.STAGE_3:
                self.phase = TaskPhase.DONE
                self.prompt = None
                return "final_release"
        return None

    def set_color(self, color: str) -> None:
        if self.phase != TaskPhase.WAIT_FOR_COLOR:
            raise RuntimeError(f"Cannot set indicator color during {self.phase.value}")
        if color not in {"GREEN", "YELLOW"}:
            raise ValueError(f"Unsupported indicator color: {color}")
        self.color = color
        self.phase = TaskPhase.STAGE_2
        self.prompt = self._prompt_for_stage(2)

    def _prompt_for_stage(self, stage: int) -> str:
        if self.color == "GREEN":
            return GREEN_STAGE_2_PROMPT if stage == 2 else GREEN_STAGE_3_PROMPT
        if self.color == "YELLOW":
            return YELLOW_STAGE_2_PROMPT if stage == 2 else YELLOW_STAGE_3_PROMPT
        raise RuntimeError("Indicator color is required after stage 1")


def build_observation(
    state: Any,
    frames: dict[str, Any],
    input_features: dict[str, Any],
    bindings: dict[str, str],
    gripper_state: float,
    include_task: bool,
    task_prompt: str | None,
) -> dict[str, Any]:
    joints = [float(value) for value in state.q_actual_deg[:6]]
    if len(joints) < 6:
        joints.extend([0.0] * (6 - len(joints)))
    values: dict[str, Any] = {
        "observation.state": torch.tensor([*state.pose_mm_deg[:6], gripper_state], dtype=torch.float32),
        "observation.robot_joints": torch.tensor(joints, dtype=torch.float32),
        # Compatibility fallback for older checkpoints trained with VR fields.
        "observation.vr_position": torch.zeros(3, dtype=torch.float32),
        "observation.vr_rotation": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
    }
    for feature_name, physical_name in bindings.items():
        feature = input_features[feature_name]
        shape = tuple(int(value) for value in feature.shape)
        frame = frames.get(physical_name)
        if frame is None:
            die(f"Required camera '{physical_name}' for feature '{feature_name}' has no frame")
        values[feature_name] = image_tensor(frame, shape)
    unsupported = set(input_features) - set(values)
    if unsupported:
        die("Unsupported required checkpoint inputs: " + ", ".join(sorted(unsupported)))
    observation = {name: values[name] for name in input_features}
    if include_task:
        if not task_prompt:
            die("A non-empty task prompt is required by this checkpoint")
        observation["task"] = task_prompt
    return observation


def action_vector(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        action = action.get("action", action)
    if not isinstance(action, torch.Tensor):
        die(f"Unsupported policy action type: {type(action).__name__}")
    result = np.asarray(action.squeeze(0).detach().cpu(), dtype=np.float64).reshape(-1)
    if result.shape != (7,) or not np.isfinite(result).all():
        die(f"Robot execution requires a finite 7-D TCP action, got shape {result.shape}")
    return result


class RTCActionWorker:
    """Run policy chunk prediction in one background thread.

    The main robot loop only publishes the newest observation and consumes
    postprocessed actions.  Model inference, RTC prefix guidance, and chunk
    replacement stay on this worker so the policy processors are never called
    concurrently.
    """

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        period_s: float,
    ) -> None:
        rtc_config = policy.config.rtc_config
        if rtc_config is None or not rtc_config.enabled:
            raise ValueError("RTCActionWorker requires an RTC-enabled policy")
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.period_s = period_s
        self.chunk_size = int(policy.config.chunk_size)
        self.rtc_config = rtc_config
        self.action_queue = ActionQueue(rtc_config)
        self.latencies = LatencyTracker(maxlen=50)
        self.queue_threshold = min(RTC_QUEUE_THRESHOLD, self.chunk_size - 1)
        self._observation: dict[str, Any] | None = None
        self._observation_lock = threading.Lock()
        self._generation = 0
        self._ready_generation = -1
        self._generation_inference_count = 0
        self._ever_ready = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._inference_count = 0

    def start(self) -> None:
        self.action_queue.clear()
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="dobot-rtc-inference",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def switch_observation(self, observation: dict[str, Any]) -> int:
        """Invalidate the old task/chunk and start a new prompt generation."""
        with self._observation_lock:
            self._generation += 1
            self._observation = observation
            self._ready_generation = -1
            self._generation_inference_count = 0
            self.action_queue.clear()
            generation = self._generation
        print(f"rtc_generation={generation} state=switched queue=cleared", flush=True)
        return generation

    def invalidate(self) -> int:
        """Discard queued and in-flight results without starting another inference."""
        with self._observation_lock:
            self._generation += 1
            self._observation = None
            self._ready_generation = -1
            self._generation_inference_count = 0
            self.action_queue.clear()
            generation = self._generation
        print(f"rtc_generation={generation} state=invalidated queue=cleared", flush=True)
        return generation

    def publish_observation(self, observation: dict[str, Any], generation: int) -> None:
        with self._observation_lock:
            if generation == self._generation:
                self._observation = observation

    def get_action(self, generation: int) -> torch.Tensor | None:
        with self._observation_lock:
            if generation != self._generation or self._ready_generation != generation:
                return None
            return self.action_queue.get()

    def queue_size(self) -> int:
        return self.action_queue.qsize()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("RTC inference thread failed") from self._error

    def generation_ready(self, generation: int) -> bool:
        with self._observation_lock:
            return generation == self._generation and self._ready_generation == generation

    def wait_until_ready(self, generation: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while not self.generation_ready(generation):
            self.raise_if_failed()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"RTC generation {generation} did not produce a chunk within {timeout_s:.1f}s"
                )
            self._stop_event.wait(0.05)
        self.raise_if_failed()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._observation_lock:
                    observation = self._observation
                    generation = self._generation
                    generation_count = self._generation_inference_count
                if observation is None:
                    self._stop_event.wait(0.005)
                    continue

                is_warmup = not self._ever_ready and generation_count < RTC_WARMUP_INFERENCES
                if not is_warmup and self.action_queue.qsize() > self.queue_threshold:
                    self._stop_event.wait(0.005)
                    continue

                queue_had_actions = not self.action_queue.empty() and not is_warmup
                action_index_before = self.action_queue.get_action_index()
                previous_actions = self.action_queue.get_left_over() if not is_warmup else None
                predicted_latency = (
                    float(self.latencies.p95() or 0.0)
                    if not is_warmup and queue_had_actions
                    else 0.0
                )
                predicted_delay = math.ceil(predicted_latency / self.period_s) if not is_warmup else 0
                execution_horizon = min(
                    self.chunk_size - 1,
                    max(
                        RTC_INITIAL_EXECUTION_HORIZON,
                        predicted_delay + RTC_BLEND_STEPS_AFTER_DELAY,
                    ),
                )
                self.rtc_config.execution_horizon = execution_horizon

                started = time.perf_counter()
                processed_observation = self.preprocessor(observation)
                actions = self.policy.predict_action_chunk(
                    processed_observation,
                    inference_delay=predicted_delay,
                    prev_chunk_left_over=previous_actions,
                )
                original_actions = actions.squeeze(0).clone()
                processed_actions = self.postprocessor(actions).squeeze(0)
                latency_s = time.perf_counter() - started
                actual_delay = math.ceil(latency_s / self.period_s)

                # The robot is stationary while the first chunk is prefetched,
                # so none of that first trajectory is stale.  Later chunks must
                # discard the timesteps that elapsed while inference was running.
                merge_delay = actual_delay if queue_had_actions else 0
                if queue_had_actions and merge_delay >= self.chunk_size:
                    raise RuntimeError(
                        f"RTC inference latency {latency_s:.3f}s ({merge_delay} steps) "
                        f"exhausts chunk_size={self.chunk_size}"
                    )
                with self._observation_lock:
                    if generation != self._generation:
                        print(
                            f"rtc_generation={generation} state=stale_result_discarded",
                            flush=True,
                        )
                        continue
                    self.action_queue.merge(
                        original_actions,
                        processed_actions,
                        merge_delay,
                        action_index_before if queue_had_actions else None,
                    )
                    self._inference_count += 1
                    self._generation_inference_count += 1
                    ready_after = RTC_WARMUP_INFERENCES if not self._ever_ready else 1
                    if self._generation_inference_count >= ready_after:
                        if not self._ever_ready:
                            if actual_delay >= self.chunk_size:
                                raise RuntimeError(
                                    f"Warm inference latency {latency_s:.3f}s ({actual_delay} steps) "
                                    f"exhausts chunk_size={self.chunk_size}"
                                )
                            self.latencies.reset()
                            self._ever_ready = True
                        self._ready_generation = generation
                    if self._ever_ready:
                        self.latencies.add(latency_s)
                if actual_delay * 2 >= self.chunk_size:
                    print(
                        f"rtc_warning=low_latency_margin delay_steps={actual_delay} "
                        f"chunk_size={self.chunk_size}",
                        file=sys.stderr,
                        flush=True,
                    )
                print(
                    f"rtc_inference={self._inference_count} latency_s={latency_s:.3f} "
                    f"delay_steps={actual_delay} horizon={execution_horizon} "
                    f"queue={self.action_queue.qsize()} warmup={is_warmup} "
                    f"generation={generation}",
                    flush=True,
                )
        except BaseException as exc:
            self._error = exc
            print(f"rtc_fatal={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def normalized_gripper_state(gripper: TriggerGripperController | None, config: GripperConfig | None) -> float:
    if gripper is None or config is None:
        return 1.0
    position = gripper.snapshot().get("actual_position")
    if position is None:
        return 1.0
    span = config.open_position - config.close_position
    return float(np.clip((float(position) - config.close_position) / max(abs(span), 1.0), 0.0, 1.0))


def apply_policy_gripper(gripper: TriggerGripperController | None, value: float) -> str:
    if gripper is None:
        return "disabled"
    value = float(np.clip(value, 0.0, 1.0))
    if value <= GRIPPER_CLOSE_THRESHOLD:
        gripper.update(1.0, input_valid=True)
        return "close"
    if value >= GRIPPER_OPEN_THRESHOLD:
        gripper.update(0.0, input_valid=True)
        return "open"
    gripper.update(0.5, input_valid=True)
    return "hold"


def configure_preview_display() -> None:
    if PREVIEW_TARGET == "session":
        return
    if PREVIEW_TARGET != "host":
        die(f"Invalid PREVIEW_TARGET={PREVIEW_TARGET!r}; use 'host' or 'session'")
    display_number = HOST_DESKTOP_DISPLAY.removeprefix(":").split(".", 1)[0]
    socket_path = Path("/tmp/.X11-unix") / f"X{display_number}"
    if not socket_path.exists():
        print(
            f"[preview disabled] Host display socket not found: {socket_path}",
            flush=True,
        )
        os.environ.pop("DISPLAY", None)
        return
    os.environ["DISPLAY"] = HOST_DESKTOP_DISPLAY
    xauthority = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
    if xauthority.is_file():
        os.environ["XAUTHORITY"] = str(xauthority)


def preview_window_enabled() -> bool:
    if not SHOW_CAMERA or PREVIEW_MODE == "off":
        return False
    if PREVIEW_MODE == "window":
        return True
    if PREVIEW_MODE != "auto":
        die(f"Invalid PREVIEW_MODE={PREVIEW_MODE!r}; use 'auto', 'window' or 'off'")
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def start_preview_process() -> None:
    global _PREVIEW_PROCESS, _PREVIEW_QUEUE, _PREVIEW_THREAD
    if _PREVIEW_PROCESS is not None:
        return
    command = [
        "ffplay",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-f",
        "mjpeg",
        "-framerate",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-window_title",
        "Dobot inference",
    ]
    try:
        _PREVIEW_PROCESS = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffplay is required for non-blocking camera preview") from exc
    _PREVIEW_QUEUE = queue.Queue(maxsize=1)

    def write_latest_frames() -> None:
        assert _PREVIEW_PROCESS is not None
        assert _PREVIEW_PROCESS.stdin is not None
        assert _PREVIEW_QUEUE is not None
        while True:
            encoded = _PREVIEW_QUEUE.get()
            if encoded is None:
                return
            try:
                _PREVIEW_PROCESS.stdin.write(encoded)
                _PREVIEW_PROCESS.stdin.flush()
            except (BrokenPipeError, OSError):
                return

    _PREVIEW_THREAD = threading.Thread(
        target=write_latest_frames,
        name="inference-preview",
        daemon=True,
    )
    _PREVIEW_THREAD.start()
    print("preview_backend=ffplay", flush=True)


def submit_preview_frame(canvas: np.ndarray) -> None:
    if not preview_window_enabled():
        return
    import cv2

    start_preview_process()
    if _PREVIEW_PROCESS is None or _PREVIEW_PROCESS.poll() is not None:
        return
    ok, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not ok or _PREVIEW_QUEUE is None:
        return
    try:
        _PREVIEW_QUEUE.put_nowait(encoded.tobytes())
    except queue.Full:
        try:
            _PREVIEW_QUEUE.get_nowait()
        except queue.Empty:
            pass
        try:
            _PREVIEW_QUEUE.put_nowait(encoded.tobytes())
        except queue.Full:
            pass


def stop_preview_process() -> None:
    global _PREVIEW_PROCESS, _PREVIEW_QUEUE, _PREVIEW_THREAD
    if _PREVIEW_QUEUE is not None:
        try:
            _PREVIEW_QUEUE.put_nowait(None)
        except queue.Full:
            pass
    if _PREVIEW_PROCESS is not None:
        _PREVIEW_PROCESS.terminate()
        try:
            _PREVIEW_PROCESS.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _PREVIEW_PROCESS.kill()
            _PREVIEW_PROCESS.wait(timeout=1.0)
    _PREVIEW_PROCESS = None
    _PREVIEW_QUEUE = None
    _PREVIEW_THREAD = None


def draw_preview(
    frames: dict[str, Any],
    step: int,
    action: np.ndarray,
    phase: str,
    color: str | None,
    color_reading: IndicatorReading | None,
) -> bool:
    if not SHOW_CAMERA and not RECORD_VIDEO:
        return True
    import cv2

    images = []
    for name, frame in frames.items():
        image = frame.color_bgr.copy()
        cv2.putText(
            image,
            f"{name} age={frame.age_s * 1000.0:.0f}ms seq={frame.sequence}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if TASK_MODE == "segmented_cosmetics" and name in INDICATOR_ROIS:
            height_px, width_px = image.shape[:2]
            x1, y1, x2, y2 = INDICATOR_ROIS[name]
            cv2.rectangle(
                image,
                (round(x1 * width_px), round(y1 * height_px)),
                (round(x2 * width_px), round(y2 * height_px)),
                (255, 0, 255),
                2,
            )
        images.append(image)
    if not images:
        return True
    height = min(image.shape[0] for image in images)
    images = [cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height)) for image in images]
    canvas = cv2.hconcat(images)
    cv2.putText(
        canvas,
        f"step={step} phase={phase} color={color or 'UNKNOWN'}",
        (12, height - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        1,
    )
    if color_reading is not None:
        cv2.putText(
            canvas,
            f"indicator green={color_reading.green_fraction:.3f} yellow={color_reading.yellow_fraction:.3f}",
            (12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
    else:
        cv2.putText(canvas, f"action={np.round(action, 2).tolist()}", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if RECORD_VIDEO:
        global _VIDEO_WRITER
        if _VIDEO_WRITER is None:
            output = ROOT / "outputs" / "inference_videos"
            output.mkdir(parents=True, exist_ok=True)
            video_path = output / f"inference_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            _VIDEO_WRITER = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (canvas.shape[1], canvas.shape[0]),
            )
            if not _VIDEO_WRITER.isOpened():
                die(f"Failed to open inference video writer: {video_path}")
            print(f"recording_video={video_path}", flush=True)
        _VIDEO_WRITER.write(canvas)
    submit_preview_frame(canvas)
    return True


_VIDEO_WRITER: Any = None
_PREVIEW_PROCESS: subprocess.Popen[bytes] | None = None
_PREVIEW_QUEUE: queue.Queue[bytes | None] | None = None
_PREVIEW_THREAD: threading.Thread | None = None


def main() -> int:
    if CUDA_VISIBLE_DEVICES.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES.strip()
    configure_preview_display()
    if TASK_MODE not in {"fixed", "segmented_cosmetics"}:
        die("TASK_MODE must be 'fixed' or 'segmented_cosmetics'")
    if TASK_MODE == "segmented_cosmetics" and not RTC_ENABLED:
        die("segmented_cosmetics mode requires RTC_ENABLED=True for safe prompt switching")
    if SELECTED_MODEL not in MODEL_REGISTRY:
        die(f"Unknown SELECTED_MODEL={SELECTED_MODEL!r}. Available: {', '.join(MODEL_REGISTRY)}")
    if not 1 <= SPEED_FACTOR_PERCENT <= 100:
        die("Invalid speed factor configuration")
    model = MODEL_REGISTRY[SELECTED_MODEL]
    policy, preprocessor, postprocessor, checkpoint = load_policy(model)
    output_shape = tuple(int(value) for value in policy.config.output_features["action"].shape)
    if output_shape != (7,):
        die(f"Checkpoint action feature must be 7-D TCP+gripper, got {output_shape}")
    bindings = required_camera_bindings(policy.config.input_features, model)
    include_task = "task" in policy.config.input_features or policy.config.type in {
        "pi0", "pi0_fast", "pi05", "smolvla", "groot", "xvla", "wall_x",
        "eo1", "evo1", "molmoact2", "vla_jepa", "lingbot_va", "fastwam",
    }
    if TASK_MODE == "segmented_cosmetics" and not include_task:
        die("The selected checkpoint does not accept language task prompts")
    task_controller = SegmentedCosmeticsTask() if TASK_MODE == "segmented_cosmetics" else None
    color_detector = IndicatorColorDetector() if task_controller is not None else None
    active_prompt = task_controller.prompt if task_controller is not None else TASK_PROMPT

    config = load_project_config()
    safety = SafetySupervisor(config, require_robot_enabled=EXECUTE)
    if safety.max_step_mm <= 0 or safety.max_step_deg <= 0:
        die("safety.max_step_mm and safety.max_step_deg must be positive")
    orientation_reference = [
        float(value)
        for value in (
            INFERENCE_START_POSE_MM_DEG
            if MOVE_TO_INFERENCE_START
            else config["calibration"]["robot_initial_pose_mm_deg"]
        )
    ]
    camera_configs: dict[str, dict[str, Any]] = {"top": dict(config.get("camera", {}))}
    camera_configs.update({name: dict(value) for name, value in config.get("cameras", {}).items()})
    camera_overrides = model.get("camera_overrides", {})
    cameras = []
    for name in sorted(set(bindings.values())):
        values = camera_configs.get(name)
        if values is None:
            die(f"No camera config named '{name}'")
        if name in camera_overrides:
            values.update(dict(camera_overrides[name]))
        values["name"] = name
        values["preview"] = False
        cameras.append(create_camera(CameraConfig.from_mapping(values)))
    camera = CameraGroup(cameras) if cameras else None
    robot_net, robot_cfg = config["robot_network"], config["robot"]
    if EXECUTE:
        robot = DobotTcpAdapter(
            robot_net["ip"],
            int(robot_net["dashboard_port"]),
            int(robot_net["move_port"]),
            int(robot_net["feedback_port"]),
            lock_j6=bool(robot_cfg.get("lock_j6", False)),
            user_index=int(robot_cfg.get("user_index", 0)),
            tool_index=int(robot_cfg.get("tool_index", 0)),
        )
    else:
        robot = MockRobotAdapter(
            [float(value) for value in robot_cfg["default_pose_mm_deg"]]
        )
    gripper = None
    gripper_config = None
    if EXECUTE_GRIPPER and bool(config.get("gripper", {}).get("enabled", False)):
        gripper_config = replace(GripperConfig.from_mapping(config["gripper"]), require_initial_press_before_open=False)
        gripper = TriggerGripperController(gripper_config, robot_net["ip"], execute=EXECUTE)
    if TASK_MODE == "segmented_cosmetics" and gripper is None:
        die("segmented_cosmetics mode requires an enabled gripper for phase transitions")
    print(f"model={SELECTED_MODEL} policy_type={policy.config.type} checkpoint={checkpoint}")
    print(
        f"device={policy.config.device} cameras={bindings} execute={EXECUTE} "
        f"robot_state={'live' if EXECUTE else 'mock_initial_pose'} task_mode={TASK_MODE} "
        f"task={active_prompt!r}"
    )
    print(
        f"preview_window={preview_window_enabled()} record_video={RECORD_VIDEO} "
        f"display={os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or 'none'}",
        flush=True,
    )
    print(
        f"rtc={RTC_ENABLED} chunk_size={policy.config.chunk_size} "
        f"step_limit_mm={safety.max_step_mm} step_limit_deg={safety.max_step_deg}",
        flush=True,
    )
    if camera:
        camera.start()
        print("camera_streams_ready", flush=True)
    robot.connect()
    print("robot_state_ready", flush=True)
    rtc_worker: RTCActionWorker | None = None
    active_generation = -1
    try:
        if EXECUTE and ENABLE_ROBOT:
            robot.enable_robot()
        if gripper:
            gripper.connect(robot.dashboard)
        if EXECUTE and MOVE_TO_INFERENCE_START:
            # MovJ's own v/a values define the approach speed.  Use an unscaled
            # controller factor for that move, then restore the inference factor.
            robot.set_speed_factor(100)
            move_to_inference_start(robot, safety)
        if EXECUTE:
            robot.set_speed_factor(SPEED_FACTOR_PERCENT)
        period = float(robot_cfg.get("servo_period_s", 0.05))
        servo_t = float(robot_cfg.get("servo_t_s", period))
        if period <= 0 or servo_t <= 0:
            die("robot servo_period_s and servo_t_s must be positive")

        if RTC_ENABLED:
            rtc_worker = RTCActionWorker(policy, preprocessor, postprocessor, period)
            rtc_worker.start()
            state = robot.state()
            if (
                not state.connected
                or state.feedback_age_s > safety.feedback_timeout_s
                or state.error_status
                or state.collision_state
                or not safety.pose_in_workspace(state.pose_mm_deg)
            ):
                die("Safety block while preparing the first RTC observation")
            if camera:
                camera.raise_if_failed()
            frames = camera.latest_frames() if camera else {}
            first_observation = build_observation(
                state,
                frames,
                policy.config.input_features,
                bindings,
                normalized_gripper_state(gripper, gripper_config),
                include_task,
                active_prompt,
            )
            active_generation = rtc_worker.switch_observation(first_observation)
            print("rtc_prefill: waiting for first action chunk", flush=True)
            rtc_worker.wait_until_ready(active_generation, RTC_PREFILL_TIMEOUT_S)
            print(f"rtc_prefill_ready queue={rtc_worker.queue_size()}", flush=True)

        queue_empty_since: float | None = None
        last_action = np.asarray(
            [*robot.state().pose_mm_deg[:6], normalized_gripper_state(gripper, gripper_config)],
            dtype=np.float64,
        )
        step_iterator = itertools.count() if CONTINUOUS_TASKS else range(STEPS)
        completed_tasks = 0
        for step in step_iterator:
            started = time.monotonic()
            if rtc_worker is not None:
                rtc_worker.raise_if_failed()
            if step == 0:
                print("first_step: reading state and camera frames", flush=True)
            state = robot.state()
            issues = []
            if not state.connected or state.feedback_age_s > safety.feedback_timeout_s or state.error_status or state.collision_state:
                issues.append("robot feedback/safety state")
            if not safety.pose_in_workspace(state.pose_mm_deg):
                issues.append("pose outside workspace")
            if issues:
                die("Safety block: " + ", ".join(issues))
            if camera:
                camera.raise_if_failed()
            frames = camera.latest_frames() if camera else {}
            if step % 30 == 0:
                camera_ages_ms = {
                    name: round(frame.age_s * 1000.0, 1)
                    for name, frame in frames.items()
                    if frame is not None
                }
                print(f"camera_age_ms={camera_ages_ms}", flush=True)
            if step == 0:
                print("first_step: building observation", flush=True)
            grip_state = normalized_gripper_state(gripper, gripper_config)

            phase = task_controller.phase if task_controller is not None else None
            if phase == TaskPhase.DONE:
                print("segmented_task=complete reason=final_release", flush=True)
                break

            # After the first scale release no old model action may run.  The
            # robot holds its current pose until a stable color is observed and
            # the first chunk for the new prompt is ready.
            if phase == TaskPhase.WAIT_FOR_COLOR:
                assert color_detector is not None
                detected_color = color_detector.update(frames)
                reading = color_detector.last_reading
                if step % 15 == 0:
                    print(
                        f"indicator_wait elapsed_s={color_detector.elapsed_s:.2f} "
                        f"candidate={reading.color or 'UNKNOWN'} "
                        f"green={reading.green_fraction:.4f} yellow={reading.yellow_fraction:.4f}",
                        flush=True,
                    )
                if detected_color is not None:
                    task_controller.set_color(detected_color)
                    active_prompt = task_controller.prompt
                    observation = build_observation(
                        state,
                        frames,
                        policy.config.input_features,
                        bindings,
                        grip_state,
                        include_task,
                        active_prompt,
                    )
                    assert rtc_worker is not None
                    active_generation = rtc_worker.switch_observation(observation)
                    print(
                        f"task_transition=indicator_confirmed color={detected_color} "
                        f"phase={task_controller.phase.value} generation={active_generation} "
                        f"task={active_prompt!r}",
                        flush=True,
                    )
                elif color_detector.elapsed_s >= INDICATOR_TIMEOUT_S:
                    die(
                        "Indicator color remained uncertain; robot held position and inference stopped. "
                        f"green={reading.green_fraction:.4f}, yellow={reading.yellow_fraction:.4f}"
                    )
                if not draw_preview(
                    {name: frame for name, frame in frames.items() if frame is not None},
                    step,
                    last_action,
                    task_controller.phase.value,
                    task_controller.color,
                    reading,
                ):
                    break
                if EXECUTE:
                    robot.servo_p(
                        state.pose_mm_deg,
                        t=servo_t,
                        aheadtime=float(robot_cfg.get("servo_aheadtime", 35)),
                        gain=float(robot_cfg.get("servo_gain", 450)),
                    )
                elapsed = time.monotonic() - started
                if elapsed < period:
                    time.sleep(period - elapsed)
                continue

            active_prompt = task_controller.prompt if task_controller is not None else TASK_PROMPT
            observation = build_observation(
                state,
                frames,
                policy.config.input_features,
                bindings,
                grip_state,
                include_task,
                active_prompt,
            )
            if rtc_worker is not None:
                rtc_worker.raise_if_failed()
                rtc_worker.publish_observation(observation, active_generation)
                generation_wait = not rtc_worker.generation_ready(active_generation)
                policy_action = rtc_worker.get_action(active_generation)
                if policy_action is None:
                    if queue_empty_since is None:
                        queue_empty_since = time.monotonic()
                        reason = "prompt_switch" if generation_wait else "queue_empty"
                        print(f"rtc_hold reason={reason} generation={active_generation}", flush=True)
                    empty_duration = time.monotonic() - queue_empty_since
                    if not generation_wait and empty_duration >= RTC_QUEUE_EMPTY_ABORT_S:
                        die(f"RTC action queue remained empty for {empty_duration:.3f}s")
                    if EXECUTE:
                        robot.servo_p(
                            state.pose_mm_deg,
                            t=servo_t,
                            aheadtime=float(robot_cfg.get("servo_aheadtime", 35)),
                            gain=float(robot_cfg.get("servo_gain", 450)),
                        )
                    elapsed = time.monotonic() - started
                    if elapsed < period:
                        time.sleep(period - elapsed)
                    continue
                queue_empty_since = None
                action = action_vector(policy_action)
            else:
                with torch.inference_mode():
                    processed_observation = preprocessor(observation)
                    if step == 0:
                        print("first_step: running policy forward", flush=True)
                    policy_action = policy.select_action(processed_observation)
                    if step == 0:
                        print("first_step: postprocessing action", flush=True)
                    action = action_vector(postprocessor(policy_action))
            last_action = action.copy()
            guarded_action, orientation_corrections = guard_orientation(
                action,
                orientation_reference,
            )
            target = safety.limit_command_lead(
                state.pose_mm_deg,
                safety.limit_step(state.pose_mm_deg, guarded_action, period),
            )
            gripper_command = apply_policy_gripper(gripper, float(action[6]))
            phase_event = (
                task_controller.update_gripper(gripper_command)
                if task_controller is not None
                else None
            )
            if step == 0 or step % 30 == 0:
                if orientation_corrections:
                    print(
                        "orientation_guard=" + ", ".join(orientation_corrections),
                        flush=True,
                    )
                print(
                    f"step={step} raw_action={action.tolist()} target={target} "
                    f"gripper={gripper_command}",
                    flush=True,
                )
            if phase_event is not None:
                print(
                    f"gripper_event={phase_event} command={gripper_command} "
                    f"phase={task_controller.phase.value if task_controller else 'fixed'}",
                    flush=True,
                )
                if phase_event == "scale_release":
                    assert rtc_worker is not None and color_detector is not None
                    active_generation = rtc_worker.invalidate()
                    color_detector.reset()
                    print("task_transition=wait_for_indicator robot=hold", flush=True)
                elif phase_event == "second_grasp":
                    assert rtc_worker is not None and task_controller is not None
                    active_prompt = task_controller.prompt
                    switched_observation = dict(observation)
                    switched_observation["task"] = active_prompt
                    active_generation = rtc_worker.switch_observation(switched_observation)
                    print(
                        f"task_transition=second_grasp color={task_controller.color} "
                        f"phase={task_controller.phase.value} generation={active_generation} "
                        f"task={active_prompt!r}",
                        flush=True,
                    )
                elif phase_event == "final_release":
                    assert rtc_worker is not None
                    active_generation = rtc_worker.invalidate()
            phase_name = task_controller.phase.value if task_controller is not None else "fixed"
            color_name = task_controller.color if task_controller is not None else None
            reading = color_detector.last_reading if color_detector is not None else None
            if not draw_preview(
                {name: frame for name, frame in frames.items() if frame is not None},
                step,
                action,
                phase_name,
                color_name,
                reading,
            ):
                break
            if EXECUTE:
                robot.servo_p(target, t=servo_t, aheadtime=float(robot_cfg.get("servo_aheadtime", 35)), gain=float(robot_cfg.get("servo_gain", 450)))
            if phase_event == "final_release":
                completed_tasks += 1
                print(
                    f"segmented_task=complete reason=final_release "
                    f"completed_tasks={completed_tasks}",
                    flush=True,
                )
                if not CONTINUOUS_TASKS:
                    break

                # Keep the completed RTC generation invalid while returning to
                # the same inference start used before the first task.
                if CONTINUOUS_RESTART_DELAY_S > 0:
                    time.sleep(CONTINUOUS_RESTART_DELAY_S)
                if EXECUTE and MOVE_TO_INFERENCE_START:
                    robot.set_speed_factor(100)
                    move_to_inference_start(robot, safety)
                    robot.set_speed_factor(SPEED_FACTOR_PERCENT)

                task_controller = SegmentedCosmeticsTask()
                color_detector = IndicatorColorDetector()
                active_prompt = task_controller.prompt
                state = robot.state()
                if (
                    not state.connected
                    or state.feedback_age_s > safety.feedback_timeout_s
                    or state.error_status
                    or state.collision_state
                    or not safety.pose_in_workspace(state.pose_mm_deg)
                ):
                    die("Safety block while preparing the next continuous task")
                if camera:
                    camera.raise_if_failed()
                frames = camera.latest_frames() if camera else {}
                next_observation = build_observation(
                    state,
                    frames,
                    policy.config.input_features,
                    bindings,
                    normalized_gripper_state(gripper, gripper_config),
                    include_task,
                    active_prompt,
                )
                assert rtc_worker is not None
                active_generation = rtc_worker.switch_observation(next_observation)
                print(
                    f"continuous_task=restart next_task={completed_tasks + 1} "
                    f"generation={active_generation} task={active_prompt!r}",
                    flush=True,
                )
                rtc_worker.wait_until_ready(active_generation, RTC_PREFILL_TIMEOUT_S)
                queue_empty_since = None
                last_action = np.asarray(
                    [
                        *state.pose_mm_deg[:6],
                        normalized_gripper_state(gripper, gripper_config),
                    ],
                    dtype=np.float64,
                )
                print(
                    f"continuous_task=ready task={completed_tasks + 1} "
                    f"queue={rtc_worker.queue_size()}",
                    flush=True,
                )
                continue
            elapsed = time.monotonic() - started
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        if rtc_worker is not None:
            rtc_worker.stop()
        if _VIDEO_WRITER is not None:
            _VIDEO_WRITER.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        stop_preview_process()
        if camera:
            camera.stop()
        if gripper:
            gripper.close()
        if EXECUTE:
            robot.stop()
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
