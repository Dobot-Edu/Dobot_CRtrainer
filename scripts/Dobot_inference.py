#!/usr/bin/env python3
"""Run a selected LeRobot checkpoint on the Dobot CR5A.

Edit SELECTED_MODEL and TASK_PROMPT below. The checkpoint config determines
the actual LeRobot policy class, required camera streams and preprocessing.
Only checkpoints producing this project's 7-D TCP action contract may command
the real robot.
"""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

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

# 语言条件模型使用的任务提示词。Pi0、Pi0.5、VLA 等策略推理时应与
# 数据转换和训练时写入数据集的任务描述保持一致。
TASK_PROMPT = "Pick up the gear and place it at the target position."

# 是否把模型输出发送给真实机械臂。False 使用初始位姿模拟状态，只检查
# 相机、模型输出、实时预览和视频录制；True 会连接并控制真实机械臂。
EXECUTE = True

# EXECUTE=True 时是否主动发送机械臂使能命令。机械臂已经使能时也可保留 True。
ENABLE_ROBOT = True

# 是否根据模型输出控制真实夹爪。只有 EXECUTE 和本参数都为 True，且项目
# 配置中启用了夹爪，才会实际执行夹爪动作。
EXECUTE_GRIPPER = True

# 推理循环总步数。实际运行时长约为 STEPS × 机械臂伺服周期。
STEPS = 1500

# Dobot 全局运动速度百分比，范围为 1～100；数值越大，机械臂动作越快。
SPEED_FACTOR_PERCENT = 5

# 每个推理周期允许 TCP 在 X/Y/Z 各方向变化的最大距离，单位为毫米。
# 该限制用于抑制位置突跳，不代表机械臂的最终工作空间范围。
MAX_STEP_MM = 1.0

# 每个推理周期允许 TCP 在 Rx/Ry/Rz 各轴变化的最大角度，单位为度。
# 该限制使用等价角度的最短路径，降低腕部姿态突跳风险。
MAX_STEP_DEG = 0.5

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
        "checkpoint": "outputs/train/test123_pi05/checkpoints/003000/pretrained_model",
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


def clamp_step(current: list[float], target: np.ndarray) -> list[float]:
    result = list(current)
    for index in range(3):
        result[index] += float(np.clip(target[index] - current[index], -MAX_STEP_MM, MAX_STEP_MM))
    for index in range(3, 6):
        delta = float(np.clip(wrap_delta_deg(float(target[index]), current[index]), -MAX_STEP_DEG, MAX_STEP_DEG))
        result[index] = normalize_deg(current[index] + delta)
    return result


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
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    policy.eval()
    policy.reset()
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


def build_observation(
    state: Any,
    frames: dict[str, Any],
    input_features: dict[str, Any],
    bindings: dict[str, str],
    gripper_state: float,
    include_task: bool,
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
        observation["task"] = TASK_PROMPT
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


def draw_preview(frames: dict[str, Any], step: int, action: np.ndarray) -> bool:
    if not SHOW_CAMERA and not RECORD_VIDEO:
        return True
    import cv2

    images = []
    for name, frame in frames.items():
        image = frame.color_bgr.copy()
        cv2.putText(image, name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        images.append(image)
    if not images:
        return True
    height = min(image.shape[0] for image in images)
    images = [cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height)) for image in images]
    canvas = cv2.hconcat(images)
    cv2.putText(canvas, f"step={step} action={np.round(action, 2).tolist()}", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
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
    configure_preview_display()
    if SELECTED_MODEL not in MODEL_REGISTRY:
        die(f"Unknown SELECTED_MODEL={SELECTED_MODEL!r}. Available: {', '.join(MODEL_REGISTRY)}")
    if not 1 <= SPEED_FACTOR_PERCENT <= 100 or MAX_STEP_MM <= 0 or MAX_STEP_DEG <= 0:
        die("Invalid speed or step limit configuration")
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

    config = load_project_config()
    safety = SafetySupervisor(config, require_robot_enabled=EXECUTE)
    orientation_reference = [
        float(value)
        for value in config["calibration"]["robot_initial_pose_mm_deg"]
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
    print(f"model={SELECTED_MODEL} policy_type={policy.config.type} checkpoint={checkpoint}")
    print(
        f"device={policy.config.device} cameras={bindings} execute={EXECUTE} "
        f"robot_state={'live' if EXECUTE else 'mock_initial_pose'} task={TASK_PROMPT!r}"
    )
    print(
        f"preview_window={preview_window_enabled()} record_video={RECORD_VIDEO} "
        f"display={os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or 'none'}",
        flush=True,
    )
    if camera:
        camera.start()
        print("camera_streams_ready", flush=True)
    robot.connect()
    print("robot_state_ready", flush=True)
    try:
        if EXECUTE and ENABLE_ROBOT:
            robot.enable_robot()
        if EXECUTE:
            robot.set_speed_factor(SPEED_FACTOR_PERCENT)
        if gripper:
            gripper.connect(robot.dashboard)
        period = float(robot_cfg.get("servo_period_s", 0.05))
        for step in range(STEPS):
            started = time.monotonic()
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
            if step == 0:
                print("first_step: building observation", flush=True)
            grip_state = normalized_gripper_state(gripper, gripper_config)
            observation = build_observation(
                state, frames, policy.config.input_features, bindings, grip_state, include_task
            )
            with torch.inference_mode():
                processed_observation = preprocessor(observation)
                if step == 0:
                    print("first_step: running policy forward", flush=True)
                policy_action = policy.select_action(processed_observation)
                if step == 0:
                    print("first_step: postprocessing action", flush=True)
                action = action_vector(postprocessor(policy_action))
            guarded_action, orientation_corrections = guard_orientation(
                action,
                orientation_reference,
            )
            target = safety.limit_command_lead(
                state.pose_mm_deg,
                safety.clamp_pose(clamp_step(state.pose_mm_deg, guarded_action)),
            )
            gripper_command = apply_policy_gripper(gripper, float(action[6]))
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
            if not draw_preview({name: frame for name, frame in frames.items() if frame is not None}, step, action):
                break
            if EXECUTE:
                robot.servo_p(target, t=float(robot_cfg.get("servo_t_s", period)), aheadtime=float(robot_cfg.get("servo_aheadtime", 35)), gain=float(robot_cfg.get("servo_gain", 450)))
            elapsed = time.monotonic() - started
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
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
