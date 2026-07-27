from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from ..control.models import RobotState, TeleopCommand, VRInput
from ..devices.camera import CameraFrame


SCHEMA_VERSION = "dobot-vr-teleop-episode-v5"


@dataclass(frozen=True)
class RecorderEvent:
    action: str
    episode_dir: Path
    frame_count: int = 0
    valid: bool = False


class EpisodeRecorder:
    """Crash-tolerant episode recorder for synchronized teleoperation samples."""

    def __init__(
        self,
        output_dir: str | Path,
        project_name: str,
        task: str,
        control_hz: float,
        record_button: str = "button_b",
        flush_every_frames: int = 10,
        min_episode_frames: int = 10,
        save_npz: bool = True,
        config_snapshot: dict[str, Any] | None = None,
        robot_name: str = "dobot_cr5a",
        image_jpeg_quality: int = 90,
        image_queue_size: int = 120,
    ) -> None:
        if control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if flush_every_frames <= 0:
            raise ValueError("flush_every_frames must be positive")
        if min_episode_frames < 1:
            raise ValueError("min_episode_frames must be at least 1")
        if record_button not in ("button_a", "button_b"):
            raise ValueError("record_button must be button_a or button_b")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.project_name = project_name.strip() or "Dobot_CRtrainer"
        self.task = task.strip() or "vr_teleoperation"
        self.control_hz = float(control_hz)
        self.record_button = record_button
        self.flush_every_frames = int(flush_every_frames)
        self.min_episode_frames = int(min_episode_frames)
        self.save_npz = bool(save_npz)
        self.config_snapshot = config_snapshot or {}
        self.robot_name = robot_name
        self.image_jpeg_quality = int(image_jpeg_quality)
        self.image_queue_size = int(image_queue_size)
        if not 1 <= self.image_jpeg_quality <= 100:
            raise ValueError("image_jpeg_quality must be in [1, 100]")
        if self.image_queue_size < 1:
            raise ValueError("image_queue_size must be at least 1")

        self.recording = False
        self._last_record_button = False
        self._episode_dir: Path | None = None
        self._frames_file: TextIO | None = None
        self._frame_count = 0
        self._skipped_frames = 0
        self._hold_reasons: dict[str, int] = {}
        self._start_unix_s = 0.0
        self._start_monotonic_s = 0.0
        self._last_elapsed_s = 0.0
        self._metadata: dict[str, Any] = {}
        self._finalizer_threads: list[threading.Thread] = []
        self._image_queue: queue.Queue[Any] | None = None
        self._image_writer_thread: threading.Thread | None = None
        self._image_writer_error = ""
        self._camera_frames_enqueued = 0
        self._camera_frames_written = 0
        self._camera_frames_dropped = 0
        self._last_camera_sequences: dict[str, int | None] = {}
        self._last_camera_rgb_paths: dict[str, str | None] = {}
        self._last_camera_depth_paths: dict[str, str | None] = {}

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_dir

    def handle_record_button(self, vr: VRInput | None) -> RecorderEvent | None:
        pressed = bool(getattr(vr, self.record_button, False)) if vr is not None else False
        rising = pressed and not self._last_record_button
        self._last_record_button = pressed
        if not rising:
            return None
        if self.recording:
            return self.stop_episode("record_button")
        return self.start_episode()

    def start_episode(self) -> RecorderEvent:
        if self.recording:
            assert self._episode_dir is not None
            return RecorderEvent("already_recording", self._episode_dir, self._frame_count, False)

        episode_dir = self._next_episode_dir()
        episode_dir.mkdir(parents=True, exist_ok=False)
        self._episode_dir = episode_dir
        self._frames_file = (episode_dir / "frames.jsonl").open(
            "w", encoding="utf-8", buffering=64 * 1024
        )
        self._frame_count = 0
        self._skipped_frames = 0
        self._hold_reasons = {}
        self._start_unix_s = time.time()
        self._start_monotonic_s = time.monotonic()
        self._last_elapsed_s = 0.0
        self._image_writer_error = ""
        self._camera_frames_enqueued = 0
        self._camera_frames_written = 0
        self._camera_frames_dropped = 0
        self._last_camera_sequences = {}
        self._last_camera_rgb_paths = {}
        self._last_camera_depth_paths = {}
        self.recording = True
        cameras = self._configured_cameras()
        self._metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "recording",
            "valid": False,
            "task": self.task,
            "project_name": self.project_name,
            "robot_name": self.robot_name,
            "control_hz": self.control_hz,
            "record_button": self.record_button,
            "started_at": datetime.now().astimezone().isoformat(),
            "started_unix_s": self._start_unix_s,
            "frame_count": 0,
            "skipped_frames": 0,
            "cameras": cameras,
            "units": {
                "robot_tcp_pose": ["mm", "mm", "mm", "deg", "deg", "deg"],
                "robot_joint_position": "deg",
                "robot_joint_velocity": "deg/s",
                "vr_position": "m",
                "vr_rotation": "quaternion_xyzw",
                "gripper_position": "device_units_0_1000",
            },
            "config_snapshot": self.config_snapshot,
        }
        self._write_metadata()
        return RecorderEvent("started", episode_dir)

    def record(
        self,
        vr: VRInput | None,
        robot: RobotState,
        command: TeleopCommand,
        dt_s: float,
        execute: bool,
        gripper_state: dict[str, Any] | None = None,
        camera_frames: dict[str, CameraFrame | None] | None = None,
    ) -> bool:
        if not self.recording:
            return False
        if vr is None or not command.allowed or command.target_pose_mm_deg is None:
            self._skipped_frames += 1
            reason = command.reason or "missing_vr"
            reason_key = reason.split(":", 1)[0]
            self._hold_reasons[reason_key] = self._hold_reasons.get(reason_key, 0) + 1
            return False

        assert self._frames_file is not None
        elapsed_s = time.monotonic() - self._start_monotonic_s
        gripper = gripper_state or {}
        cameras = self._queue_camera_frames(camera_frames or {})
        primary_camera = cameras.get("top") or next(iter(cameras.values()), self._empty_camera_record())
        frame = {
            "frame_index": self._frame_count,
            "timestamp_unix_s": time.time(),
            "elapsed_s": elapsed_s,
            "dt_s": float(dt_s),
            "observation": {
                "robot_tcp_pose_mm_deg": _vector(robot.pose_mm_deg, 6),
                "robot_joint_positions_deg": _optional_vector(robot.q_actual_deg, 6),
                "robot_joint_velocities_deg_s": _optional_vector(
                    robot.q_actual_velocity_deg_s, 6
                ),
                "robot_tcp_speed_mm_deg_s": _optional_vector(
                    robot.tcp_speed_actual_mm_deg_s, 6
                ),
                "robot_tcp_force": _optional_vector(robot.tcp_force_actual, 6),
                "robot_mode": robot.robot_mode,
                "safety_state": robot.safety_state,
                "error_status": int(robot.error_status),
                "collision_state": int(robot.collision_state),
                "speed_scaling": _optional_float(robot.speed_scaling),
                "feedback_age_s": float(robot.feedback_age_s),
                "gripper_actual_position": gripper.get("actual_position"),
                "gripper_target_position": gripper.get("target_position"),
                "gripper_applied_closed": gripper.get("applied_closed"),
                "vr_right_position_m": _vector(vr.right_pos, 3),
                "vr_right_rotation_xyzw": _vector(vr.right_rot, 4),
                "vr_right_grip": float(vr.right_grip),
                "vr_right_trigger": float(vr.right_trigger),
                "vr_right_thumbstick": _vector(vr.right_thumbstick, 2),
                "vr_tracking_valid": bool(vr.tracking_valid),
                "vr_source_timestamp_s": float(vr.timestamp),
                "vr_packet_age_s": float(vr.age_s),
                # Keep the legacy primary-camera fields so older analysis tools
                # can still read v4 episodes. New converters use cameras.top and
                # cameras.wrist below.
                "camera_frame_sequence": primary_camera["sequence"],
                "camera_timestamp_unix_s": primary_camera["timestamp_unix_s"],
                "camera_age_s": primary_camera["age_s"],
                "camera_rgb_path": primary_camera["rgb_path"],
                "camera_depth_path": primary_camera["depth_path"],
                "cameras": cameras,
            },
            "action": {
                "target_tcp_pose_mm_deg": _vector(command.target_pose_mm_deg, 6),
                "executed": bool(execute),
                "clutched": bool(command.clutched),
                "reason": command.reason,
                "gripper_desired_closed": gripper.get("desired_closed"),
                "gripper_applied_closed": gripper.get("applied_closed"),
                "gripper_target_position": gripper.get("target_position"),
                "gripper_executed": bool(gripper.get("executed", False)),
                "gripper_initialize_status": gripper.get("initialize_status"),
                "gripper_status": gripper.get("grip_status"),
                "gripper_actual_position": gripper.get("actual_position"),
            },
        }
        self._frames_file.write(
            json.dumps(frame, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        self._frame_count += 1
        self._last_elapsed_s = elapsed_s
        if self._frame_count % self.flush_every_frames == 0:
            self._frames_file.flush()
        return True

    def stop_episode(self, reason: str = "manual") -> RecorderEvent:
        if not self.recording or self._episode_dir is None:
            raise RuntimeError("No episode is being recorded")

        episode_dir = self._episode_dir
        if self._frames_file is not None:
            self._frames_file.flush()
            self._frames_file.close()
            self._frames_file = None
        self.recording = False
        self._stop_image_writer()

        valid = self._frame_count >= self.min_episode_frames
        metadata = dict(self._metadata)
        metadata.update(
            {
                "status": "raw_complete_pending_npz" if self.save_npz else "complete",
                "valid": valid,
                "stop_reason": reason,
                "stopped_at": datetime.now().astimezone().isoformat(),
                "duration_s": self._last_elapsed_s,
                "frame_count": self._frame_count,
                "skipped_frames": self._skipped_frames,
                "hold_reasons": self._hold_reasons,
                "minimum_valid_frames": self.min_episode_frames,
                "camera_frames_enqueued": self._camera_frames_enqueued,
                "camera_frames_written": self._camera_frames_written,
                "camera_frames_dropped": self._camera_frames_dropped,
                "camera_writer_error": self._image_writer_error or None,
                "artifacts": {
                    "raw_frames": "frames.jsonl",
                    "array_data": None,
                },
            }
        )
        self._metadata = metadata
        self._write_metadata_to(episode_dir, metadata)
        if self.save_npz:
            thread = threading.Thread(
                target=self._finalize_npz,
                args=(episode_dir, metadata),
                name=f"episode-finalizer-{episode_dir.name}",
                daemon=False,
            )
            thread.start()
            self._finalizer_threads.append(thread)
        event = RecorderEvent("stopped", episode_dir, self._frame_count, valid)
        self._episode_dir = None
        return event

    def close(self) -> RecorderEvent | None:
        event = self.stop_episode("application_exit") if self.recording else None
        for thread in self._finalizer_threads:
            thread.join()
        self._finalizer_threads.clear()
        return event

    def _next_episode_dir(self) -> Path:
        task_dir = (
            self.output_dir
            / _safe_name(self.task)
            / "episodes"
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        candidate = task_dir / f"episode_{stamp}"
        suffix = 1
        while candidate.exists():
            candidate = task_dir / f"episode_{stamp}_{suffix:02d}"
            suffix += 1
        return candidate

    def _write_metadata(self) -> None:
        assert self._episode_dir is not None
        self._write_metadata_to(self._episode_dir, self._metadata)

    @staticmethod
    def _write_metadata_to(episode_dir: Path, metadata: dict[str, Any]) -> None:
        destination = episode_dir / "metadata.json"
        temporary = episode_dir / "metadata.json.tmp"
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def _finalize_npz(self, episode_dir: Path, metadata: dict[str, Any]) -> None:
        try:
            self._write_npz(episode_dir)
            metadata["status"] = "complete"
            metadata["artifacts"]["array_data"] = "data.npz"
        except Exception as exc:
            metadata["status"] = "raw_complete_npz_failed"
            metadata["valid"] = False
            metadata["npz_error"] = str(exc)
        self._write_metadata_to(episode_dir, metadata)

    def _write_npz(self, episode_dir: Path) -> None:
        frames = []
        with (episode_dir / "frames.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    frames.append(json.loads(line))

        arrays = {
            "frame_index": np.asarray([frame["frame_index"] for frame in frames], dtype=np.int64),
            "timestamp_unix_s": np.asarray(
                [frame["timestamp_unix_s"] for frame in frames], dtype=np.float64
            ),
            "elapsed_s": np.asarray([frame["elapsed_s"] for frame in frames], dtype=np.float64),
            "dt_s": np.asarray([frame["dt_s"] for frame in frames], dtype=np.float64),
            "observation_robot_tcp_pose_mm_deg": _frame_vectors(
                frames, "observation", "robot_tcp_pose_mm_deg", 6
            ),
            "observation_robot_joint_positions_deg": _frame_vectors(
                frames, "observation", "robot_joint_positions_deg", 6
            ),
            "observation_robot_joint_velocities_deg_s": _frame_vectors(
                frames, "observation", "robot_joint_velocities_deg_s", 6
            ),
            "observation_robot_tcp_speed_mm_deg_s": _frame_vectors(
                frames, "observation", "robot_tcp_speed_mm_deg_s", 6
            ),
            "observation_robot_tcp_force": _frame_vectors(
                frames, "observation", "robot_tcp_force", 6
            ),
            "observation_gripper_actual_position": _frame_scalars(
                frames, "observation", "gripper_actual_position"
            ),
            "observation_gripper_target_position": _frame_scalars(
                frames, "observation", "gripper_target_position"
            ),
            "observation_gripper_applied_closed": _frame_scalars(
                frames, "observation", "gripper_applied_closed"
            ),
            "observation_vr_right_position_m": _frame_vectors(
                frames, "observation", "vr_right_position_m", 3
            ),
            "observation_vr_right_rotation_xyzw": _frame_vectors(
                frames, "observation", "vr_right_rotation_xyzw", 4
            ),
            "observation_vr_grip_trigger": np.asarray(
                [
                    [
                        frame["observation"]["vr_right_grip"],
                        frame["observation"]["vr_right_trigger"],
                    ]
                    for frame in frames
                ],
                dtype=np.float64,
            ).reshape((-1, 2)),
            "observation_vr_right_thumbstick": _frame_vectors(
                frames, "observation", "vr_right_thumbstick", 2
            ),
            "observation_camera_frame_sequence": _frame_scalars(
                frames, "observation", "camera_frame_sequence"
            ),
            "observation_camera_timestamp_unix_s": _frame_scalars(
                frames, "observation", "camera_timestamp_unix_s"
            ),
            "observation_camera_age_s": _frame_scalars(
                frames, "observation", "camera_age_s"
            ),
            "observation_camera_rgb_path": _frame_strings(
                frames, "observation", "camera_rgb_path"
            ),
            "observation_camera_depth_path": _frame_strings(
                frames, "observation", "camera_depth_path"
            ),
            "action_target_tcp_pose_mm_deg": _frame_vectors(
                frames, "action", "target_tcp_pose_mm_deg", 6
            ),
            "action_executed": np.asarray(
                [frame["action"]["executed"] for frame in frames], dtype=np.bool_
            ),
            "action_gripper_desired_closed": _frame_scalars(
                frames, "action", "gripper_desired_closed"
            ),
            "action_gripper_applied_closed": _frame_scalars(
                frames, "action", "gripper_applied_closed"
            ),
            "action_gripper_target_position": _frame_scalars(
                frames, "action", "gripper_target_position"
            ),
            "action_gripper_executed": np.asarray(
                [
                    bool(frame["action"].get("gripper_executed", False))
                    for frame in frames
                ],
                dtype=np.bool_,
            ),
            "action_gripper_initialize_status": _frame_scalars(
                frames, "action", "gripper_initialize_status"
            ),
            "action_gripper_status": _frame_scalars(
                frames, "action", "gripper_status"
            ),
            "action_gripper_actual_position": _frame_scalars(
                frames, "action", "gripper_actual_position"
            ),
            "robot_mode": _frame_scalars(frames, "observation", "robot_mode"),
            "safety_state": _frame_scalars(frames, "observation", "safety_state"),
            "speed_scaling": _frame_scalars(frames, "observation", "speed_scaling"),
            "error_status": _frame_scalars(frames, "observation", "error_status"),
            "collision_state": _frame_scalars(frames, "observation", "collision_state"),
        }
        temporary = episode_dir / "data.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, episode_dir / "data.npz")

    @staticmethod
    def _empty_camera_record() -> dict[str, Any]:
        return {
            "sequence": None,
            "timestamp_unix_s": None,
            "age_s": None,
            "rgb_path": None,
            "depth_path": None,
        }

    def _queue_camera_frames(
        self,
        frames: dict[str, CameraFrame | None],
    ) -> dict[str, dict[str, Any]]:
        return {name: self._queue_camera_frame(name, frame) for name, frame in frames.items()}

    def _queue_camera_frame(self, name: str, frame: CameraFrame | None) -> dict[str, Any]:
        empty = self._empty_camera_record()
        if frame is None or self._episode_dir is None:
            return empty

        camera_name = _safe_name(name)
        if frame.sequence != self._last_camera_sequences.get(camera_name):
            rgb_path = f"images/{camera_name}/rgb/{frame.sequence:08d}.jpg"
            depth_path = (
                f"images/{camera_name}/depth/{frame.sequence:08d}.png"
                if frame.depth_mm is not None
                else None
            )
            self._ensure_image_writer()
            assert self._image_queue is not None
            try:
                self._image_queue.put_nowait(
                    (
                        self._episode_dir,
                        rgb_path,
                        depth_path,
                        frame.color_bgr,
                        frame.depth_mm,
                    )
                )
            except queue.Full:
                self._camera_frames_dropped += 1
                rgb_path = None
                depth_path = None
            else:
                self._camera_frames_enqueued += 1
            self._last_camera_sequences[camera_name] = frame.sequence
            self._last_camera_rgb_paths[camera_name] = rgb_path
            self._last_camera_depth_paths[camera_name] = depth_path

        return {
            "sequence": frame.sequence,
            "timestamp_unix_s": frame.timestamp_unix_s,
            "age_s": frame.age_s,
            "rgb_path": self._last_camera_rgb_paths.get(camera_name),
            "depth_path": self._last_camera_depth_paths.get(camera_name),
        }

    def _configured_cameras(self) -> list[dict[str, Any]]:
        configured: list[dict[str, Any]] = []
        values = [("top", self.config_snapshot.get("camera", {}))]
        additional = self.config_snapshot.get("cameras", {})
        if isinstance(additional, dict):
            values.extend((str(name), item) for name, item in additional.items())
        for fallback_name, camera in values:
            if not isinstance(camera, dict) or not bool(camera.get("enabled", False)):
                continue
            configured.append(
                {
                    "name": str(camera.get("name", fallback_name)),
                    "driver": camera.get("driver", "realsense"),
                    "serial": camera.get("serial", ""),
                    "width": camera.get("width"),
                    "height": camera.get("height"),
                    "fps": camera.get("fps"),
                    "depth": bool(camera.get("enable_depth", False)),
                }
            )
        return configured

    def _ensure_image_writer(self) -> None:
        if self._image_writer_thread is not None:
            return
        self._image_queue = queue.Queue(maxsize=self.image_queue_size)
        self._image_writer_thread = threading.Thread(
            target=self._image_writer_loop,
            name="camera-image-writer",
            daemon=False,
        )
        self._image_writer_thread.start()

    def _stop_image_writer(self) -> None:
        if self._image_writer_thread is None or self._image_queue is None:
            return
        self._image_queue.put(None)
        self._image_writer_thread.join()
        self._image_writer_thread = None
        self._image_queue = None

    def _image_writer_loop(self) -> None:
        assert self._image_queue is not None
        try:
            import cv2
        except ImportError as exc:
            self._image_writer_error = f"opencv-python is not installed: {exc}"
            cv2 = None

        while True:
            item = self._image_queue.get()
            if item is None:
                return
            episode_dir, rgb_path, depth_path, color_bgr, depth_mm = item
            if cv2 is None:
                self._camera_frames_dropped += 1
                continue
            try:
                rgb_destination = episode_dir / rgb_path
                rgb_destination.parent.mkdir(parents=True, exist_ok=True)
                written = cv2.imwrite(
                    str(rgb_destination),
                    color_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.image_jpeg_quality],
                )
                if not written:
                    raise RuntimeError(f"Failed to write {rgb_destination}")
                if depth_path is not None and depth_mm is not None:
                    depth_destination = episode_dir / depth_path
                    depth_destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(depth_destination), depth_mm):
                        raise RuntimeError(f"Failed to write {depth_destination}")
                self._camera_frames_written += 1
            except Exception as exc:
                self._camera_frames_dropped += 1
                if not self._image_writer_error:
                    self._image_writer_error = str(exc)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "vr_teleoperation"


def _vector(values: list[float], size: int) -> list[float]:
    if len(values) < size:
        raise ValueError(f"Expected at least {size} values, got {len(values)}")
    result = [float(value) for value in values[:size]]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Non-finite value in recorded vector")
    return result


def _optional_vector(values: list[float], size: int) -> list[float] | None:
    return _vector(values, size) if len(values) >= size else None


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _frame_vectors(
    frames: list[dict[str, Any]], group: str, key: str, size: int
) -> np.ndarray:
    rows = []
    for frame in frames:
        values = frame[group].get(key)
        rows.append(values if values is not None else [math.nan] * size)
    return np.asarray(rows, dtype=np.float64).reshape((-1, size))


def _frame_scalars(frames: list[dict[str, Any]], group: str, key: str) -> np.ndarray:
    values = []
    for frame in frames:
        value = frame[group].get(key)
        values.append(math.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def _frame_strings(frames: list[dict[str, Any]], group: str, key: str) -> np.ndarray:
    return np.asarray(
        [frame[group].get(key) or "" for frame in frames],
        dtype=np.str_,
    )
