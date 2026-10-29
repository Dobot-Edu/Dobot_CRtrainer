from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class MasterSyncTrace:
    """Low-rate JSONL trace for diagnosing leader/follower alignment."""

    def __init__(self, output_dir: str | Path, config: dict, interval_s: float = 0.2) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = self.output_dir / f"sync_{stamp}.jsonl"
        self.interval_s = max(float(interval_s), 0.05)
        self._last_tick = 0.0
        self._handle = self.path.open("w", encoding="utf-8", buffering=1)

        master = config.get("master_hand", {})
        profile_name = str(master.get("side", "auto"))
        profile = master.get("profiles", {}).get(profile_name, {})
        self.write(
            "session_start",
            {
                "profile": profile_name,
                "device": master.get("device"),
                "mapping": master.get("mapping", {}),
                "profile_calibration": profile,
            },
        )

    def write(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_s": round(time.monotonic(), 6),
            "event": str(event),
        }
        if payload:
            record.update(_json_safe(payload))
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_tick(
        self,
        *,
        sample: Any,
        robot: Any,
        command: Any,
        mapper: Any,
        button_state: Any,
        alignment_ready: bool,
        approach_active: bool,
        execute: bool,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_tick < self.interval_s:
            return
        self._last_tick = now
        master_rad = list(getattr(sample, "joint_positions_rad", []) or [])
        self.write(
            "control_tick",
            {
                "master": {
                    "profile": getattr(sample, "profile_name", None),
                    "raw_positions": list(getattr(sample, "raw_positions", []) or []),
                    "joint_rad": master_rad,
                    "joint_deg": [math.degrees(value) for value in master_rad],
                    "gripper_open": getattr(sample, "gripper_open", None),
                    "age_s": getattr(sample, "age_s", None),
                    "tracking_valid": getattr(sample, "tracking_valid", None),
                },
                "robot": _robot_payload(robot),
                "buttons": {
                    "servo_enabled": getattr(button_state, "servo_enabled", None),
                    "master_locked": getattr(button_state, "master_locked", None),
                },
                "alignment": {
                    "ready": bool(alignment_ready),
                    "approach_active": bool(approach_active),
                    "frozen_target_deg": getattr(mapper, "soft_alignment_target_deg", None),
                    "desired_target_deg": getattr(mapper, "desired_target_deg", None),
                    "mapper_aligned": getattr(mapper, "aligned", None),
                },
                "command": {
                    "allowed": getattr(command, "allowed", None),
                    "reason": getattr(command, "reason", None),
                    "target_joint_deg": getattr(
                        command, "target_joint_positions_deg", None
                    ),
                    "target_pose_mm_deg": getattr(command, "target_pose_mm_deg", None),
                },
                "execute": bool(execute),
            },
        )

    def record_button_event(
        self,
        event: Any,
        sample: Any,
        robot: Any,
        **extra: Any,
    ) -> None:
        master_rad = list(getattr(sample, "joint_positions_rad", []) or [])
        payload = {
            "button_event": {
                "action": getattr(event, "action", None),
                "value": getattr(event, "value", None),
                "duration_s": getattr(event, "duration_s", None),
            },
            "master": {
                "raw_positions": list(getattr(sample, "raw_positions", []) or []),
                "joint_rad": master_rad,
                "joint_deg": [math.degrees(value) for value in master_rad],
                "gripper_open": getattr(sample, "gripper_open", None),
            },
            "robot": _robot_payload(robot),
        }
        payload.update(extra)
        self.write("button_event", payload)

    def close(self) -> None:
        if self._handle.closed:
            return
        self.write("session_end")
        self._handle.close()


def write_calibration_report(
    output_dir: str | Path,
    *,
    profile_name: str,
    target_deg: list[float],
    sample_count: int,
    first_calibrated_rad: list[float],
    averaged_raw_positions: list[float],
    joint_signs: list[int] | tuple[int, ...],
    position_units_per_pi: float,
    calibration: dict[str, Any],
    saved_config_path: str | Path | None,
) -> Path:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"calibration_{stamp}.json"
    payload = {
        "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "profile": profile_name,
        "target_joint_deg": target_deg,
        "sample_count": int(sample_count),
        "calibrated_before_rad": first_calibrated_rad,
        "calibrated_before_deg": [
            math.degrees(value) for value in first_calibrated_rad
        ],
        "averaged_raw_positions": averaged_raw_positions,
        "joint_signs": list(joint_signs),
        "position_units_per_pi": float(position_units_per_pi),
        "computed_calibration": calibration,
        "saved_config_path": None if saved_config_path is None else str(saved_config_path),
    }
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_master_pose_report(
    output_dir: str | Path,
    *,
    profile_name: str,
    reference_joint_deg: list[float],
    current_joint_rad: list[float],
    relative_joint_deg: list[float],
    averaged_raw_positions: list[float],
    sample_count: int,
    loaded_joint_offsets_rad: list[float] | tuple[float, ...],
    joint_signs: list[int] | tuple[int, ...],
) -> Path:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"master_pose_{stamp}.json"
    payload = {
        "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "mode": "inspect_only",
        "profile": profile_name,
        "reference_joint_deg": reference_joint_deg,
        "current_joint_rad": current_joint_rad,
        "current_joint_deg": [math.degrees(value) for value in current_joint_rad],
        "relative_to_reference_deg": relative_joint_deg,
        "averaged_raw_positions": averaged_raw_positions,
        "sample_count": int(sample_count),
        "loaded_joint_offsets_rad": list(loaded_joint_offsets_rad),
        "joint_signs": list(joint_signs),
        "configuration_changed": False,
    }
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _robot_payload(robot: Any) -> dict[str, Any]:
    return {
        "connected": getattr(robot, "connected", None),
        "enabled": getattr(robot, "enabled", None),
        "mode": getattr(robot, "robot_mode", None),
        "collision_state": getattr(robot, "collision_state", None),
        "error_status": getattr(robot, "error_status", None),
        "feedback_age_s": getattr(robot, "feedback_age_s", None),
        "joint_actual_deg": list(getattr(robot, "q_actual_deg", []) or []),
        "joint_target_deg": list(getattr(robot, "q_target_deg", []) or []),
        "pose_mm_deg": list(getattr(robot, "pose_mm_deg", []) or []),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
