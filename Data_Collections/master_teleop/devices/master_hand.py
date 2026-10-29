from __future__ import annotations

import importlib
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..control.models import MasterHandInput


class MasterHandError(RuntimeError):
    pass


@dataclass(frozen=True)
class MasterHandProfile:
    name: str
    joint_ids: tuple[int, ...]
    button_board_id: int
    gripper_id: int
    joint_offsets_rad: tuple[float, ...]
    joint_signs: tuple[int, ...]
    start_joints_rad: tuple[float, ...]
    gripper_closed_deg: float
    gripper_open_deg: float

    @staticmethod
    def from_mapping(name: str, values: dict[str, Any]) -> "MasterHandProfile":
        profile = MasterHandProfile(
            name=name,
            joint_ids=tuple(int(value) for value in values.get("joint_ids", [])),
            button_board_id=int(values.get("button_board_id", -1)),
            gripper_id=int(values.get("gripper_id", -1)),
            joint_offsets_rad=tuple(float(value) for value in values.get("joint_offsets_rad", [])),
            joint_signs=tuple(int(value) for value in values.get("joint_signs", [])),
            start_joints_rad=tuple(float(value) for value in values.get("start_joints_rad", [])),
            gripper_closed_deg=float(values.get("gripper_closed_deg", 0.0)),
            gripper_open_deg=float(values.get("gripper_open_deg", 1.0)),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if len(self.joint_ids) != 6 or len(set(self.joint_ids)) != 6:
            raise ValueError(f"master_hand profile {self.name} requires six unique joint IDs")
        if len(self.joint_offsets_rad) != 6 or len(self.joint_signs) != 6:
            raise ValueError(f"master_hand profile {self.name} requires six offsets and signs")
        if len(self.start_joints_rad) != 6:
            raise ValueError(f"master_hand profile {self.name} requires six start joints")
        if any(sign not in (-1, 1) for sign in self.joint_signs):
            raise ValueError(f"master_hand profile {self.name} signs must be -1 or 1")
        if self.gripper_id < 0 or self.button_board_id < 0:
            raise ValueError(f"master_hand profile {self.name} IDs must be non-negative")
        if self.gripper_closed_deg == self.gripper_open_deg:
            raise ValueError(f"master_hand profile {self.name} gripper endpoints must differ")


@dataclass(frozen=True)
class MasterHandConfig:
    device: str
    baud_rate: int
    baud_rate_candidates: tuple[int, ...]
    protocol_version: float
    side: str
    read_hz: float
    watchdog_timeout_s: float
    present_position_address: int
    present_position_length: int
    position_units_per_pi: float
    buttons_active_low: bool
    torque_enable_address: int
    goal_position_address: int
    sync_goal_before_lock: bool
    profiles: dict[str, MasterHandProfile]

    @staticmethod
    def from_mapping(values: dict[str, Any] | None) -> "MasterHandConfig":
        data = values or {}
        profile_values = data.get("profiles", {})
        if not isinstance(profile_values, dict):
            raise ValueError("master_hand.profiles must be a mapping")
        profiles = {
            str(name): MasterHandProfile.from_mapping(str(name), profile)
            for name, profile in profile_values.items()
            if isinstance(profile, dict)
        }
        config = MasterHandConfig(
            device=str(data.get("device", "")).strip(),
            baud_rate=int(data.get("baud_rate", 1000000)),
            baud_rate_candidates=tuple(
                int(value)
                for value in data.get(
                    "baud_rate_candidates",
                    [data.get("baud_rate", 1000000)],
                )
            ),
            protocol_version=float(data.get("protocol_version", 2.0)),
            side=str(data.get("side", "auto")).strip().lower(),
            read_hz=float(data.get("read_hz", 100.0)),
            watchdog_timeout_s=float(data.get("watchdog_timeout_s", 0.10)),
            present_position_address=int(data.get("present_position_address", 140)),
            present_position_length=int(data.get("present_position_length", 4)),
            position_units_per_pi=float(data.get("position_units_per_pi", 2048.0)),
            buttons_active_low=bool(data.get("buttons_active_low", True)),
            torque_enable_address=int(data.get("torque_enable_address", 64)),
            goal_position_address=int(data.get("goal_position_address", 116)),
            sync_goal_before_lock=bool(data.get("sync_goal_before_lock", False)),
            profiles=profiles,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.device:
            raise ValueError("master_hand.device must not be empty")
        if self.baud_rate <= 0 or self.read_hz <= 0.0 or self.watchdog_timeout_s <= 0.0:
            raise ValueError("master_hand baud/read/watchdog values must be positive")
        if not self.baud_rate_candidates or any(
            value <= 0 for value in self.baud_rate_candidates
        ):
            raise ValueError("master_hand.baud_rate_candidates must contain positive values")
        if self.present_position_length != 4:
            raise ValueError("master_hand present position length must be four bytes")
        if self.position_units_per_pi <= 0.0:
            raise ValueError("master_hand.position_units_per_pi must be positive")
        if self.torque_enable_address < 0 or self.goal_position_address < 0:
            raise ValueError("master_hand torque/goal addresses must be non-negative")
        if not self.profiles:
            raise ValueError("master_hand requires at least one profile")
        if self.side != "auto" and self.side not in self.profiles:
            raise ValueError("master_hand.side must be auto or a configured profile name")


class MockMasterHand:
    def __init__(self, profile_name: str = "mock") -> None:
        self.profile_name = profile_name
        self._connected = False
        self._button_a = False
        self._button_b = False
        self._joints = [0.0] * 6
        self._gripper_open = 1.0
        self.locked = False

    def connect(self) -> None:
        self._connected = True

    def close(self, release_torque: bool = True) -> None:
        self._connected = False

    def set_locked(self, locked: bool) -> None:
        self.locked = bool(locked)

    def latest(self) -> MasterHandInput | None:
        if not self._connected:
            return None
        return MasterHandInput(
            received_monotonic=time.monotonic(),
            timestamp=time.time(),
            joint_positions_rad=list(self._joints),
            gripper_open=self._gripper_open,
            button_a=self._button_a,
            button_b=self._button_b,
            buttons_raw=[int(not self._button_a), int(not self._button_b)],
            profile_name=self.profile_name,
        )

    def set_mock_state(
        self,
        joints_rad: list[float],
        gripper_open: float = 1.0,
        button_a: bool = False,
        button_b: bool = False,
    ) -> None:
        if len(joints_rad) != 6:
            raise ValueError("mock master hand requires six joints")
        self._joints = [float(value) for value in joints_rad]
        self._gripper_open = float(gripper_open)
        self._button_a = bool(button_a)
        self._button_b = bool(button_b)


class DynamixelMasterHand:
    """Read one Xtrainer-style hand and optionally lock it at its current pose."""

    def __init__(self, config: MasterHandConfig) -> None:
        self.config = config
        self.profile: MasterHandProfile | None = None
        self.active_baud_rate: int | None = None
        self._sdk: Any = None
        self._port: Any = None
        self._packet: Any = None
        self._group_read: Any = None
        self._latest: MasterHandInput | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._locked = False
        self._torque_was_controlled = False

    def connect(self) -> None:
        device = Path(self.config.device)
        if not device.exists():
            raise MasterHandError(f"Master hand serial device does not exist: {device}")
        self._sdk = _load_dynamixel_sdk()
        self._port = self._sdk.PortHandler(str(device))
        self._packet = self._sdk.PacketHandler(self.config.protocol_version)
        try:
            opened = self._port.openPort()
        except Exception as exc:
            raise MasterHandError(
                f"Failed to open master hand serial device {device}: {exc}. "
                "Ensure the current user belongs to the dialout group and has logged in again."
            ) from exc
        if not opened:
            raise MasterHandError(f"Failed to open master hand serial device: {device}")
        try:
            self.profile = self._select_profile()
            ids = (*self.profile.joint_ids, self.profile.gripper_id)
            self._group_read = self._sdk.GroupSyncRead(
                self._port,
                self._packet,
                self.config.present_position_address,
                self.config.present_position_length,
            )
            for motor_id in ids:
                if not self._group_read.addParam(motor_id):
                    raise MasterHandError(f"Failed to register master motor ID {motor_id}")
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._read_loop,
                name="single-master-hand-reader",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self.close()
            raise

    def close(self, release_torque: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if (
            release_torque
            and self._port is not None
            and self._torque_was_controlled
            and self._locked
        ):
            try:
                self.set_locked(False)
            except Exception as exc:
                print(f"[master warning] failed to release torque during close: {exc}", flush=True)
        if self._port is not None:
            try:
                self._port.closePort()
            except Exception:
                pass
        self._port = None

    @property
    def locked(self) -> bool:
        return self._locked

    def set_locked(self, locked: bool) -> None:
        """Lock/release all leader joints, gripper and the append/button board.

        Xtrainer masters normally lock by changing Torque Enable only. Optional
        goal synchronization is retained for other hardware, but is disabled
        for this hand because ID 17 rejects its present value as a goal value.
        """
        if self.profile is None or self._port is None or self._packet is None:
            raise MasterHandError("Master hand must be connected before changing its lock")
        locked = bool(locked)
        if locked == self._locked and self._torque_was_controlled:
            return
        motor_ids = (*self.profile.joint_ids, self.profile.gripper_id)
        with self._io_lock:
            if locked and self.config.sync_goal_before_lock:
                if self._group_read is None:
                    raise MasterHandError("Master hand position reader is unavailable")
                communication = self._group_read.txRxPacket()
                if communication != self._sdk.COMM_SUCCESS:
                    raise MasterHandError(
                        f"Cannot capture leader lock pose; sync read failed: {communication}"
                    )
                for motor_id in motor_ids:
                    raw_position = self._read_position(motor_id) & 0xFFFFFFFF
                    result, error = self._packet.write4ByteTxRx(
                        self._port,
                        motor_id,
                        self.config.goal_position_address,
                        raw_position,
                    )
                    self._check_write(result, error, motor_id, "goal position")
            torque_value = 1 if locked else 0
            for motor_id in (*motor_ids, self.profile.button_board_id):
                result, error = self._packet.write1ByteTxRx(
                    self._port,
                    motor_id,
                    self.config.torque_enable_address,
                    torque_value,
                )
                self._check_write(result, error, motor_id, "torque")
        self._locked = locked
        self._torque_was_controlled = True

    def _check_write(self, result: int, error: int, motor_id: int, operation: str) -> None:
        if result != self._sdk.COMM_SUCCESS or error != 0:
            detail = (
                self._packet.getTxRxResult(result)
                if result != self._sdk.COMM_SUCCESS
                else self._packet.getRxPacketError(error)
            )
            raise MasterHandError(
                f"Failed to write leader {operation} for ID {motor_id}: {detail}"
            )

    def latest(self) -> MasterHandInput | None:
        with self._lock:
            sample = self._latest
            error = self._error
        if sample is None and error is not None:
            raise MasterHandError(f"Master hand read failed: {error}") from error
        return sample

    def discovered_profile(self) -> str:
        return self.profile.name if self.profile is not None else ""

    def _select_profile(self) -> MasterHandProfile:
        candidates = (
            [self.config.profiles[self.config.side]]
            if self.config.side != "auto"
            else list(self.config.profiles.values())
        )
        baud_rates = list(dict.fromkeys(self.config.baud_rate_candidates))
        results: list[tuple[int, MasterHandProfile, int, list[int]]] = []
        for baud_rate in baud_rates:
            if not self._port.setBaudRate(baud_rate):
                continue
            for profile in candidates:
                found = self._probe_profile(profile)
                results.append((len(found), profile, baud_rate, found))
                if len(found) == len(profile.joint_ids) + 1:
                    self.active_baud_rate = baud_rate
                    print(
                        f"Single master hand detected: profile={profile.name} "
                        f"baud={baud_rate} joints={list(profile.joint_ids)} "
                        f"gripper={profile.gripper_id}",
                        flush=True,
                    )
                    return profile
        if not results:
            raise MasterHandError("Could not set any configured master-hand baud rate")
        score, profile, baud_rate, found = max(results, key=lambda item: item[0])
        required_count = len(profile.joint_ids) + 1
        if score != required_count:
            summary = "; ".join(
                f"{item_profile.name}@{item_baud}:found={item_found}"
                for _, item_profile, item_baud, item_found in results
            )
            raise MasterHandError(
                "Could not identify a complete single master hand by position read; "
                + summary
            )
        self.active_baud_rate = baud_rate
        return profile

    def _probe_profile(self, profile: MasterHandProfile) -> list[int]:
        assert self._packet is not None and self._port is not None
        ids = [*profile.joint_ids, profile.gripper_id]
        group_read = self._sdk.GroupSyncRead(
            self._port,
            self._packet,
            self.config.present_position_address,
            self.config.present_position_length,
        )
        for motor_id in ids:
            if not group_read.addParam(motor_id):
                return []
        with self._io_lock:
            group_read.txRxPacket()
            return [
                motor_id
                for motor_id in ids
                if group_read.isAvailable(
                    motor_id,
                    self.config.present_position_address,
                    self.config.present_position_length,
                )
            ]

    def _read_loop(self) -> None:
        period = 1.0 / self.config.read_hz
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                sample = self._read_once()
                with self._lock:
                    self._latest = sample
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = exc
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                self._stop.wait(remaining)

    def _read_once(self) -> MasterHandInput:
        assert self.profile is not None and self._group_read is not None
        with self._io_lock:
            communication = self._group_read.txRxPacket()
            if communication != self._sdk.COMM_SUCCESS:
                raise MasterHandError(f"Dynamixel sync read failed: {communication}")
            raw_positions = [
                self._read_position(motor_id)
                for motor_id in (*self.profile.joint_ids, self.profile.gripper_id)
            ]
            buttons = self._read_buttons()

        raw_joint_rad = [
            value / self.config.position_units_per_pi * math.pi
            for value in raw_positions[:6]
        ]
        joints = [
            (value - offset) * sign
            for value, offset, sign in zip(
                raw_joint_rad,
                self.profile.joint_offsets_rad,
                self.profile.joint_signs,
            )
        ]
        gripper_deg = raw_positions[6] / self.config.position_units_per_pi * 180.0
        gripper_open = (gripper_deg - self.profile.gripper_closed_deg) / (
            self.profile.gripper_open_deg - self.profile.gripper_closed_deg
        )
        gripper_open = min(max(gripper_open, 0.0), 1.0)
        button_a = len(buttons) >= 1 and self._button_pressed(buttons[0])
        button_b = len(buttons) >= 2 and self._button_pressed(buttons[1])
        return MasterHandInput(
            received_monotonic=time.monotonic(),
            timestamp=time.time(),
            joint_positions_rad=joints,
            gripper_open=gripper_open,
            button_a=button_a,
            button_b=button_b,
            buttons_raw=buttons,
            tracking_valid=True,
            profile_name=self.profile.name,
            raw_positions=raw_positions,
        )

    def _read_position(self, motor_id: int) -> int:
        if not self._group_read.isAvailable(
            motor_id,
            self.config.present_position_address,
            self.config.present_position_length,
        ):
            raise MasterHandError(f"No position data for master motor ID {motor_id}")
        value = int(
            self._group_read.getData(
                motor_id,
                self.config.present_position_address,
                self.config.present_position_length,
            )
        )
        return value - (1 << 32) if value & (1 << 31) else value

    def _read_buttons(self) -> list[int]:
        assert self._port is not None
        self._port.writePort([0xAA, 0x55, 0xAA])
        self._port.setPacketTimeout(2)
        response: list[int] = []
        while len(response) < 2 and not self._port.isPacketTimeout():
            response.extend(self._port.readPort(2 - len(response)))
        if len(response) != 2:
            return []
        if (response[0] | response[1]) != 0xFF or (response[0] & response[1]) != 0x00:
            return []
        return [(response[0] >> 4) & 0x0F, response[0] & 0x0F]

    def _button_pressed(self, value: int) -> bool:
        return value == 0 if self.config.buttons_active_low else value != 0


def _load_dynamixel_sdk() -> Any:
    try:
        return importlib.import_module("dynamixel_sdk")
    except ImportError:
        project_root = Path(__file__).resolve().parents[3]
        vendored = project_root / "dobot_xtrainer" / "third_party" / "DynamixelSDK" / "python" / "src"
        if vendored.exists():
            sys.path.insert(0, str(vendored))
            try:
                return importlib.import_module("dynamixel_sdk")
            except ImportError:
                pass
        raise MasterHandError(
            "Dynamixel SDK is unavailable. Install pyserial and dynamixel-sdk "
            "in the dobot-vr-teleop environment."
        )
