from __future__ import annotations

import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any


class GripperError(RuntimeError):
    pass


@dataclass(frozen=True)
class GripperConfig:
    enabled: bool = False
    port: int = 60000
    slave_id: int = 1
    baud: int = 115200
    parity: str = "N"
    stop_bits: int = 1
    force: int = 20
    speed: int = 20
    initialize_register: int = 0x0100
    # 0x01 performs the normal homing cycle using the saved travel.
    # 0xA5 is reserved for a deliberate full recalibration.
    initialize_value: int = 0x0001
    force_register: int = 0x0101
    position_register: int = 0x0103
    speed_register: int = 0x0104
    open_position: int = 1000
    close_position: int = 0
    open_trigger_threshold: float = 0.20
    close_trigger_threshold: float = 0.70
    pressed_closes: bool = True
    require_initial_press_before_open: bool = True
    initialize_timeout_s: float = 5.0
    register_delay_s: float = 0.05
    command_interval_s: float = 0.10
    feedback_poll_interval_s: float = 0.10
    socket_timeout_s: float = 3.0
    write_retries: int = 3
    retry_delay_s: float = 0.50

    @staticmethod
    def from_mapping(values: dict[str, Any] | None) -> "GripperConfig":
        data = values or {}
        config = GripperConfig(
            enabled=bool(data.get("enabled", False)),
            port=int(data.get("port", 60000)),
            slave_id=int(data.get("slave_id", 1)),
            baud=int(data.get("baud", 115200)),
            parity=str(data.get("parity", "N")).upper(),
            stop_bits=int(data.get("stop_bits", 1)),
            force=int(data.get("force", 20)),
            speed=int(data.get("speed", 20)),
            initialize_register=_as_int(data.get("initialize_register", 0x0100)),
            initialize_value=_as_int(data.get("initialize_value", 0x0001)),
            force_register=_as_int(data.get("force_register", 0x0101)),
            position_register=_as_int(data.get("position_register", 0x0103)),
            speed_register=_as_int(data.get("speed_register", 0x0104)),
            open_position=int(data.get("open_position", 1000)),
            close_position=int(data.get("close_position", 0)),
            open_trigger_threshold=float(data.get("open_trigger_threshold", 0.20)),
            close_trigger_threshold=float(data.get("close_trigger_threshold", 0.70)),
            pressed_closes=bool(data.get("pressed_closes", True)),
            require_initial_press_before_open=bool(
                data.get("require_initial_press_before_open", True)
            ),
            initialize_timeout_s=float(data.get("initialize_timeout_s", 5.0)),
            register_delay_s=float(data.get("register_delay_s", 0.05)),
            command_interval_s=float(data.get("command_interval_s", 0.10)),
            feedback_poll_interval_s=float(data.get("feedback_poll_interval_s", 0.10)),
            socket_timeout_s=float(data.get("socket_timeout_s", 3.0)),
            write_retries=int(data.get("write_retries", 3)),
            retry_delay_s=float(data.get("retry_delay_s", 0.50)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("gripper.port must be in [1, 65535]")
        if not 0 <= self.slave_id <= 247:
            raise ValueError("gripper.slave_id must be in [0, 247]")
        if self.parity not in ("N", "E", "O"):
            raise ValueError("gripper.parity must be N, E, or O")
        if self.stop_bits not in (1, 2):
            raise ValueError("gripper.stop_bits must be 1 or 2")
        if not 20 <= self.force <= 100:
            raise ValueError("gripper.force must be in [20, 100]")
        if not 1 <= self.speed <= 100:
            raise ValueError("gripper.speed must be in [1, 100]")
        if self.initialize_value not in (0x01, 0xA5):
            raise ValueError("gripper.initialize_value must be 0x01 or 0xA5")
        for name, value in (
            ("open_position", self.open_position),
            ("close_position", self.close_position),
        ):
            if not 0 <= value <= 1000:
                raise ValueError(f"gripper.{name} must be in [0, 1000]")
        if not 0.0 <= self.open_trigger_threshold < self.close_trigger_threshold <= 1.0:
            raise ValueError(
                "gripper trigger thresholds must satisfy "
                "0 <= open < close <= 1"
            )
        if (
            self.initialize_timeout_s <= 0.0
            or self.register_delay_s < 0.0
            or self.command_interval_s < 0.0
            or self.feedback_poll_interval_s <= 0.0
            or self.retry_delay_s < 0.0
        ):
            raise ValueError("gripper timing values are invalid")
        if self.socket_timeout_s <= 0.0:
            raise ValueError("gripper.socket_timeout_s must be positive")
        if self.write_retries < 1:
            raise ValueError("gripper.write_retries must be at least 1")


class TriggerGripperController:
    """Maps Quest trigger values to a Dobot RS485 Modbus gripper."""

    def __init__(
        self,
        config: GripperConfig,
        robot_ip: str,
        execute: bool,
    ) -> None:
        self.config = config
        self.robot_ip = robot_ip
        self.execute = bool(execute)
        self._socket: socket.socket | None = None
        self._desired_closed: bool | None = None
        self._applied_closed: bool | None = None
        self._worker_error: Exception | None = None
        self._initialize_status: int | None = None
        self._grip_status: int | None = None
        self._actual_position: int | None = None
        self._initial_press_seen = False
        self._last_command_monotonic = 0.0
        self._state_lock = threading.Lock()
        self._command_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    @property
    def desired_closed(self) -> bool | None:
        with self._state_lock:
            return self._desired_closed

    @property
    def applied_closed(self) -> bool | None:
        with self._state_lock:
            return self._applied_closed

    def connect(self, dashboard: Any = None) -> None:
        if not self.config.enabled:
            return
        if not self.execute:
            print(
                "Gripper trigger control enabled in DRY mode: "
                f"open<={self.config.open_trigger_threshold:.2f}, "
                f"close>={self.config.close_trigger_threshold:.2f}"
            )
            return
        if dashboard is None:
            raise GripperError("A connected Dobot Dashboard object is required")

        self._dashboard_command(dashboard, "SetToolPower(1)")
        time.sleep(0.5)
        self._dashboard_command(
            dashboard,
            f'SetTool485({self.config.baud},"{self.config.parity}",'
            f"{self.config.stop_bits})",
        )
        self._dashboard_command(dashboard, "SetToolMode(1,0)")

        try:
            self._connect_gripper_socket()
            self._write_register_reliably(
                self.config.initialize_register,
                self.config.initialize_value,
            )
            self._wait_for_initialization()
            self._write_register_reliably(
                self.config.force_register,
                self.config.force,
            )
            time.sleep(self.config.register_delay_s)
            self._write_register_reliably(
                self.config.speed_register,
                self.config.speed,
            )
        except Exception:
            self.close()
            raise

        self._stop_event.clear()
        self._command_event.clear()
        with self._state_lock:
            self._worker_error = None
            self._applied_closed = None
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="dobot-gripper",
            daemon=True,
        )
        self._worker.start()
        print(
            f"Dobot gripper ready on {self.robot_ip}:{self.config.port}; "
            f"force={self.config.force}, speed={self.config.speed}"
        )

    def update(self, trigger_value: float, input_valid: bool = True) -> None:
        if not self.config.enabled:
            return
        self._raise_worker_error()
        if not input_valid:
            return

        trigger = min(max(float(trigger_value), 0.0), 1.0)
        desired: bool | None = None
        if trigger >= self.config.close_trigger_threshold:
            desired = self.config.pressed_closes
            if desired:
                self._initial_press_seen = True
        elif trigger <= self.config.open_trigger_threshold:
            desired = not self.config.pressed_closes
            if (
                not desired
                and self.config.require_initial_press_before_open
                and not self._initial_press_seen
            ):
                return
        if desired is None:
            return

        with self._state_lock:
            if desired == self._desired_closed:
                return
            self._desired_closed = desired

        state_name = "CLOSE" if desired else "OPEN"
        if not self.execute:
            with self._state_lock:
                self._applied_closed = desired
            print(f"[gripper DRY] trigger={trigger:.2f} -> {state_name}")
            return
        self._command_event.set()

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            desired = self._desired_closed
            applied = self._applied_closed
        target = None
        if desired is not None:
            target = (
                self.config.close_position if desired else self.config.open_position
            )
        return {
            "enabled": self.config.enabled,
            "desired_closed": desired,
            "applied_closed": applied,
            "target_position": target,
            "executed": bool(self.execute and applied is not None),
            "initialize_status": self._initialize_status,
            "grip_status": self._grip_status,
            "actual_position": self._actual_position,
            "error": str(self._worker_error) if self._worker_error else None,
        }

    def close(self) -> None:
        self._stop_event.set()
        self._command_event.set()
        if self._worker is not None:
            self._worker.join(timeout=self.config.socket_timeout_s + 1.0)
            self._worker = None
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                # A failed initialization can leave a socket object whose peer
                # has already closed. Cleanup should not mask that root error.
                pass
            finally:
                self._socket = None

    def _worker_loop(self) -> None:
        next_feedback = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            self._command_event.wait(timeout=0.05)
            self._command_event.clear()
            if self._stop_event.is_set():
                return
            try:
                with self._state_lock:
                    desired = self._desired_closed
                    applied = self._applied_closed
                if desired is not None and desired != applied:
                    remaining = self.config.command_interval_s - (
                        time.monotonic() - self._last_command_monotonic
                    )
                    if remaining > 0.0 and self._stop_event.wait(remaining):
                        return
                    position = self.config.close_position if desired else self.config.open_position
                    self._write_register_reliably(
                        self.config.position_register,
                        position,
                    )
                    self._last_command_monotonic = time.monotonic()
                    with self._state_lock:
                        self._applied_closed = desired
                    state_name = "CLOSE" if desired else "OPEN"
                    print(f"[gripper EXEC] {state_name} position={position}")

                if time.monotonic() >= next_feedback:
                    self._refresh_feedback()
                    next_feedback = time.monotonic() + self.config.feedback_poll_interval_s
            except Exception as exc:
                with self._state_lock:
                    self._worker_error = exc
                return

    def _wait_for_initialization(self) -> None:
        deadline = time.monotonic() + self.config.initialize_timeout_s
        while time.monotonic() < deadline:
            status = self._read_register_reliably(0x0200)
            self._initialize_status = status
            if status == 1:
                return
            time.sleep(0.05)
        raise GripperError(
            f"Gripper initialization timed out: status={self._initialize_status}"
        )

    def _refresh_feedback(self) -> None:
        self._initialize_status = self._read_register_reliably(0x0200)
        self._grip_status = self._read_register_reliably(0x0201)
        self._actual_position = self._read_register_reliably(0x0202)

    def _raise_worker_error(self) -> None:
        with self._state_lock:
            error = self._worker_error
        if error is not None:
            raise GripperError(f"Gripper worker failed: {error}") from error

    @staticmethod
    def _dashboard_command(dashboard: Any, command: str) -> None:
        send_receive = getattr(dashboard, "sendRecvMsg", None)
        if not callable(send_receive):
            raise GripperError("Dobot Dashboard does not provide sendRecvMsg")
        response = send_receive(command)
        match = re.search(r"-?\d+", str(response))
        if match is None or int(match.group()) != 0:
            raise GripperError(f"{command} failed: {response}")

    def _write_register(self, address: int, value: int) -> None:
        if self._socket is None:
            raise GripperError("Gripper port is not connected")
        pdu = bytes(
            [
                self.config.slave_id,
                0x06,
                address >> 8,
                address & 0xFF,
                value >> 8,
                value & 0xFF,
            ]
        )
        checksum = _crc16(pdu)
        frame = pdu + bytes([checksum & 0xFF, checksum >> 8])
        print(
            f"[gripper Modbus] write register=0x{address:04X} value={value}",
            flush=True,
        )
        self._socket.sendall(frame)
        reply = _receive_exact(self._socket, 8)
        reply_crc = reply[-2] | (reply[-1] << 8)
        if _crc16(reply[:-2]) != reply_crc:
            raise GripperError(f"Invalid Modbus CRC: {reply.hex(' ')}")
        if reply[1] & 0x80:
            raise GripperError(f"Modbus exception: {reply.hex(' ')}")
        if reply != frame:
            raise GripperError(
                f"Unexpected Modbus reply: sent={frame.hex(' ')} "
                f"received={reply.hex(' ')}"
            )
        print(
            f"[gripper Modbus] acknowledged register=0x{address:04X}",
            flush=True,
        )

    def _write_register_reliably(self, address: int, value: int) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.config.write_retries + 1):
            try:
                self._write_register(address, value)
                return
            except (OSError, GripperError) as exc:
                last_error = exc
                if attempt >= self.config.write_retries:
                    break
                print(
                    f"[gripper Modbus] register=0x{address:04X} "
                    f"attempt={attempt} failed: {exc}; reconnecting port "
                    f"{self.config.port}",
                    flush=True,
                )
                time.sleep(self.config.retry_delay_s)
                self._connect_gripper_socket()
        raise GripperError(
            f"Register 0x{address:04X} failed after "
            f"{self.config.write_retries} attempts: {last_error}"
        ) from last_error

    def _read_register(self, address: int) -> int:
        if self._socket is None:
            raise GripperError("Gripper port is not connected")
        pdu = bytes(
            [
                self.config.slave_id,
                0x03,
                address >> 8,
                address & 0xFF,
                0x00,
                0x01,
            ]
        )
        checksum = _crc16(pdu)
        frame = pdu + bytes([checksum & 0xFF, checksum >> 8])
        self._socket.sendall(frame)
        header = _receive_exact(self._socket, 3)
        if header[1] & 0x80:
            detail = _receive_exact(self._socket, 2)
            raise GripperError(f"Modbus read exception: {(header + detail).hex(' ')}")
        if header[1] != 0x03 or header[2] != 2:
            raise GripperError(f"Unexpected Modbus read header: {header.hex(' ')}")
        body = _receive_exact(self._socket, 4)
        reply = header + body
        reply_crc = reply[-2] | (reply[-1] << 8)
        if _crc16(reply[:-2]) != reply_crc:
            raise GripperError(f"Invalid Modbus CRC: {reply.hex(' ')}")
        return (reply[3] << 8) | reply[4]

    def _read_register_reliably(self, address: int) -> int:
        last_error: Exception | None = None
        for attempt in range(1, self.config.write_retries + 1):
            try:
                return self._read_register(address)
            except (OSError, GripperError) as exc:
                last_error = exc
                if attempt >= self.config.write_retries:
                    break
                time.sleep(self.config.retry_delay_s)
                self._connect_gripper_socket()
        raise GripperError(
            f"Register 0x{address:04X} read failed after "
            f"{self.config.write_retries} attempts: {last_error}"
        ) from last_error

    def _connect_gripper_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
        self._socket = socket.create_connection(
            (self.robot_ip, self.config.port),
            timeout=self.config.socket_timeout_s,
        )
        self._socket.settimeout(self.config.socket_timeout_s)


def _as_int(value: Any) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise GripperError("Gripper connection closed before the reply completed")
        data.extend(chunk)
    return bytes(data)
