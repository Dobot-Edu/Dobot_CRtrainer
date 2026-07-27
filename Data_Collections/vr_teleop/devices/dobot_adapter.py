from __future__ import annotations

import importlib
import re
import socket
import sys
import threading
from pathlib import Path
from time import monotonic
from typing import Any

from ..control.models import Pose6, RobotState


class RobotAdapterError(RuntimeError):
    pass


class BaseRobotAdapter:
    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def state(self) -> RobotState:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def servo_p(self, pose_mm_deg: Pose6, t: float, aheadtime: float, gain: float) -> None:
        raise NotImplementedError

    def set_speed_factor(self, percent: int) -> None:
        raise NotImplementedError

    def move_j_pose(
        self,
        pose_mm_deg: Pose6,
        acceleration: int = 10,
        velocity: int = 10,
    ) -> None:
        raise NotImplementedError


class MockRobotAdapter(BaseRobotAdapter):
    def __init__(self, initial_pose: Pose6) -> None:
        self._state = RobotState(connected=False, enabled=True, pose_mm_deg=list(initial_pose))
        self.commands_sent = 0

    def connect(self) -> None:
        self._state.connected = True
        self._state.last_feedback_monotonic = monotonic()

    def close(self) -> None:
        self._state.connected = False

    def state(self) -> RobotState:
        self._state.last_feedback_monotonic = monotonic()
        return self._state

    def stop(self) -> None:
        self._state.message = "mock_stop"

    def servo_p(self, pose_mm_deg: Pose6, t: float, aheadtime: float, gain: float) -> None:
        self.commands_sent += 1
        self._state.pose_mm_deg = list(pose_mm_deg)
        self._state.last_feedback_monotonic = monotonic()
        self._state.message = f"mock_servo_p:{self.commands_sent}"

    def set_speed_factor(self, percent: int) -> None:
        self._state.speed_scaling = float(percent)
        self._state.message = f"mock_speed_factor:{percent}"

    def move_j_pose(
        self,
        pose_mm_deg: Pose6,
        acceleration: int = 10,
        velocity: int = 10,
    ) -> None:
        self._state.pose_mm_deg = list(pose_mm_deg)
        self._state.last_feedback_monotonic = monotonic()
        self._state.message = (
            f"mock_move_j:a={acceleration},v={velocity}"
        )


def _load_dobot_api() -> Any:
    project_root = Path(__file__).resolve().parents[3]
    for candidate in (
        project_root / "scripts",
        project_root / "vendor",
        project_root / "src",
    ):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    try:
        return importlib.import_module("dobot_api")
    except ImportError as exc:
        raise RobotAdapterError(
            "Could not import dobot_api.py. Place the official Dobot TCP-IP-Python-V4 "
            "dobot_api.py under scripts/, vendor/, or src/, or install it on PYTHONPATH."
        ) from exc


class DobotTcpAdapter(BaseRobotAdapter):
    def __init__(
        self,
        robot_ip: str,
        dashboard_port: int,
        move_port: int,
        feedback_port: int,
        lock_j6: bool = False,
    ) -> None:
        self.robot_ip = robot_ip
        self.dashboard_port = dashboard_port
        self.move_port = move_port
        self.feedback_port = feedback_port
        self.dashboard: Any = None
        self.move: Any = None
        self.feedback: Any = None
        self.lock_j6 = bool(lock_j6)
        self._j6_lock_deg: float | None = None
        self._state = RobotState(connected=False)
        self._feedback_stop = threading.Event()
        self._feedback_thread: threading.Thread | None = None

    def connect(self) -> None:
        api = _load_dobot_api()
        required_ports = [self.dashboard_port, self.feedback_port]
        if hasattr(api, "DobotApiMove"):
            required_ports.append(self.move_port)
        for port in sorted(set(required_ports)):
            _probe_tcp_port(self.robot_ip, port)

        self.dashboard = api.DobotApiDashboard(self.robot_ip, self.dashboard_port)
        move_class = getattr(api, "DobotApiMove", None)
        self.move = move_class(self.robot_ip, self.move_port) if move_class else self.dashboard
        if hasattr(api, "DobotApiFeedBack"):
            self.feedback = api.DobotApiFeedBack(self.robot_ip, self.feedback_port)
        self._state.connected = True
        self._refresh_pose_best_effort()
        if self.lock_j6:
            self._refresh_joints_best_effort()
            if len(self._state.q_actual_deg) < 6:
                raise RobotAdapterError(
                    "J6 lock requires six joint feedback values before teleoperation"
                )
            self._j6_lock_deg = float(self._state.q_actual_deg[5])
            print(f"Dobot J6 lock enabled at {self._j6_lock_deg:.3f} deg")
        if self.feedback is not None:
            self._feedback_stop.clear()
            self._feedback_thread = threading.Thread(
                target=self._feedback_loop,
                name="dobot-feedback",
                daemon=True,
            )
            self._feedback_thread.start()

    def close(self) -> None:
        self._feedback_stop.set()
        if self.feedback is not None:
            _close_api_object(self.feedback)
        if self._feedback_thread is not None:
            self._feedback_thread.join(timeout=1.0)
            self._feedback_thread = None
        for obj in _unique_objects(self.move, self.dashboard):
            _close_api_object(obj)
        self._state.connected = False

    def state(self) -> RobotState:
        if self.feedback is None:
            self._refresh_pose_best_effort()
        return self._state

    def stop(self) -> None:
        if self.dashboard is None:
            return
        method = getattr(self.dashboard, "Stop", None)
        if callable(method):
            _require_success(method(), "Stop")
            self._state.message = "Stop"

    def pause(self) -> None:
        if self.dashboard is None:
            return
        method = getattr(self.dashboard, "Pause", None)
        if callable(method):
            _require_success(method(), "Pause")
            self._state.message = "Pause"

    def servo_p(self, pose_mm_deg: Pose6, t: float, aheadtime: float, gain: float) -> None:
        if self.move is None:
            raise RobotAdapterError("Robot is not connected")
        if self.lock_j6:
            self._servo_j_with_locked_j6(pose_mm_deg, t, aheadtime, gain)
            return
        method = getattr(self.move, "ServoP", None)
        if not callable(method):
            raise RobotAdapterError("ServoP is not available in the selected Dobot API class")
        response = method(
            pose_mm_deg[0],
            pose_mm_deg[1],
            pose_mm_deg[2],
            pose_mm_deg[3],
            pose_mm_deg[4],
            pose_mm_deg[5],
            t=t,
            aheadtime=aheadtime,
            gain=gain,
        )
        _require_success(response, "ServoP")

    def _servo_j_with_locked_j6(
        self,
        pose_mm_deg: Pose6,
        t: float,
        aheadtime: float,
        gain: float,
    ) -> None:
        if self.dashboard is None or self.move is None:
            raise RobotAdapterError("Robot is not connected")
        if self._j6_lock_deg is None:
            raise RobotAdapterError("J6 lock angle has not been initialized")

        if len(self._state.q_actual_deg) < 6:
            self._refresh_joints_best_effort()
        near = self._state.q_actual_deg
        if len(near) < 6:
            raise RobotAdapterError("J6 lock requires six joint feedback values")

        response = _inverse_kin_with_recovery(self.dashboard, pose_mm_deg, near)
        target_joints = _parse_pose_response(response)
        if target_joints is None or len(target_joints) < 6:
            raise RobotAdapterError(f"InverseKin returned invalid joints: {response!r}")
        target_joints[5] = self._j6_lock_deg

        method = getattr(self.move, "ServoJ", None)
        if not callable(method):
            raise RobotAdapterError("ServoJ is required for J6 lock mode")
        response = method(
            target_joints[0],
            target_joints[1],
            target_joints[2],
            target_joints[3],
            target_joints[4],
            target_joints[5],
            t=t,
            aheadtime=aheadtime,
            gain=gain,
        )
        _require_success(response, "ServoJ(J6 locked)")

    def set_speed_factor(self, percent: int) -> None:
        if self.dashboard is None:
            raise RobotAdapterError("Robot is not connected")
        if not 1 <= percent <= 100:
            raise RobotAdapterError(f"SpeedFactor must be in [1, 100], got {percent}")
        method = getattr(self.dashboard, "SpeedFactor", None)
        if not callable(method):
            raise RobotAdapterError("SpeedFactor is not available in the selected Dobot API")
        _require_success(method(percent), "SpeedFactor")
        self._state.speed_scaling = float(percent)

    def move_j_pose(self, pose_mm_deg: Pose6, acceleration: int = 10, velocity: int = 10) -> None:
        if self.dashboard is None:
            raise RobotAdapterError("Robot is not connected")
        method = getattr(self.dashboard, "MovJ", None)
        if not callable(method):
            raise RobotAdapterError("MovJ is not available in the selected Dobot API")
        response = method(
            pose_mm_deg[0],
            pose_mm_deg[1],
            pose_mm_deg[2],
            pose_mm_deg[3],
            pose_mm_deg[4],
            pose_mm_deg[5],
            0,
            a=acceleration,
            v=velocity,
        )
        _require_success(response, "MovJ")

    def clear_error(self) -> None:
        if self.dashboard is not None and hasattr(self.dashboard, "ClearError"):
            _require_success(self.dashboard.ClearError(), "ClearError")

    def disable_robot(self) -> None:
        if self.dashboard is not None and hasattr(self.dashboard, "DisableRobot"):
            _require_success(self.dashboard.DisableRobot(), "DisableRobot")
            self._state.enabled = False

    def enable_robot(self) -> None:
        if self.dashboard is not None and hasattr(self.dashboard, "EnableRobot"):
            response = self.dashboard.EnableRobot()
            _require_success(response, "EnableRobot")
            self._state.enabled = True

    def _feedback_loop(self) -> None:
        while not self._feedback_stop.is_set():
            self._refresh_feedback_best_effort()

    def _refresh_pose_best_effort(self) -> None:
        if self.dashboard is None:
            return
        get_pose = getattr(self.dashboard, "GetPose", None)
        if not callable(get_pose):
            return
        try:
            response = get_pose()
            pose = _parse_pose_response(response)
            if pose:
                self._state.pose_mm_deg = pose
                self._state.last_feedback_monotonic = monotonic()
        except Exception as exc:
            self._state.message = f"GetPose failed: {exc}"

    def _refresh_joints_best_effort(self) -> None:
        if self.dashboard is None:
            return
        get_angle = getattr(self.dashboard, "GetAngle", None)
        if not callable(get_angle):
            return
        try:
            response = get_angle()
            joints = _parse_pose_response(response)
            if joints is not None and len(joints) >= 6:
                self._state.q_actual_deg = joints[:6]
        except Exception as exc:
            self._state.message = f"GetAngle failed: {exc}"

    def _refresh_feedback_best_effort(self) -> None:
        if self.feedback is None:
            self._refresh_pose_best_effort()
            return
        feed_method = getattr(self.feedback, "feedBackData", None)
        if not callable(feed_method):
            self._refresh_pose_best_effort()
            return
        try:
            data = feed_method()
            self._state.last_feedback_monotonic = monotonic()
            _copy_feedback_fields(data, self._state)
        except Exception as exc:
            self._state.message = f"feedback failed: {exc}"
            if not self._feedback_stop.is_set():
                self._refresh_pose_best_effort()


def _parse_pose_response(response: Any) -> Pose6 | None:
    if isinstance(response, (list, tuple)) and len(response) >= 6:
        return [float(v) for v in response[:6]]
    text = str(response)
    start = text.find("{")
    end = text.find("}", start + 1)
    if start < 0 or end < 0:
        return None
    values = text[start + 1 : end].split(",")
    if len(values) < 6:
        return None
    try:
        return [float(v.strip()) for v in values[:6]]
    except ValueError:
        return None


def _inverse_kin_with_joint_near(
    dashboard: Any,
    pose_mm_deg: Pose6,
    joint_near: str,
) -> Any:
    send_receive = getattr(dashboard, "sendRecvMsg", None)
    if not callable(send_receive):
        raise RobotAdapterError("InverseKin requires a Dobot Dashboard connection")
    command = (
        "InverseKin({:f},{:f},{:f},{:f},{:f},{:f},"
        "useJointNear=1,JointNear={})"
    ).format(*pose_mm_deg, joint_near)
    return send_receive(command)


def _inverse_kin_with_recovery(
    dashboard: Any,
    pose_mm_deg: Pose6,
    near: list[float],
) -> Any:
    """Solve IK with small seed perturbations around a wrist singularity.

    The robot can be physically at a valid pose while its IK solver rejects
    an exactly singular JointNear seed. These retries only query IK; they do
    not move the robot. The selected solution is still forced to the locked
    J6 angle by the caller before ServoJ is sent.
    """
    seeds: list[list[float]] = [list(near[:6])]
    for offset in (1.0, -1.0, 2.0, -2.0):
        seed = list(near[:6])
        seed[4] += offset
        seeds.append(seed)

    # The confirmed initial pose is close to a wrist singularity. If the
    # exact orientation has no numerical IK solution, query nearby poses by
    # at most one degree. This is only an IK query; no motion is sent here.
    pose_candidates: list[tuple[Pose6, float]] = [(list(pose_mm_deg), 0.0)]
    for orientation_offset in (0.5, -0.5, 1.0, -1.0):
        candidate = list(pose_mm_deg)
        candidate[4] += orientation_offset
        pose_candidates.append((candidate, orientation_offset))

    last_response: Any = ""
    for pose_candidate, orientation_offset in pose_candidates:
        for seed_index, seed in enumerate(seeds):
            joint_near = "{" + ",".join(f"{value:.6f}" for value in seed) + "}"
            response = _inverse_kin_with_joint_near(
                dashboard,
                pose_candidate,
                joint_near,
            )
            last_response = response
            try:
                _require_success(response, "InverseKin")
                if seed_index or orientation_offset:
                    print(
                        "[Dobot IK] recovered with "
                        f"J5 seed offset={seed[4] - near[4]:+.1f} deg, "
                        f"Ry IK bias={orientation_offset:+.1f} deg",
                        flush=True,
                    )
                return response
            except RobotAdapterError:
                if "-50001" not in str(response):
                    raise
                print(
                    "[Dobot IK] singular IK seed rejected; retrying "
                    f"J5 offset={seed[4] - near[4]:+.1f} deg, "
                    f"Ry bias={orientation_offset:+.1f} deg",
                    flush=True,
                )
    _require_success(last_response, "InverseKin")
    return last_response


def _field(data: Any, name: str) -> Any:
    if hasattr(data, name):
        value = getattr(data, name)
        try:
            return value[0]
        except Exception:
            return value
    try:
        return data[name][0]
    except Exception:
        return None


def _copy_feedback_fields(data: Any, state: RobotState) -> None:
    pose = _field(data, "ToolVectorActual")
    if pose is not None and len(pose) >= 6:
        state.pose_mm_deg = [float(v) for v in pose[:6]]

    q_actual = _field(data, "QActual")
    if q_actual is not None and len(q_actual) >= 6:
        state.q_actual_deg = [float(v) for v in q_actual[:6]]

    q_actual_velocity = _field(data, "QDActual")
    if q_actual_velocity is not None and len(q_actual_velocity) >= 6:
        state.q_actual_velocity_deg_s = [float(v) for v in q_actual_velocity[:6]]

    tcp_speed = _field(data, "TCPSpeedActual")
    if tcp_speed is not None and len(tcp_speed) >= 6:
        state.tcp_speed_actual_mm_deg_s = [float(v) for v in tcp_speed[:6]]

    tcp_force = _field(data, "ActualTCPForce")
    if tcp_force is not None and len(tcp_force) >= 6:
        state.tcp_force_actual = [float(v) for v in tcp_force[:6]]

    for attr, field_name in (
        ("robot_mode", "RobotMode"),
        ("safety_state", "SafetyState"),
        ("error_status", "ErrorStatus"),
        ("collision_state", "CollisionState"),
        ("speed_scaling", "SpeedScaling"),
    ):
        value = _field(data, field_name)
        if value is not None:
            try:
                setattr(state, attr, int(value) if attr != "speed_scaling" else float(value))
            except Exception:
                pass

    if state.robot_mode is not None:
        state.enabled = state.robot_mode in (5, 7, 8)


def _probe_tcp_port(ip: str, port: int, timeout_s: float = 1.0) -> None:
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return
    except OSError as exc:
        raise RobotAdapterError(
            f"Cannot connect to Dobot TCP {ip}:{port}. "
            "Check the robot IP, DobotStudio TCP/IP control mode, and firewall."
        ) from exc


def _unique_objects(*objects: Any) -> list[Any]:
    unique: list[Any] = []
    seen: set[int] = set()
    for obj in objects:
        if obj is not None and id(obj) not in seen:
            unique.append(obj)
            seen.add(id(obj))
    return unique


def _close_api_object(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()
    if hasattr(obj, "socket_dobot"):
        obj.socket_dobot = 0


def _require_success(response: Any, command: str) -> None:
    text = str(response)
    if "Not Tcp" in text:
        raise RobotAdapterError(f"{command} rejected: robot is not in TCP/IP control mode")
    match = re.search(r"-?\d+", text)
    if match is None:
        raise RobotAdapterError(f"{command} returned an invalid response: {text!r}")
    if int(match.group()) != 0:
        raise RobotAdapterError(f"{command} failed: {text}")
