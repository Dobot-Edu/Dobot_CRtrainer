from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from time import monotonic
import math


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from master_teleop.control.button_controller import MasterButtonController, MasterButtonEvent
from master_teleop.collection.workflow import calibration_from_raw
from master_teleop.collection.run_control import _build_start_argv
from master_teleop.collection.launch_nodes import (
    DATASET_TASK_NAME,
    _ensure_robot_ready,
    _joint_path,
    _run_gripper_open_close_cycle,
)
from master_teleop.control.joint_mapper import RelativeJointMapper
from master_teleop.control.models import MasterHandInput
from master_teleop.core.config import load_master_project_config
from master_teleop.core.app import MasterHandTeleopApp
from master_teleop.devices.master_hand import MockMasterHand
from vr_teleop.control.models import RobotState, TeleopCommand
from vr_teleop.core.app import TeleopRuntimeConfig
from vr_teleop.control.safety_supervisor import SafetySupervisor
from vr_teleop.core.config import load_project_config
from vr_teleop.devices.dobot_adapter import DobotTcpAdapter, MockRobotAdapter
from vr_teleop.devices.gripper_controller import GripperConfig
from vr_teleop.recording.dataset_recorder import EpisodeRecorder
from scripts.recording_status_window import (
    NOT_RECORDING_BGR,
    RECORDING_BGR,
    make_status_frame,
)


class SingleMasterTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_master_project_config(ROOT / "config")

    def sample(self, joints=None, pressed=True) -> MasterHandInput:
        return MasterHandInput(
            received_monotonic=monotonic(),
            timestamp=0.0,
            joint_positions_rad=list(joints or [0.0] * 6),
            gripper_open=1.0,
            button_a=pressed,
            button_b=False,
            buttons_raw=[0 if pressed else 1, 1],
            profile_name="right",
        )

    def robot_state(self) -> RobotState:
        return RobotState(
            connected=True,
            enabled=True,
            pose_mm_deg=[908.0, -158.0, 469.0, 179.0, 0.0, -90.0],
            target_pose_mm_deg=[908.0, -158.0, 469.0, 179.0, 0.0, -90.0],
            q_actual_deg=[90.0, 0.0, 90.0, 0.0, -90.0, -90.0],
            q_target_deg=[90.0, 0.0, 90.0, 0.0, -90.0, -90.0],
            robot_mode=5,
            last_feedback_monotonic=monotonic(),
        )

    def test_servo_disabled_holds_and_start_aligns_without_jump_when_already_aligned(self) -> None:
        mapper = RelativeJointMapper(self.config, SafetySupervisor(self.config, True))
        robot = self.robot_state()
        aligned = [math.radians(value) for value in robot.q_actual_deg]
        self.assertFalse(mapper.map(self.sample(aligned), robot, 0.02, active=False).allowed)
        command = mapper.map(self.sample(aligned), robot, 0.02, active=True)
        self.assertTrue(command.allowed)
        self.assertEqual(command.target_joint_positions_deg, robot.q_actual_deg)
        self.assertTrue(mapper.aligned)

    def test_master_config_is_not_loaded_by_vr_mode(self) -> None:
        vr_config = load_project_config(ROOT / "config")
        self.assertNotIn("master_hand", vr_config)
        self.assertIn("master_hand", self.config)
        self.assertEqual(vr_config["safety"]["workspace"]["x_max_mm"], 920.0)
        self.assertEqual(self.config["safety"]["workspace"]["x_min_mm"], -250.0)
        self.assertEqual(self.config["safety"]["workspace"]["x_max_mm"], 1150.0)
        self.assertEqual(self.config["safety"]["workspace"]["y_min_mm"], -750.0)
        self.assertEqual(self.config["safety"]["workspace"]["y_max_mm"], 600.0)
        self.assertEqual(self.config["safety"]["workspace"]["z_min_mm"], 250.0)
        self.assertEqual(self.config["safety"]["workspace"]["z_max_mm"], 750.0)
        self.assertEqual(self.config["safety"]["watchdog_timeout_s"], 0.35)
        self.assertFalse(self.config["master_hand"]["mapping"]["predict_target_workspace"])

    def test_recording_status_frames_match_camera_size_and_state_colors(self) -> None:
        idle = make_status_frame(False, 640, 480)
        recording = make_status_frame(True, 640, 480)

        self.assertEqual(idle.shape, (480, 640, 3))
        self.assertEqual(recording.shape, (480, 640, 3))
        self.assertTrue((idle == NOT_RECORDING_BGR).all())
        self.assertTrue((recording == RECORDING_BGR).all())

    def test_predictive_workspace_guard_accepts_in_bounds_target(self) -> None:
        command = TeleopCommand(
            allowed=True,
            target_joint_positions_deg=[90, 0, 90, 0, -90, -90],
            reason="ok",
        )
        app = object.__new__(MasterHandTeleopApp)
        app.predict_target_workspace = True
        app.robot = SimpleNamespace(
            positive_kinematics=lambda joints: [908, -158, 469, 180, 0, -90]
        )
        app.mapper = SimpleNamespace(
            safety=SimpleNamespace(pose_in_workspace=lambda pose: True)
        )

        app._apply_predictive_workspace_guard(command)

        self.assertTrue(command.allowed)
        self.assertEqual(command.target_pose_mm_deg, [908, -158, 469, 180, 0, -90])
        self.assertEqual(app._last_predicted_pose_mm_deg, command.target_pose_mm_deg)

    def test_predictive_workspace_guard_rejects_out_of_bounds_target(self) -> None:
        command = TeleopCommand(
            allowed=True,
            target_joint_positions_deg=[90, 0, 90, 0, -90, -90],
            reason="ok",
        )
        app = object.__new__(MasterHandTeleopApp)
        app.predict_target_workspace = True
        app.robot = SimpleNamespace(
            positive_kinematics=lambda joints: [1180, -158, 469, 180, 0, -90]
        )
        app.mapper = SimpleNamespace(
            safety=SimpleNamespace(pose_in_workspace=lambda pose: False)
        )

        app._apply_predictive_workspace_guard(command)

        self.assertFalse(command.allowed)
        self.assertEqual(
            command.reason,
            "predicted_target_pose_outside_workspace:1180.0,-158.0,469.0",
        )

    def test_predictive_workspace_guard_fails_closed_on_fk_error(self) -> None:
        def fail(_joints):
            raise RuntimeError("controller timeout")

        command = TeleopCommand(
            allowed=True,
            target_joint_positions_deg=[90, 0, 90, 0, -90, -90],
            reason="ok",
        )
        app = object.__new__(MasterHandTeleopApp)
        app.predict_target_workspace = True
        app.robot = SimpleNamespace(positive_kinematics=fail)

        app._apply_predictive_workspace_guard(command)

        self.assertFalse(command.allowed)
        self.assertEqual(command.reason, "positive_kinematics_failed:controller timeout")
        self.assertIsNone(app._last_predicted_pose_mm_deg)

    def test_predictive_workspace_guard_can_restore_baseline_behavior(self) -> None:
        command = TeleopCommand(
            allowed=True,
            target_joint_positions_deg=[90, 0, 90, 0, -90, -90],
            reason="ok",
        )
        app = object.__new__(MasterHandTeleopApp)
        app.predict_target_workspace = False
        app.robot = SimpleNamespace(
            positive_kinematics=lambda joints: self.fail("FK must remain disabled")
        )

        app._apply_predictive_workspace_guard(command)

        self.assertTrue(command.allowed)
        self.assertEqual(command.reason, "ok")
        self.assertIsNone(command.target_pose_mm_deg)

    def test_dobot_positive_kinematics_uses_controller_default_frames(self) -> None:
        calls = []

        class Dashboard:
            def PositiveKin(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "0,{908,-158,469,180,0,-90},PositiveKin();"

        robot = DobotTcpAdapter("192.0.2.1", 29999, 30003, 30004)
        robot.dashboard = Dashboard()
        pose = robot.positive_kinematics([90, 0, 90, 0, -90, -90])

        self.assertEqual(pose, [908.0, -158.0, 469.0, 180.0, 0.0, -90.0])
        self.assertEqual(calls[0][0], (90, 0, 90, 0, -90, -90))
        self.assertEqual(calls[0][1], {})

    def test_joint_motion_is_rate_limited(self) -> None:
        mapper = RelativeJointMapper(self.config, SafetySupervisor(self.config, True))
        robot = self.robot_state()
        aligned = [math.radians(value) for value in robot.q_actual_deg]
        mapper.map(self.sample(aligned), robot, 0.02, active=True)
        aligned[0] += 0.2
        moved = self.sample(joints=aligned)
        command = mapper.map(moved, robot, 0.02, active=True)
        self.assertAlmostEqual(command.target_joint_positions_deg[0], 93.5, places=6)

    def test_xtrainer_dynamic_approach_uses_reference_rate_limit(self) -> None:
        mapper = RelativeJointMapper(self.config, SafetySupervisor(self.config, True))
        robot = self.robot_state()
        aligned = [math.radians(value) for value in robot.q_actual_deg]
        mapper.map(self.sample(aligned), robot, 0.02, active=True)
        mapper.begin_soft_alignment()
        aligned[0] += 0.2
        command = mapper.map(self.sample(aligned), robot, 0.02, active=True)
        self.assertAlmostEqual(command.target_joint_positions_deg[0], 90.23, places=6)
        self.assertTrue(mapper.soft_alignment_active)
        mapper.end_soft_alignment()
        self.assertFalse(mapper.soft_alignment_active)

    def test_dynamic_approach_freezes_long_press_target_until_aligned(self) -> None:
        mapper = RelativeJointMapper(self.config, SafetySupervisor(self.config, True))
        robot = self.robot_state()
        first_target = [math.radians(value) for value in [95, 0, 90, 0, -90, -90]]
        mapper.begin_soft_alignment(self.sample(first_target), robot)

        first = mapper.map(self.sample(first_target), robot, 0.02, active=True)
        moved_leader = [math.radians(value) for value in [80, 0, 90, 0, -90, -90]]
        second = mapper.map(self.sample(moved_leader), robot, 0.02, active=True)

        self.assertEqual(mapper.soft_alignment_target_deg[0], 95.0)
        self.assertEqual(mapper.desired_target_deg[0], 95.0)
        self.assertGreater(second.target_joint_positions_deg[0], first.target_joint_positions_deg[0])

    def test_dynamic_approach_rejects_large_target_before_motion(self) -> None:
        config = copy.deepcopy(self.config)
        config["master_hand"]["mapping"]["enforce_initial_alignment_delta"] = True
        mapper = RelativeJointMapper(config, SafetySupervisor(config, True))
        robot = self.robot_state()
        target = [math.radians(value) for value in [90, 50, 90, 0, -90, -90]]

        with self.assertRaisesRegex(RuntimeError, "alignment_mismatch:J2"):
            mapper.begin_soft_alignment(self.sample(target), robot)

    def test_large_initial_master_follower_mismatch_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["master_hand"]["mapping"]["enforce_initial_alignment_delta"] = True
        mapper = RelativeJointMapper(config, SafetySupervisor(config, True))
        command = mapper.map(self.sample([0.0] * 6), self.robot_state(), 0.02, active=True)
        self.assertFalse(command.allowed)
        self.assertIn("alignment_mismatch", command.reason)

    def test_workspace_gated_alignment_allows_large_joint_delta(self) -> None:
        mapper = RelativeJointMapper(self.config, SafetySupervisor(self.config, True))
        robot = self.robot_state()
        target = [math.radians(value) for value in [90, 50, 90, 0, -90, -90]]

        mapper.begin_soft_alignment(self.sample(target), robot)

        self.assertEqual(mapper.soft_alignment_target_deg[1], 50.0)

    def test_stale_robot_target_pose_does_not_block_joint_alignment(self) -> None:
        mapper = RelativeJointMapper(self.config, SafetySupervisor(self.config, True))
        robot = self.robot_state()
        robot.target_pose_mm_deg = [2000.0, 2000.0, 2000.0, 0.0, 0.0, 0.0]
        aligned = [math.radians(value) for value in robot.q_actual_deg]

        command = mapper.map(self.sample(aligned), robot, 0.02, active=True)

        self.assertTrue(command.allowed)
        self.assertEqual(command.target_joint_positions_deg, robot.q_actual_deg)

    def test_xtrainer_button_press_semantics(self) -> None:
        buttons = MasterButtonController()
        released = self.sample(pressed=False)
        pressed = self.sample(pressed=True)

        buttons.update(pressed, 10.0)
        state = buttons.update(released, 10.2)
        self.assertTrue(state.master_locked)
        self.assertEqual(state.events[0].action, "master_lock")

        buttons.update(pressed, 20.0)
        state = buttons.update(released, 21.2)
        self.assertTrue(state.servo_enabled)
        self.assertEqual(state.events[0].action, "servo")

        b_pressed = self.sample(pressed=False)
        b_pressed.button_b = True
        buttons.update(b_pressed, 30.0)
        state = buttons.update(released, 30.2)
        self.assertEqual(state.events[0].action, "record")

    def test_xtrainer_locked_start_first_short_press_unlocks(self) -> None:
        buttons = MasterButtonController(initial_locked=True)
        pressed = self.sample(pressed=True)
        released = self.sample(pressed=False)
        buttons.update(pressed, 1.0)
        state = buttons.update(released, 1.2)
        self.assertFalse(state.master_locked)
        self.assertEqual(state.events[0].action, "master_lock")
        self.assertFalse(state.events[0].value)

    def test_xtrainer_short_a_toggles_leader_before_servo(self) -> None:
        class MapperStub:
            def __init__(self) -> None:
                self.soft_alignment_active = False

            def begin_soft_alignment(self, *args) -> None:
                self.soft_alignment_active = True

            def end_soft_alignment(self) -> None:
                self.soft_alignment_active = False

        master = MockMasterHand()
        master.connect()
        master.set_locked(True)
        buttons = MasterButtonController(initial_locked=True)
        mapper = MapperStub()
        buttons.state.master_locked = False
        holder = SimpleNamespace(
            master=master,
            buttons=buttons,
            mapper=mapper,
            _alignment_ready=False,
            _dynamic_approach_active=False,
            _unlock_after_approach=False,
        )
        buttons.state.master_locked = False
        MasterHandTeleopApp._handle_button_event(
            holder, MasterButtonEvent("master_lock", False, 0.2)
        )
        self.assertFalse(master.locked)

        buttons.state.master_locked = True
        MasterHandTeleopApp._handle_button_event(
            holder, MasterButtonEvent("master_lock", True, 0.2)
        )
        self.assertTrue(master.locked)
        self.assertTrue(buttons.state.master_locked)

    def test_xtrainer_long_a_approaches_locked_then_keeps_logical_lock(self) -> None:
        class MapperStub:
            def __init__(self) -> None:
                self.soft_alignment_active = False

            def begin_soft_alignment(self, *args) -> None:
                self.soft_alignment_active = True

            def end_soft_alignment(self) -> None:
                self.soft_alignment_active = False

        master = MockMasterHand()
        master.connect()
        master.set_locked(True)
        buttons = MasterButtonController(initial_locked=True)
        mapper = MapperStub()
        holder = SimpleNamespace(
            master=master,
            buttons=buttons,
            mapper=mapper,
            robot=SimpleNamespace(stop=lambda: None, state=self.robot_state),
            runtime=SimpleNamespace(execute=False),
            _alignment_ready=False,
            _dynamic_approach_active=False,
            _unlock_after_approach=False,
            _was_commanding=False,
            _require_alignment_ready=lambda sample, state: None,
            _require_frozen_target_workspace=lambda: None,
            _align_gripper_to_master=lambda sample: None,
            _print_alignment_target=lambda sample: None,
        )
        MasterHandTeleopApp._handle_button_event(
            holder, MasterButtonEvent("servo", True, 1.2)
        )
        self.assertTrue(master.locked)
        self.assertTrue(holder._dynamic_approach_active)
        self.assertTrue(mapper.soft_alignment_active)
        self.assertTrue(MasterHandTeleopApp._complete_dynamic_approach(holder))
        self.assertTrue(master.locked)
        self.assertTrue(buttons.state.master_locked)
        self.assertTrue(holder._alignment_ready)
        self.assertFalse(holder._dynamic_approach_active)
        self.assertFalse(mapper.soft_alignment_active)

        buttons.state.master_locked = False
        MasterHandTeleopApp._handle_button_event(
            holder, MasterButtonEvent("master_lock", False, 0.2)
        )
        self.assertFalse(master.locked)

    def test_long_a_dynamic_approach_sends_joint_command(self) -> None:
        master = MockMasterHand()
        master.connect()
        master.set_locked(True)
        target_deg = [95.0, 0.0, 90.0, 0.0, -90.0, -90.0]
        master.set_mock_state([math.radians(value) for value in target_deg])
        robot = MockRobotAdapter(
            [908.0, -158.0, 469.0, 180.0, 0.0, -90.0],
            initial_joints_deg=[90.0, 0.0, 90.0, 0.0, -90.0, -90.0],
        )
        robot.connect()
        buttons = MasterButtonController(initial_locked=True)
        mapper = RelativeJointMapper(self.config, SafetySupervisor(self.config, True))
        app = MasterHandTeleopApp(
            master=master,
            robot=robot,
            mapper=mapper,
            runtime=TeleopRuntimeConfig(
                servo_period_s=0.02,
                servo_t_s=0.02,
                servo_aheadtime=20,
                servo_gain=680,
                execute=True,
            ),
            buttons=buttons,
            predict_target_workspace=False,
        )
        buttons.state.servo_enabled = True

        app._handle_button_event(
            MasterButtonEvent("servo", True, 1.2), master.latest()
        )
        app.tick(0.02)

        self.assertTrue(app._dynamic_approach_active)
        self.assertEqual(robot.commands_sent, 1)
        self.assertGreater(robot.state().q_target_deg[0], 90.0)

    def test_xtrainer_preunlocked_leader_reunlocks_after_dynamic_approach(self) -> None:
        class MapperStub:
            def __init__(self) -> None:
                self.soft_alignment_active = False

            def begin_soft_alignment(self, *args) -> None:
                self.soft_alignment_active = True

            def end_soft_alignment(self) -> None:
                self.soft_alignment_active = False

        master = MockMasterHand()
        master.connect()
        master.set_locked(False)
        buttons = MasterButtonController(initial_locked=False)
        mapper = MapperStub()
        holder = SimpleNamespace(
            master=master,
            buttons=buttons,
            mapper=mapper,
            robot=SimpleNamespace(stop=lambda: None, state=self.robot_state),
            runtime=SimpleNamespace(execute=False),
            _alignment_ready=False,
            _dynamic_approach_active=False,
            _unlock_after_approach=False,
            _was_commanding=False,
            _require_alignment_ready=lambda sample, state: None,
            _require_frozen_target_workspace=lambda: None,
            _align_gripper_to_master=lambda sample: None,
            _print_alignment_target=lambda sample: None,
        )
        MasterHandTeleopApp._handle_button_event(
            holder, MasterButtonEvent("servo", True, 1.2)
        )
        self.assertTrue(master.locked)
        self.assertTrue(holder._unlock_after_approach)
        self.assertTrue(MasterHandTeleopApp._complete_dynamic_approach(holder))
        self.assertFalse(master.locked)
        self.assertFalse(buttons.state.master_locked)

    def test_xtrainer_long_a_stop_restores_logical_leader_lock(self) -> None:
        class MapperStub:
            def __init__(self) -> None:
                self.soft_alignment_active = True

            def begin_soft_alignment(self) -> None:
                self.soft_alignment_active = True

            def end_soft_alignment(self) -> None:
                self.soft_alignment_active = False

        master = MockMasterHand()
        master.connect()
        master.set_locked(True)
        buttons = MasterButtonController(initial_locked=False)
        holder = SimpleNamespace(
            master=master,
            buttons=buttons,
            mapper=MapperStub(),
            robot=SimpleNamespace(stop=lambda: None),
            runtime=SimpleNamespace(execute=False),
            _alignment_ready=False,
            _dynamic_approach_active=True,
            _unlock_after_approach=True,
            _was_commanding=False,
        )
        MasterHandTeleopApp._handle_button_event(
            holder, MasterButtonEvent("servo", False, 1.2)
        )
        self.assertFalse(master.locked)
        self.assertFalse(holder._dynamic_approach_active)
        self.assertFalse(holder.mapper.soft_alignment_active)

    def test_aligned_locked_state_reports_waiting_for_short_a(self) -> None:
        app = object.__new__(MasterHandTeleopApp)
        app._alignment_ready = True
        app._dynamic_approach_active = False
        buttons = MasterButtonController(initial_locked=True)
        buttons.state.servo_enabled = True
        app.buttons = buttons
        self.assertTrue(app.buttons.state.master_locked)
        self.assertTrue(app.buttons.state.servo_enabled)

    def test_run_control_bare_command_defaults_to_real_robot_gripper(self) -> None:
        argv = _build_start_argv([])
        self.assertIn("--execute", argv)
        self.assertIn("--enable-robot", argv)
        self.assertNotIn("--no-camera", argv)
        self.assertIn("--camera-preview", argv)
        self.assertIn("--record-data", argv)
        self.assertNotIn("--no-gripper", argv)
        speed_index = argv.index("--speed-factor")
        self.assertEqual(argv[speed_index + 1], "100")
        task_index = argv.index("--task")
        self.assertEqual(argv[task_index + 1], DATASET_TASK_NAME)

        dry = _build_start_argv(["--dry-run"])
        self.assertNotIn("--execute", dry)
        self.assertNotIn("--enable-robot", dry)

        no_camera = _build_start_argv(["--no-camera"])
        self.assertIn("--no-camera", no_camera)
        self.assertNotIn("--camera-preview", no_camera)

        with_camera = _build_start_argv(["--with-camera"])
        self.assertNotIn("--with-camera", with_camera)
        self.assertNotIn("--no-camera", with_camera)
        self.assertIn("--camera-preview", with_camera)

        no_preview = _build_start_argv(["--no-camera-preview"])
        self.assertIn("--no-camera-preview", no_preview)
        self.assertNotIn("--camera-preview", no_preview)

    def test_launch_gripper_cycle_opens_then_closes_at_full_speed(self) -> None:
        config = GripperConfig.from_mapping(self.config["gripper"])
        self.assertEqual(config.speed, 100)

        class FakeGripper:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    test_cycle_hold_s=0.0,
                    test_cycle_travel_s=0.001,
                    test_cycle_command_timeout_s=0.1,
                    test_cycle_position_tolerance=30,
                    feedback_poll_interval_s=0.001,
                    open_position=1000,
                    close_position=0,
                    pressed_closes=True,
                )
                self.commands: list[float] = []
                self.applied_closed = None
                self.actual_position = None

            def update(
                self,
                trigger: float,
                input_valid: bool = True,
                allow_initial_open: bool = False,
            ) -> None:
                self.commands.append(trigger)
                self.applied_closed = trigger >= 0.5
                self.actual_position = 0 if self.applied_closed else 1000

            def snapshot(self) -> dict[str, object]:
                return {
                    "applied_closed": self.applied_closed,
                    "actual_position": self.actual_position,
                    "error": None,
                }

        gripper = FakeGripper()
        _run_gripper_open_close_cycle(gripper)
        self.assertEqual(gripper.commands, [0.0, 1.0])

    def test_dynamic_approach_aligns_gripper_to_locked_master_pose(self) -> None:
        class FakeGripper:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    pressed_closes=True,
                )
                self.commands: list[float] = []

            def update(
                self,
                trigger: float,
                input_valid: bool = True,
                allow_initial_open: bool = False,
            ) -> None:
                self.commands.append(trigger)

        app = object.__new__(MasterHandTeleopApp)
        app.gripper = FakeGripper()
        closed_sample = self.sample()
        closed_sample.gripper_open = 0.1
        app._align_gripper_to_master(closed_sample)
        open_sample = self.sample()
        open_sample.gripper_open = 0.9
        app._align_gripper_to_master(open_sample)
        self.assertEqual(app.gripper.commands, [1.0, 0.0])

    def test_gripper_connection_is_deferred_until_long_a_alignment(self) -> None:
        class FakeGripper:
            def __init__(self) -> None:
                self.config = SimpleNamespace(pressed_closes=True)
                self.connect_calls = 0
                self.commands: list[float] = []

            def connect(self, dashboard) -> None:
                self.connect_calls += 1

            def close(self) -> None:
                pass

            def update(
                self,
                trigger: float,
                input_valid: bool = True,
                allow_initial_open: bool = False,
            ) -> None:
                self.commands.append(trigger)

        app = object.__new__(MasterHandTeleopApp)
        app.gripper = FakeGripper()
        app.robot = SimpleNamespace(dashboard=object())
        app._gripper_connected = False
        sample = self.sample()
        sample.gripper_open = 0.9

        self.assertEqual(app.gripper.connect_calls, 0)
        app._align_gripper_to_master(sample)

        self.assertEqual(app.gripper.connect_calls, 1)
        self.assertTrue(app._gripper_connected)
        self.assertEqual(app.gripper.commands, [0.0])

    def test_gripper_keeps_following_during_six_joint_approach(self) -> None:
        class FakeGripper:
            def __init__(self) -> None:
                self.commands: list[tuple[float, bool]] = []

            def update(self, trigger: float, input_valid: bool = True) -> None:
                self.commands.append((trigger, input_valid))

        app = object.__new__(MasterHandTeleopApp)
        app.gripper = FakeGripper()
        app._dynamic_approach_active = True
        app._alignment_ready = False
        app.buttons = SimpleNamespace(state=SimpleNamespace(master_locked=True))
        sample = self.sample()
        sample.gripper_open = 0.25

        app._update_gripper(sample, servo_enabled=True)

        self.assertEqual(app.gripper.commands, [(0.75, True)])

    def test_xtrainer_two_stage_pose_path_has_target_endpoint(self) -> None:
        path = _joint_path(
            [90, 0, 90, 0, -90, -90],
            [90, -30, 110, -20, -90, -90],
            interpolation_step_rad=0.01,
            max_steps=100,
        )
        self.assertGreater(len(path), 2)
        self.assertEqual(path[-1], [90.0, -30.0, 110.0, -20.0, -90.0, -90.0])

    def test_launch_nodes_clears_alarm_then_enables_robot(self) -> None:
        class RecoveringRobot:
            def __init__(self) -> None:
                self.current = SimpleNamespace(
                    error_status=1,
                    collision_state=0,
                    robot_mode=9,
                )
                self.clear_calls = 0
                self.enable_calls = 0

            def state(self):
                return self.current

            def clear_error(self) -> None:
                self.clear_calls += 1
                self.current.error_status = 0
                self.current.robot_mode = 4

            def enable_robot(self) -> None:
                self.enable_calls += 1
                self.current.robot_mode = 5

        robot = RecoveringRobot()
        _ensure_robot_ready(robot, robot.state(), timeout_s=0.1)
        self.assertEqual(robot.clear_calls, 1)
        self.assertEqual(robot.enable_calls, 1)
        self.assertEqual(robot.state().robot_mode, 5)

    def test_teleop_exit_locks_master_and_preserves_torque_state(self) -> None:
        master = MockMasterHand()
        master.connect()
        master.set_locked(False)
        holder = SimpleNamespace(master=master)
        MasterHandTeleopApp._lock_master_for_exit(holder)
        self.assertTrue(master.locked)
        self.assertIsNone(master.latest())

    def test_offset_calibration_matches_xtrainer_formula(self) -> None:
        raw = [1024.0, 0.0, 1024.0, 0.0, -1024.0, -1024.0, 1968.0]
        result = calibration_from_raw(
            raw_positions=raw,
            target_joint_deg=[90.0, 0.0, 90.0, 0.0, -90.0, -90.0],
            joint_signs=[1, 1, 1, 1, 1, 1],
            position_units_per_pi=2048.0,
        )
        self.assertEqual(result["joint_offsets_rad"], [0.0] * 6)
        self.assertAlmostEqual(result["gripper_closed_deg"], 202.969, places=3)
        self.assertAlmostEqual(result["gripper_open_deg"], 172.769, places=3)

    def test_mock_robot_accepts_single_six_joint_command(self) -> None:
        robot = MockRobotAdapter([135, -474, 467, 179, 0, -90])
        robot.connect()
        robot.servo_j([91, 1, 89, 2, -88, -89], 0.02, 35, 450)
        self.assertEqual(robot.state().q_actual_deg, [91, 1, 89, 2, -88, -89])

    def test_master_record_contains_joint_and_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                output_dir=directory,
                project_name="test",
                task="single_master",
                control_hz=50,
                min_episode_frames=1,
                save_npz=True,
                config_snapshot=self.config,
            )
            start = recorder.start_episode()
            command = TeleopCommand(
                allowed=True,
                target_pose_mm_deg=[135, -474, 467, 179, 0, -90],
                target_joint_positions_deg=[90, 0, 90, 0, -90, -90],
                reason="ok",
                clutched=True,
                control_source="master_hand",
                control_mode="servo_j",
            )
            self.assertTrue(recorder.record(self.sample(), self.robot_state(), command, 0.02, False))
            event = recorder.stop_episode()
            recorder.close()
            frame = json.loads((start.episode_dir / "frames.jsonl").read_text().strip())
            self.assertEqual(frame["action"]["control_source"], "master_hand")
            self.assertEqual(frame["action"]["control_mode"], "servo_j")
            self.assertEqual(len(frame["observation"]["master_joint_positions_rad"]), 6)
            self.assertTrue(event.valid)


if __name__ == "__main__":
    unittest.main()
