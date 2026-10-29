from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from master_teleop.collection.workflow import (
    calibration_from_raw,
    create_master,
    create_robot,
    format_six,
    update_runtime_master,
    wait_master_sample,
    wait_robot_state,
)
from master_teleop.core.config import load_master_project_config
from master_teleop.core.sync_trace import (
    write_calibration_report,
    write_master_pose_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: inspect the master's current calibrated pose relative to the "
            "configured CR5A home. Add --calibrate only when intentionally replacing "
            "the master zero-offset calibration."
        )
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--robot-ip", default=None)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-deg",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Use an explicit six-joint calibration target.",
    )
    target_group.add_argument(
        "--use-current-robot-pose",
        action="store_true",
        help=(
            "Calibrate against the CR5A's current feedback instead of the configured "
            "standard home. Use only when the leader is physically matched to that pose."
        ),
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Replace the saved zero offsets by assigning the master's current physical "
            "pose to the target joints. Without this flag the script is read-only."
        ),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Require typing CALIBRATE; default is automatic calibration.",
    )
    parser.add_argument("--no-save", action="store_true", help="Calculate without saving.")
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    calibration_only_options = bool(
        args.target_deg is not None
        or args.use_current_robot_pose
        or args.confirm
        or args.no_save
    )
    if calibration_only_options and not args.calibrate:
        parser.error(
            "--target-deg, --use-current-robot-pose, --confirm and --no-save "
            "require --calibrate"
        )

    config = load_master_project_config(args.config_dir)
    robot = None
    master = create_master(config)
    try:
        if not args.calibrate:
            master.connect()
            sample = wait_master_sample(master)
            assert master.profile is not None
            reference_deg = _configured_home_joint_deg(config)
            calibrated_rows: list[list[float]] = []
            raw_rows: list[list[int]] = []
            for _ in range(args.samples):
                sample = wait_master_sample(master, timeout_s=1.0)
                if len(sample.joint_positions_rad) != 6 or len(sample.raw_positions) < 7:
                    raise RuntimeError("master sample is missing calibrated or raw joints")
                calibrated_rows.append(list(sample.joint_positions_rad))
                raw_rows.append(list(sample.raw_positions[:7]))
                time.sleep(0.02)
            current_rad = [
                statistics.fmean(row[index] for row in calibrated_rows)
                for index in range(6)
            ]
            current_deg = [value * 180.0 / 3.141592653589793 for value in current_rad]
            relative_deg = [
                _wrapped_degree_delta(current, reference)
                for current, reference in zip(current_deg, reference_deg)
            ]
            averaged_raw = [
                statistics.fmean(row[index] for row in raw_rows)
                for index in range(7)
            ]
            print(f"master profile: {master.profile.name}")
            print(f"reference CR5A joints: {format_six(reference_deg, 2)} deg")
            print(f"current calibrated master: {format_six(current_deg, 2)} deg")
            print(f"relative to reference: {format_six(relative_deg, 2)} deg")
            print("Read-only inspection complete; no calibration file was changed.")
            report_path = write_master_pose_report(
                ROOT / "outputs" / "master_sync_diagnostics",
                profile_name=master.profile.name,
                reference_joint_deg=reference_deg,
                current_joint_rad=current_rad,
                relative_joint_deg=relative_deg,
                averaged_raw_positions=averaged_raw,
                sample_count=len(calibrated_rows),
                loaded_joint_offsets_rad=master.profile.joint_offsets_rad,
                joint_signs=master.profile.joint_signs,
            )
            print(f"master_pose_diagnostic={report_path}")
            return

        if args.use_current_robot_pose:
            robot = create_robot(config, args.robot_ip)
            robot.connect()
            robot_state = wait_robot_state(robot)
            target_deg = list(robot_state.q_actual_deg)
            print(f"CR5A pose_mm_deg: {format_six(robot_state.pose_mm_deg, 2)}")
            print(f"current CR5A calibration target: {format_six(target_deg, 2)} deg")
        elif args.target_deg is not None:
            target_deg = [float(value) for value in args.target_deg]
            print(f"explicit calibration target: {format_six(target_deg, 2)} deg")
        else:
            target_deg = _configured_home_joint_deg(config)
            print(f"standard calibration target: {format_six(target_deg, 2)} deg")

        master.connect()
        sample = wait_master_sample(master)
        assert master.profile is not None
        print(f"master profile: {master.profile.name}")
        print(f"current calibrated master: {format_six(sample.joint_positions_rad)} rad")
        calibrated_before_rad = list(sample.joint_positions_rad)
        print("Keep the master stationary and put its gripper fully OPEN.")
        print("The current physical master pose will be assigned to the target joints above.")
        if args.confirm:
            answer = input("Type CALIBRATE to continue: ").strip()
            if answer != "CALIBRATE":
                raise SystemExit("Calibration cancelled; no file was changed")
        else:
            print("CALIBRATE (automatic; use --confirm to require manual confirmation)")

        rows: list[list[int]] = []
        for _ in range(args.samples):
            sample = wait_master_sample(master, timeout_s=1.0)
            if len(sample.raw_positions) < 7:
                raise RuntimeError("master sample does not contain seven raw positions")
            rows.append(list(sample.raw_positions[:7]))
            time.sleep(0.02)
        averaged = [statistics.fmean(row[index] for row in rows) for index in range(7)]
        calibration = calibration_from_raw(
            averaged,
            target_deg,
            master.profile.joint_signs,
            master.config.position_units_per_pi,
        )
        print(f"joint_offsets_rad: {calibration['joint_offsets_rad']}")
        print(
            "gripper endpoints deg: "
            f"closed={calibration['gripper_closed_deg']} open={calibration['gripper_open_deg']}"
        )
        saved_path = None
        if not args.no_save:
            saved_path = update_runtime_master(
                {
                    "side": master.profile.name,
                    "profiles": {master.profile.name: calibration},
                }
            )
            print(f"saved: {saved_path}")
            print(
                "Calibration complete. The leader's CURRENT physical pose now maps "
                f"exactly to {format_six(target_deg, 2)} deg."
            )
            print(
                "IMPORTANT: if the leader remains in this same pose, long-press A "
                "will command no follower motion because leader and follower targets "
                "are identical. Reposition the leader before starting run_control.py "
                "when testing initial alignment."
            )
        report_path = write_calibration_report(
            ROOT / "outputs" / "master_sync_diagnostics",
            profile_name=master.profile.name,
            target_deg=target_deg,
            sample_count=len(rows),
            first_calibrated_rad=calibrated_before_rad,
            averaged_raw_positions=averaged,
            joint_signs=master.profile.joint_signs,
            position_units_per_pi=master.config.position_units_per_pi,
            calibration=calibration,
            saved_config_path=saved_path,
        )
        print(f"calibration_diagnostic={report_path}")
    finally:
        master.close()
        if robot is not None:
            robot.close()


def _configured_home_joint_deg(config: dict) -> list[float]:
    values = (
        config.get("master_hand", {})
        .get("robot_pose_init", {})
        .get("home_joint_deg", [90, 0, 90, 0, -90, -90])
    )
    result = [float(value) for value in values]
    if len(result) != 6:
        raise ValueError("master_hand.robot_pose_init.home_joint_deg must contain six values")
    return result


def _wrapped_degree_delta(target: float, reference: float) -> float:
    return (float(target) - float(reference) + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    main()
