from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from master_teleop.collection.workflow import create_master, format_six, wait_master_sample
from master_teleop.control.button_controller import MasterButtonController
from master_teleop.core.config import load_master_project_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the master hand's calibrated six-joint pose and use A to lock/unlock it. "
            "This script never connects to or moves the CR5A."
        )
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means run until Ctrl+C.")
    parser.add_argument(
        "--home-tolerance-deg",
        type=float,
        default=2.0,
        help="Maximum per-joint error used to report that the configured home pose is reached.",
    )
    parser.add_argument(
        "--log-every",
        type=float,
        default=0.25,
        help="Seconds between live joint-angle reports.",
    )
    args = parser.parse_args()
    if args.seconds < 0.0:
        parser.error("--seconds must be non-negative")
    if args.home_tolerance_deg <= 0.0 or args.log_every <= 0.0:
        parser.error("--home-tolerance-deg and --log-every must be positive")

    config = load_master_project_config(args.config_dir)
    values = config["master_hand"]
    home_joint_deg = [
        float(value) for value in values["robot_pose_init"]["home_joint_deg"]
    ]
    if len(home_joint_deg) != 6:
        raise ValueError("master_hand.robot_pose_init.home_joint_deg must contain six values")
    thresholds = values.get("buttons", {})
    buttons = MasterButtonController(
        short_press_max_s=float(thresholds.get("short_press_max_s", 0.5)),
        long_press_min_s=float(thresholds.get("long_press_min_s", 1.0)),
        initial_locked=bool(thresholds.get("initial_locked", True)),
    )
    master = create_master(config)
    master.connect()
    try:
        sample = wait_master_sample(master)
        master.set_locked(True)
        print(f"master ready: profile={sample.profile_name}")
        print(f"configured home target: {format_six(home_joint_deg, 1)} deg")
        print("Leader is LOCKED. Short-press A once to unlock it, then move each joint by hand.")
        print("Watch current_deg and error_deg; lock it again when status=HOME.")
        print("This script never connects to or moves the CR5A.")
        deadline = time.monotonic() + args.seconds if args.seconds else None
        last_log = 0.0
        while deadline is None or time.monotonic() < deadline:
            try:
                now = time.monotonic()
                sample = master.latest()
                state = buttons.update(sample, now)
                for event in state.events:
                    if event.action == "master_lock":
                        master.set_locked(bool(event.value))
                        label = "LOCKED" if event.value else "UNLOCKED"
                        print(f"[A short {event.duration_s:.2f}s] leader {label}")
                    elif event.action == "servo":
                        label = "would ALIGN/START" if event.value else "would STOP"
                        print(f"[A long {event.duration_s:.2f}s] follower {label}")
                    elif event.action == "record":
                        print(f"[B short {event.duration_s:.2f}s] record toggle detected")
                    else:
                        print(f"[{event.action} {event.duration_s:.2f}s] ignored")
                if sample is not None and now - last_log >= args.log_every:
                    current_deg = [math.degrees(value) for value in sample.joint_positions_rad]
                    error_deg = [
                        (current - target + 180.0) % 360.0 - 180.0
                        for current, target in zip(current_deg, home_joint_deg, strict=True)
                    ]
                    max_error_deg = max(abs(value) for value in error_deg)
                    status = (
                        "HOME"
                        if max_error_deg <= args.home_tolerance_deg
                        else "ADJUST"
                    )
                    print(
                        f"current_deg={format_six(current_deg, 1)} "
                        f"error_deg={format_six(error_deg, 1)} "
                        f"max_error={max_error_deg:.1f}deg status={status} "
                        f"locked={state.master_locked} A={sample.button_a} B={sample.button_b}"
                    )
                    last_log = now
                time.sleep(0.01)
            except KeyboardInterrupt:
                if master.locked:
                    print("Stopping with leader LOCKED; torque will remain enabled for offset calibration.")
                    break
                print("Leader is UNLOCKED. Short-press A to lock it, then press Ctrl+C again.")
        if not master.locked:
            print("Time limit ended while unlocked; locking the current pose for calibration.")
            master.set_locked(True)
    finally:
        # Intentional: 2_get_offset.py must read exactly this rigid pose.
        master.close(release_torque=False)


if __name__ == "__main__":
    main()
