from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from master_teleop.collection.launch_nodes import DATASET_TASK_NAME
from master_teleop.collection.start_master_teleop import main as start_master_teleop


def main() -> None:
    """Step 5: real CR5A + gripper collection entry point."""
    start_master_teleop(_build_start_argv(sys.argv[1:]))


def _build_start_argv(arguments: list[str]) -> list[str]:
    argv = list(arguments)

    dry_run = "--dry-run" in argv
    if dry_run:
        argv.remove("--dry-run")
    if dry_run and "--execute" in argv:
        raise SystemExit("--dry-run and --execute cannot be used together")
    if not dry_run and "--execute" not in argv:
        argv.append("--execute")
    if not dry_run and "--enable-robot" not in argv:
        argv.append("--enable-robot")
    if not any(
        value == "--speed-factor" or value.startswith("--speed-factor=")
        for value in argv
    ):
        argv.extend(["--speed-factor", "100"])
    if not any(value == "--task" or value.startswith("--task=") for value in argv):
        argv.extend(["--task", DATASET_TASK_NAME])

    with_camera = "--with-camera" in argv
    if with_camera:
        argv.remove("--with-camera")
    if with_camera and "--no-camera" in argv:
        raise SystemExit("--with-camera and --no-camera cannot be used together")
    if "--no-camera" not in argv and not any(
        value in ("--camera-preview", "--no-camera-preview") for value in argv
    ):
        argv.append("--camera-preview")

    no_record = "--no-record-data" in argv
    if no_record:
        argv.remove("--no-record-data")
    elif "--record-data" not in argv:
        argv.append("--record-data")
    return argv


if __name__ == "__main__":
    main()
