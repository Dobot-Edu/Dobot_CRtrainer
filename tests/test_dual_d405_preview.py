"""Manual preview test for two Intel RealSense D405 cameras.

This is a hardware test, not an automated unit test. It only accesses cameras
when executed directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = ROOT / "Data_Collections"


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    serial: str

    @property
    def is_d405(self) -> bool:
        return "D405" in self.name.upper()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open two independent RGB preview windows for two RealSense D405 cameras."
    )
    parser.add_argument(
        "--serial",
        nargs=2,
        metavar=("D405_1", "D405_2"),
        help="Use these two D405 serial numbers in the specified window order.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List connected RealSense devices without opening a stream.",
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    return args


def enumerate_devices(rs: Any) -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for device in rs.context().query_devices():
        devices.append(
            DeviceInfo(
                name=str(device.get_info(rs.camera_info.name)),
                serial=str(device.get_info(rs.camera_info.serial_number)),
            )
        )
    return devices


def print_devices(devices: Sequence[DeviceInfo]) -> None:
    if not devices:
        print("No Intel RealSense device detected.")
        return
    print("Connected Intel RealSense devices:")
    for device in devices:
        marker = "D405" if device.is_d405 else "other model"
        print(f"  - {device.name}: serial={device.serial} ({marker})")


def select_d405s(
    devices: Sequence[DeviceInfo], requested_serials: Sequence[str] | None
) -> list[DeviceInfo]:
    d405s = [device for device in devices if device.is_d405]
    by_serial = {device.serial: device for device in d405s}

    if requested_serials is not None:
        serials = [serial.strip() for serial in requested_serials]
        if not all(serials):
            raise ValueError("Both --serial values must be non-empty")
        if serials[0] == serials[1]:
            raise ValueError("The two --serial values must identify different cameras")
        missing = [serial for serial in serials if serial not in by_serial]
        if missing:
            raise RuntimeError(
                "Requested D405 serial number(s) not found: " + ", ".join(missing)
            )
        return [by_serial[serial] for serial in serials]

    if len(d405s) < 2:
        raise RuntimeError(f"Two D405 cameras are required, but only {len(d405s)} were found")
    if len(d405s) > 2:
        raise RuntimeError(
            "More than two D405 cameras were found; use --serial SERIAL_1 SERIAL_2 "
            "to select a stable pair"
        )
    return sorted(d405s, key=lambda device: device.serial)


def configure_linux_preview() -> None:
    if os.name == "nt":
        return

    sys.path.insert(0, str(COLLECTION_ROOT))
    from vr_teleop.core.config import load_project_config
    from vr_teleop.devices.camera import configure_preview_display

    project_config = load_project_config(ROOT / "config")
    configure_preview_display(project_config)


def run_preview(args: argparse.Namespace, selected: Sequence[DeviceInfo]) -> int:
    sys.path.insert(0, str(COLLECTION_ROOT))
    try:
        import cv2
        from vr_teleop.devices.camera import CameraConfig, CameraGroup, create_camera
    except ImportError as exc:
        print(f"Missing camera dependency: {exc}", file=sys.stderr)
        return 2

    cameras = []
    for index, device in enumerate(selected, start=1):
        config = CameraConfig(
            name=f"d405_{index}",
            enabled=True,
            required=True,
            driver="realsense",
            device="",
            serial=device.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            enable_depth=False,
            preview=True,
            startup_timeout_s=12.0,
            frame_timeout_ms=3000,
            max_consecutive_timeouts=3,
        )
        cameras.append(create_camera(config))

    group = CameraGroup(cameras)
    try:
        configure_linux_preview()
        group.start()
        print(
            "Two D405 previews are running. Press q or Esc in a preview window "
            "to quit; Ctrl+C also works."
        )
        while any(camera.config.preview for camera in cameras):
            group.raise_if_failed()
            group.show_previews()
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("Stopping D405 preview.")
    except Exception as exc:
        print(f"D405 preview test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        group.stop()
        cv2.destroyAllWindows()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import pyrealsense2 as rs
    except ImportError:
        print(
            "pyrealsense2 is not installed. Install the project requirements first.",
            file=sys.stderr,
        )
        return 2

    try:
        devices = enumerate_devices(rs)
        print_devices(devices)
        if args.list:
            return 0
        selected = select_d405s(devices, args.serial)
    except Exception as exc:
        print(f"D405 discovery failed: {exc}", file=sys.stderr)
        return 1

    print("Selected D405 cameras:")
    for index, device in enumerate(selected, start=1):
        print(f"  d405_{index}: {device.name}, serial={device.serial}")
    return run_preview(args, selected)


if __name__ == "__main__":
    raise SystemExit(main())
