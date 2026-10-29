from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class CameraError(RuntimeError):
    pass


def configure_preview_display(config: dict[str, Any]) -> str:
    """Route preview windows to the robot host desktop, including over SSH."""

    values = config.get("host_network", {})
    target = str(values.get("preview_target", "host")).strip().lower()
    if target not in ("host", "session"):
        raise CameraError("host_network.preview_target must be 'host' or 'session'")

    if target == "host":
        display = str(values.get("host_desktop_display", ":1")).strip()
        if not display.startswith(":"):
            raise CameraError("host_network.host_desktop_display must look like ':1'")
        display_number = display.removeprefix(":").split(".", 1)[0]
        socket_path = Path("/tmp/.X11-unix") / f"X{display_number}"
        if not socket_path.exists():
            raise CameraError(
                f"Host preview display is unavailable: X11 socket not found: {socket_path}"
            )
        os.environ["DISPLAY"] = display
        configured_xauthority = str(values.get("host_xauthority", "")).strip()
        xauthority = (
            Path(configured_xauthority).expanduser()
            if configured_xauthority
            else Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
        )
        if not xauthority.is_file():
            raise CameraError(f"Host preview Xauthority not found: {xauthority}")
        os.environ["XAUTHORITY"] = str(xauthority)
    else:
        display = str(os.environ.get("DISPLAY", "")).strip()
        if not display:
            raise CameraError(
                "Session preview requested but DISPLAY is empty; use preview_target: host "
                "or enable X11 forwarding"
            )

    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    if font_dir.is_dir():
        # OpenCV's Qt wheel may pre-populate this with a bundled directory that
        # does not exist. Always prefer the host's installed DejaVu fonts.
        os.environ["QT_QPA_FONTDIR"] = str(font_dir)

    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo is None:
        raise CameraError("xdpyinfo is required to validate the preview display")
    try:
        check = subprocess.run(
            [xdpyinfo],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CameraError(f"Failed to validate preview display {display}: {exc}") from exc
    if check.returncode != 0:
        detail = check.stderr.strip() or f"xdpyinfo exited {check.returncode}"
        raise CameraError(f"Cannot open preview display {display}: {detail}")

    print(
        f"Camera previews will open on display {display} "
        f"(target={target}, XAUTHORITY={os.environ.get('XAUTHORITY', 'session')})",
        flush=True,
    )
    return display


@dataclass(frozen=True)
class CameraConfig:
    name: str = "camera"
    enabled: bool = True
    required: bool = True
    driver: str = "realsense"
    device: str = "0"
    serial: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = ""
    enable_depth: bool = False
    preview: bool = False
    startup_timeout_s: float = 8.0
    frame_timeout_ms: int = 1000
    max_consecutive_timeouts: int = 5

    @staticmethod
    def from_mapping(values: dict[str, Any] | None) -> "CameraConfig":
        data = values or {}
        config = CameraConfig(
            name=str(data.get("name", "camera")).strip(),
            enabled=bool(data.get("enabled", True)),
            required=bool(data.get("required", True)),
            driver=str(data.get("driver", "realsense")).strip().lower(),
            device=str(data.get("device", "0")).strip(),
            serial=str(data.get("serial", "")).strip(),
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
            fps=int(data.get("fps", 30)),
            fourcc=str(data.get("fourcc", "")).strip(),
            enable_depth=bool(data.get("enable_depth", False)),
            preview=bool(data.get("preview", False)),
            startup_timeout_s=float(data.get("startup_timeout_s", 8.0)),
            frame_timeout_ms=int(data.get("frame_timeout_ms", 1000)),
            max_consecutive_timeouts=int(data.get("max_consecutive_timeouts", 5)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name:
            raise ValueError("camera.name must not be empty")
        if self.driver not in ("realsense", "opencv"):
            raise ValueError("camera.driver must be realsense or opencv")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if self.driver == "opencv" and not self.device:
            raise ValueError("camera.device is required when camera.driver is opencv")
        if self.fourcc and len(self.fourcc) != 4:
            raise ValueError("camera.fourcc must be a four-character code")
        if self.startup_timeout_s <= 0.0:
            raise ValueError("camera.startup_timeout_s must be positive")
        if self.frame_timeout_ms < 100:
            raise ValueError("camera.frame_timeout_ms must be at least 100")
        if self.max_consecutive_timeouts < 1:
            raise ValueError("camera.max_consecutive_timeouts must be at least 1")


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    timestamp_unix_s: float
    captured_monotonic: float
    color_bgr: np.ndarray
    depth_mm: np.ndarray | None = None

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.captured_monotonic


class RealSenseCamera:
    """Continuously captures RealSense RGB frames outside the servo loop."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.name = config.name
        self.device_name = ""
        self.serial = ""
        self._pipeline: Any = None
        self._rs: Any = None
        self._latest: CameraFrame | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.enabled or (self._thread and self._thread.is_alive()):
            return
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError(
                "pyrealsense2 is not installed in the active environment"
            ) from exc

        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.config.serial:
            stream_config.enable_device(self.config.serial)
        stream_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )
        if self.config.enable_depth:
            stream_config.enable_stream(
                rs.stream.depth,
                self.config.width,
                self.config.height,
                rs.format.z16,
                self.config.fps,
            )

        try:
            profile = pipeline.start(stream_config)
        except Exception as exc:
            raise CameraError(f"Failed to start RealSense pipeline: {exc}") from exc

        device = profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.serial = device.get_info(rs.camera_info.serial_number)
        self._rs = rs
        self._pipeline = pipeline
        self._stop_event.clear()
        self._first_frame_event.clear()
        with self._lock:
            self._latest = None
            self._error = None
        # RealSense can need several seconds to deliver its first color frame
        # after a UVC camera has also been opened. Confirm that frame before
        # moving the continuous capture into its background thread.
        try:
            first_frames = pipeline.wait_for_frames(max(self.config.frame_timeout_ms, 3000))
            first_frame = self._make_camera_frame(first_frames, sequence=1)
        except Exception as exc:
            self.stop()
            raise CameraError(f"No initial RealSense color frame arrived: {exc}") from exc
        if first_frame is None:
            self.stop()
            raise CameraError("RealSense returned no initial color frame")
        with self._lock:
            self._latest = first_frame
        self._first_frame_event.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(first_frame.sequence,),
            name="realsense-capture",
            daemon=True,
        )
        self._thread.start()
        if not self._first_frame_event.wait(self.config.startup_timeout_s):
            self.stop()
            raise CameraError(
                f"No RealSense frame arrived within {self.config.startup_timeout_s:.1f}s"
            )
        self.raise_if_failed()
        print(
            f"RealSense camera ready: name={self.name} device={self.device_name} serial={self.serial} "
            f"{self.config.width}x{self.config.height}@{self.config.fps}fps "
            f"depth={self.config.enable_depth}",
            flush=True,
        )

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise CameraError(f"RealSense capture failed: {error}") from error

    def stop(self) -> None:
        self._stop_event.set()
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._destroy_preview()

    def show_preview(self, poll_events: bool = True) -> None:
        if not self.config.preview:
            return
        frame = self.latest()
        if frame is None:
            return
        try:
            import cv2

            preview = frame.color_bgr.copy()
            cv2.putText(
                preview,
                f"{self.name} {self.config.width}x{self.config.height} frame={frame.sequence}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            title = f"Dobot VR Teleop - {self.name}"
            cv2.imshow(title, preview)
            if poll_events and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                cv2.destroyWindow(title)
                object.__setattr__(self.config, "preview", False)
        except Exception as exc:
            self._set_error(CameraError(f"RealSense preview failed: {exc}"))

    def _make_camera_frame(self, frames: Any, sequence: int) -> CameraFrame | None:
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        depth_frame = frames.get_depth_frame() if self.config.enable_depth else None
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = (
            np.asanyarray(depth_frame.get_data()).copy()
            if depth_frame is not None
            else None
        )
        return CameraFrame(
            sequence=sequence,
            timestamp_unix_s=time.time(),
            captured_monotonic=time.monotonic(),
            color_bgr=color,
            depth_mm=depth,
        )

    def _capture_loop(self, sequence: int) -> None:
        consecutive_timeouts = 0
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(self.config.frame_timeout_ms)
            except RuntimeError as exc:
                if self._stop_event.is_set():
                    return
                consecutive_timeouts += 1
                if consecutive_timeouts >= self.config.max_consecutive_timeouts:
                    self._set_error(exc)
                    self._first_frame_event.set()
                    return
                continue
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._set_error(exc)
                    self._first_frame_event.set()
                return

            consecutive_timeouts = 0
            frame = self._make_camera_frame(frames, sequence=sequence + 1)
            if frame is None:
                continue
            sequence += 1
            with self._lock:
                self._latest = frame
            self._first_frame_event.set()

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            self._error = error

    def _destroy_preview(self) -> None:
        if not self.config.preview:
            return
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


class OpenCVCamera:
    """Continuously captures RGB frames from a UVC/OpenCV camera."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.name = config.name
        self.serial = config.device
        self._capture: Any = None
        self._latest: CameraFrame | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.enabled or (self._thread and self._thread.is_alive()):
            return
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("opencv-python is required for opencv camera") from exc

        source: int | str
        source = int(self.config.device) if self.config.device.isdigit() else self.config.device
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise CameraError(f"Failed to open OpenCV camera source: {self.config.device}")

        if self.config.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        self._capture = capture
        self._stop_event.clear()
        self._first_frame_event.clear()
        with self._lock:
            self._latest = None
            self._error = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="opencv-camera-capture",
            daemon=True,
        )
        self._thread.start()
        if not self._first_frame_event.wait(self.config.startup_timeout_s):
            self.stop()
            raise CameraError(
                f"No OpenCV camera frame arrived within {self.config.startup_timeout_s:.1f}s"
            )
        self.raise_if_failed()
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        print(
            f"OpenCV camera ready: source={self.config.device} "
            f"{actual_width}x{actual_height}@{actual_fps:.1f}fps",
            flush=True,
        )

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise CameraError(f"OpenCV camera capture failed: {error}") from error

    def stop(self) -> None:
        self._stop_event.set()
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._destroy_preview()

    def show_preview(self, poll_events: bool = True) -> None:
        if not self.config.preview:
            return
        frame = self.latest()
        if frame is None:
            return
        try:
            import cv2

            preview = frame.color_bgr.copy()
            cv2.putText(
                preview,
                f"{self.name} {self.config.device} frame={frame.sequence}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            title = f"Dobot VR Teleop - {self.name}"
            cv2.imshow(title, preview)
            if poll_events and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                cv2.destroyWindow(title)
                object.__setattr__(self.config, "preview", False)
        except Exception as exc:
            self._set_error(CameraError(f"OpenCV preview failed: {exc}"))

    def _capture_loop(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            self._set_error(exc)
            self._first_frame_event.set()
            return

        sequence = 0
        consecutive_timeouts = 0
        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                return
            ok, color = capture.read()
            if not ok or color is None:
                consecutive_timeouts += 1
                if consecutive_timeouts >= self.config.max_consecutive_timeouts:
                    self._set_error(CameraError("OpenCV camera returned no frame"))
                    self._first_frame_event.set()
                    return
                time.sleep(max(self.config.frame_timeout_ms / 1000.0, 0.001))
                continue

            consecutive_timeouts = 0
            sequence += 1
            frame = CameraFrame(
                sequence=sequence,
                timestamp_unix_s=time.time(),
                captured_monotonic=time.monotonic(),
                color_bgr=color.copy(),
                depth_mm=None,
            )
            with self._lock:
                self._latest = frame
            self._first_frame_event.set()

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            self._error = error

    def _destroy_preview(self) -> None:
        if not self.config.preview:
            return
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


Camera = RealSenseCamera | OpenCVCamera


class CameraGroup:
    """Starts several cameras while preserving the primary camera API for recording."""

    def __init__(self, cameras: list[Camera]) -> None:
        if not cameras:
            raise ValueError("CameraGroup requires at least one enabled camera")
        self.cameras = cameras
        self.config = cameras[0].config

    def start(self) -> None:
        started: list[Camera] = []
        try:
            for camera in self.cameras:
                try:
                    camera.start()
                    started.append(camera)
                except Exception as exc:
                    camera.stop()
                    if camera.config.required:
                        raise CameraError(
                            f"Required camera '{camera.name}' failed to start: {exc}"
                        ) from exc
                    print(f"[camera disabled] {camera.name} startup failed: {exc}", flush=True)
        except Exception:
            for camera in reversed(started):
                camera.stop()
            raise

    def stop(self) -> None:
        for camera in reversed(self.cameras):
            camera.stop()

    def latest(self) -> CameraFrame | None:
        return self.cameras[0].latest()

    def latest_frames(self) -> dict[str, CameraFrame | None]:
        """Return the latest frame for every configured camera by stable name."""
        return {camera.name: camera.latest() for camera in self.cameras}

    def raise_if_failed(self) -> None:
        for camera in self.cameras:
            try:
                camera.raise_if_failed()
            except Exception:
                if camera.config.required:
                    raise

    def show_previews(self) -> None:
        preview_cameras = [camera for camera in self.cameras if camera.config.preview]
        if not preview_cameras:
            return
        for camera in preview_cameras:
            camera.show_preview(poll_events=False)
        try:
            import cv2

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                for camera in preview_cameras:
                    object.__setattr__(camera.config, "preview", False)
                cv2.destroyAllWindows()
        except Exception as exc:
            for camera in preview_cameras:
                camera._set_error(CameraError(f"Camera preview failed: {exc}"))


def create_camera(config: CameraConfig) -> Camera:
    if config.driver == "opencv":
        return OpenCVCamera(config)
    return RealSenseCamera(config)
