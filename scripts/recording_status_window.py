from __future__ import annotations

import argparse

import numpy as np


WINDOW_TITLE = "Dobot Recording Status"
NOT_RECORDING_BGR = (0, 0, 255)
RECORDING_BGR = (0, 255, 0)


def make_status_frame(
    recording: bool,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Return a solid red/green BGR image matching a camera preview frame."""

    if width <= 0 or height <= 0:
        raise ValueError("recording status window width and height must be positive")
    color = RECORDING_BGR if recording else NOT_RECORDING_BGR
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class RecordingStatusWindow:
    """OpenCV indicator: red while idle and green while recording."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        title: str = WINDOW_TITLE,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.title = str(title)
        self._frames = {
            False: make_status_frame(False, self.width, self.height),
            True: make_status_frame(True, self.width, self.height),
        }
        self._opened = False
        self._last_recording: bool | None = None

    def show(self, recording: bool) -> None:
        """Create/update the window; OpenCV's shared event loop is polled by camera previews."""

        import cv2

        state = bool(recording)
        if not self._opened:
            cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
            self._opened = True
        if self._last_recording is state:
            return
        cv2.imshow(self.title, self._frames[state])
        self._last_recording = state

    def close(self) -> None:
        if not self._opened:
            return
        try:
            import cv2

            cv2.destroyWindow(self.title)
        except Exception:
            pass
        finally:
            self._opened = False
            self._last_recording = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview the solid-color Dobot recording status window."
    )
    parser.add_argument(
        "--recording",
        action="store_true",
        help="Show green instead of the default red.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    import cv2

    window = RecordingStatusWindow(args.width, args.height)
    try:
        window.show(args.recording)
        print("Press q or Esc in the status window to close it.", flush=True)
        while cv2.waitKey(20) & 0xFF not in (ord("q"), 27):
            pass
    finally:
        window.close()


if __name__ == "__main__":
    main()
