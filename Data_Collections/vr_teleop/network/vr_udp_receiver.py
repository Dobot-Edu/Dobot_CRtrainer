from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from typing import Optional

from ..control.models import VRInput


@dataclass
class ReceiverStats:
    packets: int = 0
    bad_packets: int = 0
    last_sender: tuple[str, int] | None = None
    last_error: str = ""


class VRUDPReceiver:
    def __init__(self, listen_ip: str, listen_port: int, buffer_size: int = 4096) -> None:
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.buffer_size = buffer_size
        self.stats = ReceiverStats()
        self._latest: Optional[VRInput] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.2)
        self._sock.bind((self.listen_ip, self.listen_port))
        self._thread = threading.Thread(target=self._run, name="vr-udp-receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()
            self._sock = None

    def latest(self) -> VRInput | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                payload, sender = self._sock.recvfrom(self.buffer_size)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self.stats.last_error = str(exc)
                break

            try:
                data = json.loads(payload.decode("utf-8"))
                vr_input = VRInput.from_json(data)
            except Exception as exc:  # Keep malformed packets from killing teleop.
                self.stats.bad_packets += 1
                self.stats.last_error = str(exc)
                continue

            with self._lock:
                self._latest = vr_input
            self.stats.packets += 1
            self.stats.last_sender = sender
