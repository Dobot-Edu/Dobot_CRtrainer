from __future__ import annotations

import json
import socket
import threading


DISCOVERY_REQUEST = b"DOBOT_CRTRAINER_DISCOVER_V1"


class VRHostDiscoveryResponder:
    """Reply to Quest discovery broadcasts with the VR UDP port."""

    def __init__(self, bind_ip: str, discovery_port: int, vr_port: int) -> None:
        self.bind_ip = bind_ip
        self.discovery_port = int(discovery_port)
        self.vr_port = int(vr_port)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_ip, self.discovery_port))
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._run,
            name="vr-host-discovery",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()
            self._sock = None

    def _run(self) -> None:
        assert self._sock is not None
        response = json.dumps(
            {
                "type": "dobot_crtrainer_host",
                "version": 1,
                "port": self.vr_port,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        while not self._stop.is_set():
            try:
                payload, sender = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    raise
                return

            if payload.strip() == DISCOVERY_REQUEST:
                try:
                    self._sock.sendto(response, sender)
                except OSError:
                    if not self._stop.is_set():
                        raise
