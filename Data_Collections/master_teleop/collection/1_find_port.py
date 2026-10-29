from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Data_Collections"))

from master_teleop.collection.workflow import probe_master, serial_candidates, update_runtime_master
from master_teleop.core.config import load_master_project_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: find the single Xtrainer master-hand serial port and baud rate."
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--port", action="append", default=[], help="Only try this port; repeatable.")
    parser.add_argument("--no-save", action="store_true", help="Probe without writing master_runtime.yaml.")
    args = parser.parse_args()

    config = load_master_project_config(args.config_dir)
    values = config["master_hand"]
    configured = str(values.get("device", ""))
    ports = serial_candidates([*args.port, configured])
    if not ports:
        raise SystemExit("No /dev/serial/by-id, ttyACM, or ttyUSB device was found")
    baud_rates = list(
        dict.fromkeys(
            int(value)
            for value in values.get("baud_rate_candidates", [2000000, 1000000])
        )
    )
    profiles = list(values.get("profiles", {}))
    print(f"serial candidates: {ports}")
    print(f"baud candidates: {baud_rates}; profiles: {profiles}")

    failures: list[str] = []
    for port in ports:
        for baud_rate in baud_rates:
            for profile in profiles:
                master = None
                try:
                    master, sample = probe_master(values, port, baud_rate, profile)
                    print(
                        f"FOUND profile={sample.profile_name} port={port} baud={baud_rate} "
                        f"raw_ids={values['profiles'][profile]['joint_ids']}"
                    )
                    print(f"buttons_raw={sample.buttons_raw} raw_positions={sample.raw_positions}")
                    if not args.no_save:
                        path = update_runtime_master(
                            {
                                "device": port,
                                "baud_rate": baud_rate,
                                "baud_rate_candidates": [baud_rate, *[v for v in baud_rates if v != baud_rate]],
                                "side": profile,
                            }
                        )
                        print(f"saved: {path}")
                    return
                except Exception as exc:
                    failures.append(f"{port}@{baud_rate}/{profile}: {exc}")
                finally:
                    if master is not None:
                        master.close()
    print("Probe failures:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit("Could not identify the single master hand")


if __name__ == "__main__":
    main()
