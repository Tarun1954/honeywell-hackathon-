"""Run the existing Phase 2 MCP surface over one live EnergyPlus snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.live_energyplus_integration import create_live_services
from src.mcp_server import run_mcp_server
from src.phase2_contracts import SensorSnapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-payload",
        type=Path,
        required=True,
        help="JSON payload containing the current snapshot and bounded history",
    )
    parser.add_argument(
        "--action-output",
        type=Path,
        required=True,
        help="Path where the accepted MCP action is recorded",
    )
    return parser.parse_args()


def _load_payload(
    path: Path,
) -> tuple[SensorSnapshot, tuple[SensorSnapshot, ...]]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("live snapshot payload must be an object")
    snapshot = SensorSnapshot.model_validate(raw.get("snapshot"))
    history_raw = raw.get("history", [])
    if not isinstance(history_raw, list):
        raise ValueError("live snapshot history must be an array")
    history = tuple(
        SensorSnapshot.model_validate(item) for item in history_raw
    )
    return snapshot, history


def main() -> None:
    args = _parse_args()
    snapshot_path = args.snapshot_payload.resolve(strict=True)
    action_output_path = args.action_output.resolve()
    action_output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot, history = _load_payload(snapshot_path)
    services = create_live_services(
        snapshot,
        history,
        action_output_path=action_output_path,
    )
    run_mcp_server(services)


if __name__ == "__main__":
    main()
