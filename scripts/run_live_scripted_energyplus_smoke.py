"""Run a four-hour live EnergyPlus/MCP/ScriptedProvider smoke cycle."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.energyplus_wrapper import EnergyPlusWrapper, Phase1Config
from src.live_energyplus_integration import LiveScriptedAgentController


LOGGER = logging.getLogger("eco_loop.phase2.live_smoke")
DEFAULT_TIMESTEPS = 16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/phase1.yaml",
        help="Path to the proven Phase 1 configuration",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Completed 15-minute timesteps before a controlled stop",
    )
    return parser.parse_args()


def _setpoint_changes(event: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    previous = event["previous_setpoints"]
    chosen = event["chosen_setpoints"]
    for zone, selected in chosen.items():
        if selected.get("mode") != "set":
            continue
        before = previous[zone]
        if (
            before["heating_c"] != selected["heating_c"]
            or before["cooling_c"] != selected["cooling_c"]
        ):
            changes[zone] = {
                "previous": before,
                "updated": {
                    "heating_c": selected["heating_c"],
                    "cooling_c": selected["cooling_c"],
                },
            }
    return changes


def run_live_smoke(
    config_path: str | Path,
    *,
    timesteps: int = DEFAULT_TIMESTEPS,
) -> dict[str, Any]:
    """Run and verify the minimal live scripted integration."""

    if timesteps != DEFAULT_TIMESTEPS:
        raise ValueError("the focused live smoke must run exactly 16 timesteps")
    config = Phase1Config.load(config_path)
    run_suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_directory = (
        config.output_root / f"live-scripted-artifacts-{run_suffix}"
    )
    controller = LiveScriptedAgentController(
        repository_root=config.repository_root,
        work_directory=artifact_directory,
        controlled_zones=config.controlled_zones,
        safety_limits=config.safety,
    )
    result = EnergyPlusWrapper(config).run(
        run_id="live-scripted-smoke",
        mode="live-scripted",
        policy=controller,
        model_path=config.baseline_model,
        max_timesteps=timesteps,
    )

    decision_events = [
        dict(event)
        for event in controller.events
        if event["action_status"] == "accepted"
        and not event["fallback_used"]
    ]
    if not decision_events:
        raise RuntimeError("no live scripted action was accepted")
    accepted = decision_events[0]
    changes = _setpoint_changes(accepted)
    if not changes:
        raise RuntimeError(
            "the accepted live action did not change an EnergyPlus setpoint"
        )
    write_events = [
        dict(event)
        for event in controller.events
        if event["sequence"] > accepted["sequence"]
        and event["actuator_write_result"].get("source")
        == "phase2-scripted-agent"
    ]
    if not write_events:
        raise RuntimeError("no accepted live action reached actuator writeback")
    write_evidence = write_events[0]["actuator_write_result"]
    if (
        result.exit_code != 0
        or result.severe_count
        or result.fatal_count
        or result.timestep_count != timesteps
    ):
        raise RuntimeError("the focused EnergyPlus smoke acceptance gates failed")

    report = {
        "status": "passed",
        "simulated_hours": (
            result.timestep_count * config.timestep_minutes / 60.0
        ),
        "live_sensor_snapshot": {
            key: accepted[key]
            for key in (
                "simulated_timestamp",
                "snapshot_id",
                "zone_temperatures_c",
                "pmv",
                "occupancy",
                "facility_electricity_demand_w",
                "facility_electricity_kwh_since_start",
            )
        },
        "mcp_tools_called": accepted["mcp_tools_called"],
        "accepted_action": {
            "status": accepted["action_status"],
            "fallback_used": accepted["fallback_used"],
            "chosen_setpoints": accepted["chosen_setpoints"],
        },
        "setpoint_changes": changes,
        "actuator_write_result": write_evidence,
        "energyplus": {
            "exit_status": result.exit_code,
            "severe_count": result.severe_count,
            "fatal_count": result.fatal_count,
            "timestep_count": result.timestep_count,
            "output_directory": result.output_directory,
        },
        "integration_log": str(controller.log_path),
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_path = artifact_directory / "live_smoke_report.json"
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    report = run_live_smoke(args.config, timesteps=args.timesteps)
    LOGGER.info(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
