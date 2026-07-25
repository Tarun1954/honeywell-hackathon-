"""Run the short Eco-Loop live demo or replay verified saved evidence.

Usage:
    python -m scripts.run_final_demo
    python -m scripts.run_final_demo --mode live
    python -m scripts.run_final_demo --mode replay

The default ``auto`` mode attempts the proven four-hour live Ollama smoke and
falls back to the committed verified report if a local dependency is
unavailable. Replay mode never starts EnergyPlus or contacts Ollama.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from src.mcp_client import PHASE2_TOOL_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = (
    ROOT / "submission" / "results" / "live_ollama_smoke_report.json"
)
DEFAULT_COMPARISON_PATH = (
    ROOT / "runs" / "final" / "ollama_24h_comparison.json"
)
DemoMode = Literal["auto", "live", "replay"]
Emitter = Callable[[str], None]
LiveRunner = Callable[[], Mapping[str, Any]]


class DemoEvidenceError(RuntimeError):
    """Saved demo evidence is absent or malformed."""


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "live", "replay"),
        default="auto",
        help=(
            "auto tries live then replays verified evidence on failure; "
            "live disables replay; replay never contacts local services"
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
        help="Saved verified live-smoke report used by replay mode",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=DEFAULT_COMPARISON_PATH,
        help="Verified 24-hour comparison JSON printed at the end",
    )
    return parser.parse_args(arguments)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DemoEvidenceError(f"{label} file is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoEvidenceError(f"{label} file is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise DemoEvidenceError(f"{label} must contain one JSON object")
    return payload


def _run_live_workflow() -> Mapping[str, Any]:
    # Import lazily so replay mode cannot initialize the live-provider path.
    from scripts.run_live_ollama_energyplus_smoke import (
        run_live_ollama_smoke,
    )

    return run_live_ollama_smoke()


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DemoEvidenceError(f"demo evidence is missing object {key!r}")
    return value


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoEvidenceError(f"demo evidence has invalid number {label!r}")
    return float(value)


def _action_summary(report: Mapping[str, Any]) -> str:
    accepted = _require_mapping(report, "accepted_action")
    chosen = _require_mapping(accepted, "chosen_setpoints")
    set_commands = [
        command
        for command in chosen.values()
        if isinstance(command, Mapping) and command.get("mode") == "set"
    ]
    if set_commands:
        heating_values = {
            _number(command.get("heating_c"), label="heating_c")
            for command in set_commands
        }
        cooling_values = {
            _number(command.get("cooling_c"), label="cooling_c")
            for command in set_commands
        }
        if len(heating_values) == 1 and len(cooling_values) == 1:
            heating = next(iter(heating_values))
            cooling = next(iter(cooling_values))
            return (
                f"Set all {len(set_commands)} zones to heating "
                f"{heating:.1f} C and cooling {cooling:.1f} C for four "
                "15-minute timesteps."
            )
        return (
            f"Apply validated heating/cooling commands to "
            f"{len(set_commands)} zones for four 15-minute timesteps."
        )
    return "Release all five thermostat actuator pairs to their IDF schedules."


def _emit_architecture(emit: Emitter) -> None:
    emit("[1/8] PROJECT ARCHITECTURE")
    emit(
        "EnergyPlus sensors -> Phase 2 SensorSnapshot -> MCP client/server "
        "-> Ollama tool agent -> safety validation -> EnergyPlus actuators"
    )
    emit(
        "Control cadence: one decision per simulated hour; accepted action "
        "held for four 15-minute timesteps."
    )


def _emit_sensor_snapshot(
    report: Mapping[str, Any],
    emit: Emitter,
) -> None:
    snapshot = _require_mapping(report, "live_sensor_snapshot")
    temperatures = _require_mapping(snapshot, "zone_temperatures_c")
    pmv = _require_mapping(snapshot, "pmv")
    occupancy = _require_mapping(snapshot, "occupancy")

    emit("[2/8] LIVE SENSOR SNAPSHOT ENTERING THE AGENT")
    emit(f"Simulated timestamp: {snapshot.get('simulated_timestamp')}")
    emit(f"Snapshot ID: {snapshot.get('snapshot_id')}")
    for zone in sorted(temperatures):
        emit(
            f"  {zone}: temperature="
            f"{_number(temperatures[zone], label='temperature'):.2f} C | "
            f"PMV={_number(pmv.get(zone), label='pmv'):.2f} | "
            f"occupancy={_number(occupancy.get(zone), label='occupancy'):.0f}"
        )
    demand_w = _number(
        snapshot.get("facility_electricity_demand_w"),
        label="facility_electricity_demand_w",
    )
    energy_kwh = _number(
        snapshot.get("facility_electricity_kwh_since_start"),
        label="facility_electricity_kwh_since_start",
    )
    emit(
        f"Facility electricity: demand={demand_w:.1f} W | "
        f"energy since start={energy_kwh:.3f} kWh"
    )


def _emit_mcp_and_action(
    report: Mapping[str, Any],
    emit: Emitter,
) -> None:
    emit("[3/8] MCP TOOL DISCOVERY AND CALLS")
    for tool in PHASE2_TOOL_NAMES:
        emit(f"  DISCOVERED: {tool}")
    called = report.get("mcp_tools_called")
    if not isinstance(called, list) or not all(
        isinstance(tool, str) for tool in called
    ):
        raise DemoEvidenceError("demo evidence has invalid mcp_tools_called")
    for index, tool in enumerate(called, start=1):
        emit(f"  MCP CALL {index}: {tool}")

    accepted = _require_mapping(report, "accepted_action")
    status = str(accepted.get("status", "unknown"))
    fallback = bool(accepted.get("fallback_used", False))
    rejected = int(accepted.get("rejected_action_count", 0) or 0)
    corrected = int(accepted.get("corrected_action_count", 0) or 0)

    emit("[4/8] OLLAMA CONTROL DECISION")
    emit(
        "Concise LLM decision summary (action-level, no hidden reasoning): "
        + _action_summary(report)
    )
    emit("[5/8] SAFETY VALIDATION")
    if status == "accepted":
        emit("Safety validation: ACCEPTED; complete five-zone action, no clamping.")
    else:
        emit(f"Safety validation: {status.upper()}")
    emit(
        f"Action status: {status} | rejected={rejected} | "
        f"corrected={corrected} | fallback_used={str(fallback).lower()}"
    )


def _emit_writeback(
    report: Mapping[str, Any],
    emit: Emitter,
) -> None:
    changes = _require_mapping(report, "setpoint_changes")
    emit("[6/8] ENERGYPLUS SETPOINT WRITEBACK")
    for zone in sorted(changes):
        change = changes[zone]
        if not isinstance(change, Mapping):
            raise DemoEvidenceError("invalid setpoint change record")
        previous = _require_mapping(change, "previous")
        updated = _require_mapping(change, "updated")
        emit(
            f"  {zone}: heat "
            f"{_number(previous.get('heating_c'), label='previous heat'):.1f}"
            f" -> {_number(updated.get('heating_c'), label='updated heat'):.1f} C; "
            f"cool "
            f"{_number(previous.get('cooling_c'), label='previous cool'):.1f}"
            f" -> {_number(updated.get('cooling_c'), label='updated cool'):.1f} C"
        )

    write = _require_mapping(report, "actuator_write_result")
    zones = _require_mapping(write, "zones")
    successful = sum(
        isinstance(value, Mapping) and value.get("status") == "applied"
        for value in zones.values()
    )
    emit(
        f"Actuator-write success: status={write.get('status')} | "
        f"zones_applied={successful}/{len(zones)}"
    )

    energyplus = _require_mapping(report, "energyplus")
    emit("[7/8] ENERGYPLUS COMPLETION")
    emit(
        f"EnergyPlus exit={energyplus.get('exit_status')} | "
        f"severe={energyplus.get('severe_count')} | "
        f"fatal={energyplus.get('fatal_count')} | "
        f"timesteps={energyplus.get('timestep_count')}"
    )


def _emit_comparison(
    comparison: Mapping[str, Any],
    emit: Emitter,
) -> None:
    electricity = _require_mapping(comparison, "electricity")
    peak = _require_mapping(comparison, "peak_demand")
    baseline = _require_mapping(comparison, "baseline_comfort")
    controlled = _require_mapping(comparison, "controlled_comfort")
    actions = _require_mapping(comparison, "control_actions")
    reduction = abs(
        _number(
            electricity.get("percentage_difference_from_baseline"),
            label="percentage_difference_from_baseline",
        )
    )

    emit("[8/8] VERIFIED 24-HOUR COMPARISON RESULTS")
    emit(f"Label: {comparison.get('comparison_label')}")
    emit(
        "Electricity: baseline="
        f"{_number(electricity.get('baseline_total_kwh'), label='baseline kWh'):.3f} "
        "kWh | Ollama hybrid="
        f"{_number(electricity.get('controlled_total_kwh'), label='controlled kWh'):.3f} "
        "kWh | reduction="
        f"{_number(electricity.get('absolute_difference_kwh'), label='difference kWh'):.3f} "
        f"kWh ({reduction:.3f}%)"
    )
    emit(
        f"Peak demand: baseline="
        f"{_number(peak.get('baseline_w'), label='baseline peak') / 1000:.3f} kW | "
        f"controlled="
        f"{_number(peak.get('controlled_w'), label='controlled peak') / 1000:.3f} kW"
    )
    emit(
        "Occupied PMV compliance: baseline="
        f"{_number(baseline.get('occupied_pmv_compliance_percent'), label='baseline compliance'):.2f}% "
        "| controlled="
        f"{_number(controlled.get('occupied_pmv_compliance_percent'), label='controlled compliance'):.2f}%"
    )
    emit(
        "Comfort violations: baseline="
        f"{baseline.get('occupied_comfort_violation_count')} | controlled="
        f"{controlled.get('occupied_comfort_violation_count')}"
    )
    emit(
        f"Hourly actions: accepted={actions.get('real_actions_accepted')} | "
        f"rejected={actions.get('actions_rejected')} | "
        f"fallback={actions.get('fallback_count')}"
    )
    emit(
        "Trade-off: energy and peak demand decreased, while occupied PMV "
        "compliance decreased slightly and comfort violations increased."
    )


def _render_report(
    report: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    emit: Emitter,
) -> None:
    _emit_architecture(emit)
    _emit_sensor_snapshot(report, emit)
    _emit_mcp_and_action(report, emit)
    _emit_writeback(report, emit)
    _emit_comparison(comparison, emit)
    emit("=== DEMO COMPLETE ===")


def run_demo(
    *,
    mode: DemoMode,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    comparison_path: Path = DEFAULT_COMPARISON_PATH,
    live_runner: LiveRunner | None = None,
    emit: Emitter = print,
) -> str:
    """Run one live/replay demo and return the mode actually rendered."""

    comparison = _load_object(comparison_path, label="comparison")
    emit("=== ECO-LOOP FINAL DEMONSTRATION ===")
    report: Mapping[str, Any]
    actual_mode: str

    if mode == "replay":
        emit(
            "DEMO MODE: SAVED-EVIDENCE REPLAY — previously captured verified "
            "evidence; this is not a live run."
        )
        emit(
            f"Replay source: {evidence_path.relative_to(ROOT) if evidence_path.is_relative_to(ROOT) else evidence_path.name}"
        )
        report = _load_object(evidence_path, label="saved live-smoke evidence")
        actual_mode = "replay"
    else:
        emit(
            "DEMO MODE: LIVE — starting the proven four-hour Ollama/EnergyPlus "
            "smoke workflow."
        )
        emit("EnergyPlus starting; waiting for live sensors and hourly agent cycles...")
        runner = live_runner or _run_live_workflow
        try:
            report = runner()
        except Exception as exc:
            # Deliberately print only the exception class. Provider messages can
            # contain local URLs; environment values are never rendered.
            emit(f"Live demo unavailable: {type(exc).__name__}")
            if mode == "live":
                emit(
                    "No replay requested. Re-run with --mode replay or use "
                    "the default --mode auto."
                )
                raise
            emit(
                "FALLBACK MODE: SAVED-EVIDENCE REPLAY — previously captured "
                "verified evidence; this is not a live run."
            )
            report = _load_object(
                evidence_path,
                label="saved live-smoke evidence",
            )
            actual_mode = "replay"
        else:
            emit("Live Ollama/EnergyPlus smoke completed.")
            actual_mode = "live"

    _render_report(report, comparison, emit=emit)
    return actual_mode


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    try:
        run_demo(
            mode=args.mode,
            evidence_path=args.evidence,
            comparison_path=args.comparison,
        )
    except Exception as exc:
        print(
            f"Demo failed safely: {type(exc).__name__}. "
            "No environment values were displayed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COMPARISON_PATH",
    "DEFAULT_EVIDENCE_PATH",
    "DemoEvidenceError",
    "main",
    "run_demo",
]
