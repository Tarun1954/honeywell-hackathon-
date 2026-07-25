"""Aggregate current deterministic, scripted, and pending Ollama evidence."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_REPORT = Path("runs/phase1/phase1_report.json")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/final")
DEFAULT_DOCS_PATH = Path("docs/current_results.md")
COMFORT_PMV_LIMIT = 0.7
SCHEMA_VERSION = "eco-loop.results.v1"


def calculate_energy_reduction(
    baseline_kwh: float,
    controlled_kwh: float,
) -> tuple[float, float]:
    """Return baseline-minus-controlled kWh and percent reduction."""

    baseline = float(baseline_kwh)
    controlled = float(controlled_kwh)
    if baseline <= 0.0:
        raise ValueError("baseline electricity must be positive")
    absolute = baseline - controlled
    return absolute, absolute / baseline * 100.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"JSONL record {line_number} must be an object: {path}"
            )
        records.append(payload)
    return records


def _relative_path(repository_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _resolve_run_directory(
    repository_root: Path,
    raw_path: str,
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_dir():
        return candidate.resolve()
    rebased = repository_root / "runs" / "phase1" / candidate.name
    if rebased.is_dir():
        return rebased.resolve()
    raise FileNotFoundError(f"Run output directory is unavailable: {raw_path}")


def _range(values: Iterable[float]) -> dict[str, float] | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return {"minimum": min(collected), "maximum": max(collected)}


def _sensor_metrics(
    samples: Iterable[
        tuple[str, float, float, float]
    ],
) -> dict[str, Any]:
    temperatures: list[float] = []
    pmvs: list[float] = []
    occupied_samples = 0
    comfort_violations = 0
    by_zone_values: dict[str, dict[str, list[float]]] = {}
    for zone_id, temperature, pmv, occupancy in samples:
        temperatures.append(temperature)
        pmvs.append(pmv)
        values = by_zone_values.setdefault(
            zone_id,
            {"temperature": [], "pmv": []},
        )
        values["temperature"].append(temperature)
        values["pmv"].append(pmv)
        if occupancy > 0.0:
            occupied_samples += 1
            comfort_violations += abs(pmv) > COMFORT_PMV_LIMIT
    return {
        "zone_temperature_c_range": _range(temperatures),
        "pmv_range": _range(pmvs),
        "by_zone": {
            zone_id: {
                "zone_temperature_c_range": _range(values["temperature"]),
                "pmv_range": _range(values["pmv"]),
            }
            for zone_id, values in sorted(by_zone_values.items())
        },
        "occupied_zone_samples": occupied_samples,
        "comfort_violation_count": comfort_violations,
        "comfort_violation_definition": (
            f"occupied zone sample with absolute Fanger PMV > "
            f"{COMFORT_PMV_LIMIT}"
        ),
    }


def _phase1_samples(
    records: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[str, float, float, float]]:
    for record in records:
        snapshot = record["snapshot"]
        for zone_id, zone in snapshot["zones"].items():
            yield (
                str(zone_id),
                float(zone["air_temperature_c"]),
                float(zone["fanger_pmv"]),
                float(zone["occupant_count"]),
            )


def _live_samples(
    records: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[str, float, float, float]]:
    for record in records:
        temperatures = record["zone_temperatures_c"]
        pmvs = record["pmv"]
        occupancy = record["occupancy"]
        for zone_id in sorted(temperatures):
            yield (
                str(zone_id),
                float(temperatures[zone_id]),
                float(pmvs[zone_id]),
                float(occupancy[zone_id]),
            )


def _phase1_action_metrics(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = list(records)
    selected_statuses = [
        str(record["action_selected_for_next_timestep"]["status"])
        for record in materialized
    ]
    actuator_timestep_success_count = 0
    actuator_write_success_count = 0
    for record in materialized:
        applied = record["action_applied"]
        successful_zones = sum(
            zone_result.get("status") == "applied"
            for zone_result in applied["zones"].values()
        )
        actuator_write_success_count += successful_zones
        actuator_timestep_success_count += (
            applied.get("status") == "applied"
            and successful_zones == len(applied["zones"])
        )
    return {
        "actions_accepted": selected_statuses.count("accepted"),
        "actions_rejected": selected_statuses.count("rejected"),
        "correction_count": 0,
        "fallback_count": 0,
        "mcp_tools_called": [],
        "mcp_tool_call_count": 0,
        "actuator_write_success_count": actuator_write_success_count,
        "actuator_write_timestep_success_count": (
            actuator_timestep_success_count
        ),
    }


def _live_action_metrics(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = list(records)
    decision_records = [
        record
        for record in materialized
        if record.get("mcp_tools_called")
        or record.get("action_status")
        in {"accepted", "rejected", "fallback_release"}
    ]
    flattened_tools = [
        str(tool)
        for record in decision_records
        for tool in record.get("mcp_tools_called", [])
    ]
    unique_tools = list(dict.fromkeys(flattened_tools))
    actuator_timestep_success_count = 0
    actuator_write_success_count = 0
    for record in materialized:
        applied = record.get("actuator_write_result", {})
        zones = applied.get("zones", {})
        successful_zones = sum(
            zone_result.get("status") == "applied"
            for zone_result in zones.values()
        )
        actuator_write_success_count += successful_zones
        actuator_timestep_success_count += bool(zones) and (
            successful_zones == len(zones)
        )
    return {
        "actions_accepted": sum(
            record.get("action_status") == "accepted"
            and not record.get("fallback_used", False)
            for record in decision_records
        ),
        "actions_rejected": sum(
            record.get("action_status") == "rejected"
            for record in decision_records
        ),
        "correction_count": sum(
            int(record.get("correction_count", 0))
            for record in decision_records
        ),
        "fallback_count": sum(
            bool(record.get("fallback_used", False))
            for record in decision_records
        ),
        "mcp_tools_called": unique_tools,
        "mcp_tool_call_count": len(flattened_tools),
        "actuator_write_success_count": actuator_write_success_count,
        "actuator_write_timestep_success_count": (
            actuator_timestep_success_count
        ),
    }


def _aggregate_phase1(
    repository_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    report = _read_json(report_path)
    baseline = report["baseline"]
    controlled = report["actuated"]
    baseline_directory = _resolve_run_directory(
        repository_root,
        str(baseline["output_directory"]),
    )
    controlled_directory = _resolve_run_directory(
        repository_root,
        str(controlled["output_directory"]),
    )
    baseline_records = _read_jsonl(
        baseline_directory / "timesteps.jsonl"
    )
    controlled_records = _read_jsonl(
        controlled_directory / "timesteps.jsonl"
    )
    baseline_kwh = float(baseline["facility_electricity_kwh"])
    controlled_kwh = float(controlled["facility_electricity_kwh"])
    reduction_kwh, reduction_percent = calculate_energy_reduction(
        baseline_kwh,
        controlled_kwh,
    )
    timestep_count = int(controlled["timestep_count"])
    timestep_minutes = int(report["timestep_minutes"])
    return {
        "status": "available",
        "evidence_class": "deterministic_phase1",
        "evidence_label": (
            "Phase 1 deterministic seven-day baseline/control comparison "
            "(not an LLM result)"
        ),
        "provider": "deterministic_phase1_policy",
        "llm_generated": False,
        "source_artifacts": [
            _relative_path(repository_root, report_path),
            _relative_path(
                repository_root,
                baseline_directory / "timesteps.jsonl",
            ),
            _relative_path(
                repository_root,
                controlled_directory / "timesteps.jsonl",
            ),
        ],
        "baseline_electricity_kwh": baseline_kwh,
        "controlled_electricity_kwh": controlled_kwh,
        "absolute_energy_reduction_kwh": reduction_kwh,
        "percentage_energy_reduction": reduction_percent,
        "simulation_duration_hours": (
            timestep_count * timestep_minutes / 60.0
        ),
        "timestep_count": timestep_count,
        "timestep_minutes": timestep_minutes,
        "baseline_sensors": _sensor_metrics(
            _phase1_samples(baseline_records)
        ),
        "controlled_sensors": _sensor_metrics(
            _phase1_samples(controlled_records)
        ),
        "control_actions": _phase1_action_metrics(controlled_records),
        "energyplus": {
            "baseline_exit_status": int(baseline["exit_code"]),
            "controlled_exit_status": int(controlled["exit_code"]),
            "baseline_severe_count": int(baseline["severe_count"]),
            "controlled_severe_count": int(controlled["severe_count"]),
            "baseline_fatal_count": int(baseline["fatal_count"]),
            "controlled_fatal_count": int(controlled["fatal_count"]),
        },
    }


def _find_latest_report(
    repository_root: Path,
    pattern: str,
) -> Path | None:
    candidates = sorted(repository_root.glob(pattern))
    successful: list[Path] = []
    for candidate in candidates:
        try:
            if _read_json(candidate).get("status") == "passed":
                successful.append(candidate.resolve())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return successful[-1] if successful else None


def _aggregate_live(
    repository_root: Path,
    report_path: Path,
    *,
    evidence_class: str,
    evidence_label: str,
    provider: str,
    llm_generated: bool,
) -> dict[str, Any]:
    report = _read_json(report_path)
    integration_log = Path(str(report["integration_log"]))
    if not integration_log.is_file():
        integration_log = report_path.parent / "live_agent_timesteps.jsonl"
    records = _read_jsonl(integration_log)
    output_directory = _resolve_run_directory(
        repository_root,
        str(report["energyplus"]["output_directory"]),
    )
    energyplus_summary_path = output_directory / "summary.json"
    energyplus_summary = _read_json(energyplus_summary_path)
    timestep_count = int(energyplus_summary["timestep_count"])
    return {
        "status": "available",
        "evidence_class": evidence_class,
        "evidence_label": evidence_label,
        "provider": provider,
        "llm_generated": llm_generated,
        "source_artifacts": [
            _relative_path(repository_root, report_path),
            _relative_path(repository_root, integration_log),
            _relative_path(repository_root, energyplus_summary_path),
        ],
        "facility_electricity_kwh": float(
            energyplus_summary["facility_electricity_kwh"]
        ),
        "absolute_energy_reduction_kwh": None,
        "percentage_energy_reduction": None,
        "energy_comparison_status": (
            "not_calculated_no_matched_baseline"
        ),
        "simulation_duration_hours": float(report["simulated_hours"]),
        "timestep_count": timestep_count,
        "timestep_minutes": 15,
        "sensors": _sensor_metrics(_live_samples(records)),
        "control_actions": _live_action_metrics(records),
        "setpoint_changes": report.get("setpoint_changes", {}),
        "energyplus": {
            "exit_status": int(energyplus_summary["exit_code"]),
            "severe_count": int(energyplus_summary["severe_count"]),
            "fatal_count": int(energyplus_summary["fatal_count"]),
        },
    }


def _pending_ollama() -> dict[str, Any]:
    return {
        "status": "pending",
        "evidence_class": "real_ollama_provider",
        "evidence_label": "Real Ollama live evidence (pending)",
        "provider": "OllamaToolProvider",
        "llm_generated": None,
        "pending_reason": (
            "No completed live-Ollama smoke report is available. "
            "No real-LLM result or savings claim is reported."
        ),
        "source_artifacts": [],
        "facility_electricity_kwh": None,
        "absolute_energy_reduction_kwh": None,
        "percentage_energy_reduction": None,
        "simulation_duration_hours": None,
        "timestep_count": None,
        "sensors": None,
        "control_actions": None,
    }


def aggregate_results(
    repository_root: Path,
    *,
    phase1_report_path: Path | None = None,
    scripted_report_path: Path | None = None,
    ollama_report_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate available evidence without running a simulation or provider."""

    root = repository_root.resolve()
    phase1_path = (
        phase1_report_path
        if phase1_report_path is not None
        else root / DEFAULT_PHASE1_REPORT
    ).resolve()
    scripted_path = scripted_report_path
    if scripted_path is None:
        scripted_path = _find_latest_report(
            root,
            "runs/phase1/live-scripted-artifacts-*/live_smoke_report.json",
        )
    if scripted_path is None:
        raise FileNotFoundError(
            "No successful live ScriptedProvider smoke report is available"
        )
    ollama_path = ollama_report_path
    if ollama_path is None:
        ollama_path = _find_latest_report(
            root,
            "runs/phase1/live-ollama-artifacts-*/live_smoke_report.json",
        )
    ollama = (
        _aggregate_live(
            root,
            ollama_path.resolve(),
            evidence_class="real_ollama_provider",
            evidence_label="Real Ollama live EnergyPlus evidence",
            provider="OllamaToolProvider",
            llm_generated=True,
        )
        if ollama_path is not None
        else _pending_ollama()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Current quantitative evidence",
        "claim_scope": (
            "Phase 1 savings are deterministic policy results. "
            "ScriptedProvider evidence proves live integration only and is "
            "not real-LLM-generated. Real Ollama evidence is reported only "
            "when a completed artifact exists."
        ),
        "phase1_deterministic": _aggregate_phase1(root, phase1_path),
        "live_scripted_provider": _aggregate_live(
            root,
            scripted_path.resolve(),
            evidence_class="live_scripted_provider",
            evidence_label=(
                "Live EnergyPlus + ScriptedProvider smoke "
                "(deterministic, not an LLM result)"
            ),
            provider="ScriptedProvider",
            llm_generated=False,
        ),
        "real_ollama": ollama,
    }


def _csv_rows(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        section: str,
        label: str,
        status: str,
        metric: str,
        value: Any,
        unit: str = "",
    ) -> None:
        if isinstance(value, (dict, list)):
            rendered = json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        elif value is None:
            rendered = ""
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        rows.append(
            {
                "section": section,
                "evidence_label": label,
                "status": status,
                "metric": metric,
                "value": rendered,
                "unit": unit,
            }
        )

    phase1 = summary["phase1_deterministic"]
    for metric, unit in (
        ("baseline_electricity_kwh", "kWh"),
        ("controlled_electricity_kwh", "kWh"),
        ("absolute_energy_reduction_kwh", "kWh"),
        ("percentage_energy_reduction", "%"),
        ("simulation_duration_hours", "hours"),
        ("timestep_count", "timesteps"),
    ):
        add(
            "phase1_deterministic",
            phase1["evidence_label"],
            phase1["status"],
            metric,
            phase1[metric],
            unit,
        )
    add(
        "phase1_deterministic",
        phase1["evidence_label"],
        phase1["status"],
        "baseline_sensors",
        phase1["baseline_sensors"],
    )
    add(
        "phase1_deterministic",
        phase1["evidence_label"],
        phase1["status"],
        "controlled_sensors",
        phase1["controlled_sensors"],
    )
    add(
        "phase1_deterministic",
        phase1["evidence_label"],
        phase1["status"],
        "control_actions",
        phase1["control_actions"],
    )

    for section_name in ("live_scripted_provider", "real_ollama"):
        section = summary[section_name]
        for metric, unit in (
            ("facility_electricity_kwh", "kWh"),
            ("absolute_energy_reduction_kwh", "kWh"),
            ("percentage_energy_reduction", "%"),
            ("simulation_duration_hours", "hours"),
            ("timestep_count", "timesteps"),
        ):
            add(
                section_name,
                section["evidence_label"],
                section["status"],
                metric,
                section.get(metric),
                unit,
            )
        add(
            section_name,
            section["evidence_label"],
            section["status"],
            "sensors",
            section.get("sensors"),
        )
        add(
            section_name,
            section["evidence_label"],
            section["status"],
            "control_actions",
            section.get("control_actions"),
        )
    return rows


def _write_csv(path: Path, summary: Mapping[str, Any]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "section",
            "evidence_label",
            "status",
            "metric",
            "value",
            "unit",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(_csv_rows(summary))
    path.write_text(stream.getvalue(), encoding="utf-8", newline="\n")


def _fmt(value: float | None, digits: int = 3) -> str:
    return "pending" if value is None else f"{value:.{digits}f}"


def _range_text(value: Mapping[str, float] | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value['minimum']:.2f} to {value['maximum']:.2f}"


def _markdown(summary: Mapping[str, Any]) -> str:
    phase1 = summary["phase1_deterministic"]
    scripted = summary["live_scripted_provider"]
    ollama = summary["real_ollama"]
    phase1_actions = phase1["control_actions"]
    scripted_actions = scripted["control_actions"]
    return f"""# Current Results

This page separates deterministic Phase 1 evidence, deterministic
ScriptedProvider live-loop evidence, and pending real-Ollama evidence. Phase 1
and ScriptedProvider results are **not real-LLM-generated savings**.

## Phase 1 deterministic seven-day comparison

- Baseline electricity: **{phase1['baseline_electricity_kwh']:.3f} kWh**
- Controlled electricity: **{phase1['controlled_electricity_kwh']:.3f} kWh**
- Deterministic reduction: **{phase1['absolute_energy_reduction_kwh']:.3f} kWh ({phase1['percentage_energy_reduction']:.3f}%)**
- Duration: **{phase1['simulation_duration_hours']:.1f} hours**, **{phase1['timestep_count']} timesteps**
- Baseline zone temperature range: **{_range_text(phase1['baseline_sensors']['zone_temperature_c_range'])} °C**
- Controlled zone temperature range: **{_range_text(phase1['controlled_sensors']['zone_temperature_c_range'])} °C**
- Baseline PMV range: **{_range_text(phase1['baseline_sensors']['pmv_range'])}**
- Controlled PMV range: **{_range_text(phase1['controlled_sensors']['pmv_range'])}**
- Occupied PMV comfort violations: baseline **{phase1['baseline_sensors']['comfort_violation_count']}**, controlled **{phase1['controlled_sensors']['comfort_violation_count']}**
- Deterministic actions accepted/rejected: **{phase1_actions['actions_accepted']}/{phase1_actions['actions_rejected']}**
- Successful zone actuator writebacks: **{phase1_actions['actuator_write_success_count']}**

![Deterministic Phase 1 energy comparison](../runs/final/energy_comparison.png)

## Live ScriptedProvider smoke

- Evidence type: **deterministic ScriptedProvider, not a real LLM**
- Facility electricity during smoke: **{scripted['facility_electricity_kwh']:.3f} kWh**
- Energy savings: **not calculated** because there is no matched four-hour baseline
- Duration: **{scripted['simulation_duration_hours']:.1f} hours**, **{scripted['timestep_count']} timesteps**
- Zone temperature range: **{_range_text(scripted['sensors']['zone_temperature_c_range'])} °C**
- PMV range: **{_range_text(scripted['sensors']['pmv_range'])}**
- Occupied PMV comfort violations: **{scripted['sensors']['comfort_violation_count']}**
- Actions accepted/rejected: **{scripted_actions['actions_accepted']}/{scripted_actions['actions_rejected']}**
- Corrections/fallbacks: **{scripted_actions['correction_count']}/{scripted_actions['fallback_count']}**
- MCP tools: **{', '.join(scripted_actions['mcp_tools_called'])}**
- MCP tool calls: **{scripted_actions['mcp_tool_call_count']}**
- Successful zone actuator writebacks: **{scripted_actions['actuator_write_success_count']}**

![Zone temperature and PMV ranges](../runs/final/zone_temperature_pmv.png)

![Action and setpoint evidence](../runs/final/action_setpoint_evidence.png)

## Real Ollama evidence

- Status: **{ollama['status']}**
- Result: {ollama.get('pending_reason', 'completed evidence available')}

No real-Ollama energy or control claim is made while this section is pending.
"""


def _save_figure(figure: Any, path: Path) -> None:
    figure.savefig(
        path,
        dpi=100,
        facecolor="white",
        metadata={"Software": "Eco-Loop evidence aggregation"},
    )
    plt.close(figure)


def _chart_energy(summary: Mapping[str, Any], path: Path) -> None:
    phase1 = summary["phase1_deterministic"]
    values = [
        phase1["baseline_electricity_kwh"],
        phase1["controlled_electricity_kwh"],
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        ["Baseline", "Deterministic control"],
        values,
        color=["#667085", "#12B76A"],
        width=0.58,
    )
    axis.bar_label(bars, fmt="%.2f kWh", padding=4)
    axis.set_ylabel("Facility electricity (kWh)")
    axis.set_title("Phase 1 deterministic seven-day comparison (not LLM)")
    axis.set_ylim(0, max(values) * 1.18)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _chart_ranges(summary: Mapping[str, Any], path: Path) -> None:
    phase1 = summary["phase1_deterministic"]["controlled_sensors"]
    scripted = summary["live_scripted_provider"]["sensors"]
    zone_ids = sorted(
        set(phase1["by_zone"]) | set(scripted["by_zone"])
    )
    figure, (temperature_axis, pmv_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
    )
    y_positions = list(range(len(zone_ids)))
    for offset, (label, sensors, color) in enumerate(
        (
            ("Phase 1 deterministic", phase1, "#12B76A"),
            ("Live ScriptedProvider", scripted, "#2E90FA"),
        )
    ):
        shift = (offset - 0.5) * 0.22
        for index, zone_id in enumerate(zone_ids):
            zone = sensors["by_zone"].get(zone_id)
            if zone is None:
                continue
            temperature = zone["zone_temperature_c_range"]
            pmv = zone["pmv_range"]
            temperature_axis.hlines(
                index + shift,
                temperature["minimum"],
                temperature["maximum"],
                color=color,
                linewidth=5,
                label=label if index == 0 else None,
            )
            pmv_axis.hlines(
                index + shift,
                pmv["minimum"],
                pmv["maximum"],
                color=color,
                linewidth=5,
                label=label if index == 0 else None,
            )
    temperature_axis.set_yticks(y_positions, zone_ids)
    temperature_axis.set_xlabel("Zone temperature range (°C)")
    pmv_axis.set_xlabel("Fanger PMV range")
    temperature_axis.set_title("Temperature")
    pmv_axis.set_title("PMV")
    for axis in (temperature_axis, pmv_axis):
        axis.grid(axis="x", alpha=0.25)
        axis.legend(loc="best")
    figure.suptitle("Measured ranges by zone; evidence classes kept separate")
    figure.tight_layout()
    _save_figure(figure, path)


def _mean_setpoint_changes(
    changes: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    previous = list(changes.values())
    if not previous:
        return [0.0, 0.0], [0.0, 0.0]
    before = [
        sum(item["previous"]["heating_c"] for item in previous)
        / len(previous),
        sum(item["previous"]["cooling_c"] for item in previous)
        / len(previous),
    ]
    after = [
        sum(item["updated"]["heating_c"] for item in previous)
        / len(previous),
        sum(item["updated"]["cooling_c"] for item in previous)
        / len(previous),
    ]
    return before, after


def _chart_actions(summary: Mapping[str, Any], path: Path) -> None:
    phase1 = summary["phase1_deterministic"]["control_actions"]
    scripted_section = summary["live_scripted_provider"]
    scripted = scripted_section["control_actions"]
    figure, (action_axis, setpoint_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )
    labels = ["Phase 1 deterministic", "Live ScriptedProvider"]
    accepted = [phase1["actions_accepted"], scripted["actions_accepted"]]
    rejected = [phase1["actions_rejected"], scripted["actions_rejected"]]
    fallback = [phase1["fallback_count"], scripted["fallback_count"]]
    action_axis.bar(labels, accepted, label="Accepted", color="#12B76A")
    action_axis.bar(
        labels,
        rejected,
        bottom=accepted,
        label="Rejected",
        color="#F04438",
    )
    action_axis.bar(
        labels,
        fallback,
        bottom=[a + r for a, r in zip(accepted, rejected, strict=True)],
        label="Fallback",
        color="#FDB022",
    )
    action_axis.set_ylabel("Decision count")
    action_axis.set_title("Actions by evidence class")
    action_axis.tick_params(axis="x", rotation=12)
    action_axis.legend()
    action_axis.grid(axis="y", alpha=0.25)

    before, after = _mean_setpoint_changes(
        scripted_section["setpoint_changes"]
    )
    x_positions = [0, 1]
    width = 0.34
    setpoint_axis.bar(
        [value - width / 2 for value in x_positions],
        before,
        width,
        label="Previous",
        color="#98A2B3",
    )
    setpoint_axis.bar(
        [value + width / 2 for value in x_positions],
        after,
        width,
        label="Updated",
        color="#2E90FA",
    )
    setpoint_axis.set_xticks(x_positions, ["Heating", "Cooling"])
    setpoint_axis.set_ylabel("Mean setpoint (°C)")
    setpoint_axis.set_title("Live ScriptedProvider accepted setpoints")
    setpoint_axis.legend()
    setpoint_axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def write_results(
    summary: Mapping[str, Any],
    *,
    output_directory: Path,
    docs_path: Path,
) -> tuple[Path, ...]:
    """Write deterministic machine-readable, narrative, and chart outputs."""

    output_directory.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "results_summary.json"
    csv_path = output_directory / "results_summary.csv"
    energy_chart = output_directory / "energy_comparison.png"
    range_chart = output_directory / "zone_temperature_pmv.png"
    action_chart = output_directory / "action_setpoint_evidence.png"
    json_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(csv_path, summary)
    docs_path.write_text(
        _markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    _chart_energy(summary, energy_chart)
    _chart_ranges(summary, range_chart)
    _chart_actions(summary, action_chart)
    return (
        json_path,
        csv_path,
        docs_path,
        energy_chart,
        range_chart,
        action_chart,
    )


def generate_results_summary(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    phase1_report_path: Path | None = None,
    scripted_report_path: Path | None = None,
    ollama_report_path: Path | None = None,
    output_directory: Path | None = None,
    docs_path: Path | None = None,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Aggregate existing artifacts and write every requested result file."""

    root = repository_root.resolve()
    summary = aggregate_results(
        root,
        phase1_report_path=phase1_report_path,
        scripted_report_path=scripted_report_path,
        ollama_report_path=ollama_report_path,
    )
    outputs = write_results(
        summary,
        output_directory=(
            output_directory
            if output_directory is not None
            else root / DEFAULT_OUTPUT_DIRECTORY
        ),
        docs_path=(
            docs_path
            if docs_path is not None
            else root / DEFAULT_DOCS_PATH
        ),
    )
    return summary, outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository containing existing runs/phase1 artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary, outputs = generate_results_summary(args.repository_root)
    print(
        json.dumps(
            {
                "status": "written",
                "phase1_energy_reduction_kwh": summary[
                    "phase1_deterministic"
                ]["absolute_energy_reduction_kwh"],
                "phase1_energy_reduction_percent": summary[
                    "phase1_deterministic"
                ]["percentage_energy_reduction"],
                "scripted_status": summary["live_scripted_provider"][
                    "status"
                ],
                "real_ollama_status": summary["real_ollama"]["status"],
                "outputs": [str(path) for path in outputs],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMFORT_PMV_LIMIT",
    "SCHEMA_VERSION",
    "aggregate_results",
    "calculate_energy_reduction",
    "generate_results_summary",
    "write_results",
]
