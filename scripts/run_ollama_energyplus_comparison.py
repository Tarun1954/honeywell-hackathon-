"""Run a matched baseline versus real Ollama-supervised EnergyPlus comparison."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from src.energyplus_wrapper import EnergyPlusWrapper, Phase1Config, RunResult
from src.live_energyplus_integration import LiveScriptedAgentController
from src.ollama_provider import OllamaToolProvider


COMPARISON_LABEL = "24-hour real Ollama-supervised hybrid comparison"
DEFAULT_HOURS = 24
FALLBACK_HOURS = 8
PMV_COMPLIANCE_LIMIT = 0.7
FINAL_DIRECTORY = Path("runs/final")
MARKDOWN_PATH = Path("docs/ollama_24h_comparison.md")
FILE_STEM = "ollama_24h_comparison"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-config",
        default="config/phase1.yaml",
        help="Phase 1 EnergyPlus configuration",
    )
    parser.add_argument(
        "--phase2-config",
        default="config/phase2.yaml",
        help="Phase 2 Ollama configuration",
    )
    parser.add_argument(
        "--hours",
        type=int,
        choices=(DEFAULT_HOURS, FALLBACK_HOURS),
        default=DEFAULT_HOURS,
        help="Matched comparison duration; use 8 only for the documented fallback",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record is not an object: {path}")
            records.append(payload)
    return records


def _range(values: Iterable[float]) -> dict[str, float] | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return {"minimum": min(collected), "maximum": max(collected)}


def _sensor_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    temperatures: list[float] = []
    pmvs: list[float] = []
    occupied_samples = 0
    violations = 0
    for record in records:
        snapshot = record["snapshot"]
        for zone in snapshot["zones"].values():
            temperature = float(zone["air_temperature_c"])
            pmv = float(zone["fanger_pmv"])
            occupancy = float(zone["occupant_count"])
            temperatures.append(temperature)
            pmvs.append(pmv)
            if occupancy > 0.0:
                occupied_samples += 1
                violations += abs(pmv) > PMV_COMPLIANCE_LIMIT
    compliant_samples = occupied_samples - violations
    compliance_percent = (
        100.0 * compliant_samples / occupied_samples
        if occupied_samples
        else None
    )
    return {
        "zone_temperature_c_range": _range(temperatures),
        "pmv_range": _range(pmvs),
        "pmv_compliance_limit": PMV_COMPLIANCE_LIMIT,
        "occupied_zone_samples": occupied_samples,
        "occupied_comfort_violation_count": violations,
        "occupied_pmv_compliant_count": compliant_samples,
        "occupied_pmv_compliance_percent": compliance_percent,
    }


def _peak_demand_w(
    records: Sequence[Mapping[str, Any]],
) -> float:
    return max(
        float(record["snapshot"]["facility_electricity_demand_w"])
        for record in records
    )


def _action_metrics(
    events: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decisions = [
        event
        for event in events
        if int(event["sequence"]) % 4 == 0
    ]
    real_accepted = sum(
        event.get("action_status") == "accepted"
        and not bool(event.get("fallback_used", False))
        for event in decisions
    )
    fallbacks = sum(
        bool(event.get("fallback_used", False))
        for event in decisions
    )
    rejected = sum(
        int(event.get("rejected_action_count", 0))
        for event in decisions
    )
    corrected = sum(
        int(event.get("corrected_action_count", 0))
        for event in decisions
    )
    successful_zone_writes = 0
    successful_timestep_writes = 0
    for event in events:
        zones = event.get("actuator_write_result", {}).get("zones", {})
        successful = sum(
            zone.get("status") == "applied"
            for zone in zones.values()
        )
        successful_zone_writes += successful
        successful_timestep_writes += bool(zones) and successful == len(zones)
    latencies = [
        float(item["latency_seconds"])
        for item in diagnostics
        if isinstance(item.get("latency_seconds"), (int, float))
    ]
    return {
        "hourly_decision_count": len(decisions),
        "real_actions_accepted": real_accepted,
        "actions_rejected": rejected,
        "corrected_action_proposals": corrected,
        "fallback_count": fallbacks,
        "successful_actuator_zone_writes": successful_zone_writes,
        "successful_actuator_timestep_writes": successful_timestep_writes,
        "llm_request_count": len(latencies),
        "median_llm_latency_seconds": (
            statistics.median(latencies) if latencies else None
        ),
        "maximum_llm_latency_seconds": max(latencies) if latencies else None,
    }


def build_comparison_report(
    *,
    hours: int,
    timestep_minutes: int,
    baseline_result: Mapping[str, Any],
    controlled_result: Mapping[str, Any],
    baseline_records: Sequence[Mapping[str, Any]],
    controlled_records: Sequence[Mapping[str, Any]],
    live_events: Sequence[Mapping[str, Any]],
    provider_diagnostics: Sequence[Mapping[str, Any]],
    weather_path: str,
    run_period_start: Sequence[int],
    run_period_end: Sequence[int],
) -> dict[str, Any]:
    """Build the matched report from completed artifacts."""

    expected_timesteps = hours * 60 // timestep_minutes
    if int(baseline_result["timestep_count"]) != expected_timesteps:
        raise ValueError("baseline timestep count does not match duration")
    if int(controlled_result["timestep_count"]) != expected_timesteps:
        raise ValueError("controlled timestep count does not match duration")
    if baseline_result["model_path"] != controlled_result["model_path"]:
        raise ValueError("comparison model paths do not match")
    initial_match = (
        baseline_result.get("first_snapshot")
        == controlled_result.get("first_snapshot")
    )
    if not initial_match:
        raise ValueError("comparison initial snapshots do not match")

    baseline_kwh = float(baseline_result["facility_electricity_kwh"])
    controlled_kwh = float(controlled_result["facility_electricity_kwh"])
    signed_difference = controlled_kwh - baseline_kwh
    percentage_difference = (
        100.0 * signed_difference / baseline_kwh
        if baseline_kwh
        else None
    )
    label = (
        COMPARISON_LABEL
        if hours == DEFAULT_HOURS
        else "8-hour real Ollama-supervised hybrid comparison"
    )
    return {
        "comparison_label": label,
        "status": "completed",
        "duration_hours": hours,
        "timestep_minutes": timestep_minutes,
        "timestep_count": expected_timesteps,
        "provider": "OllamaToolProvider",
        "model": "llama3.2:3b",
        "evidence_class": "real_ollama_supervised_hybrid",
        "matched_conditions": {
            "same_model": True,
            "model_path": str(baseline_result["model_path"]),
            "same_weather": True,
            "weather_path": weather_path,
            "same_run_period_start": list(run_period_start),
            "configured_run_period_end": list(run_period_end),
            "same_timestep": True,
            "same_initial_snapshot": initial_match,
            "baseline_control": "original EnergyPlus control; no LLM actions",
            "controlled_control": (
                "hourly real Ollama supervision with four-timestep hold, "
                "MCP safety validation, and deterministic fallback"
            ),
        },
        "electricity": {
            "baseline_total_kwh": baseline_kwh,
            "controlled_total_kwh": controlled_kwh,
            "controlled_minus_baseline_kwh": signed_difference,
            "absolute_difference_kwh": abs(signed_difference),
            "percentage_difference_from_baseline": percentage_difference,
            "difference_sign_convention": (
                "negative means controlled used less electricity"
            ),
        },
        "peak_demand": {
            "baseline_w": _peak_demand_w(baseline_records),
            "controlled_w": _peak_demand_w(controlled_records),
        },
        "baseline_comfort": _sensor_metrics(baseline_records),
        "controlled_comfort": _sensor_metrics(controlled_records),
        "control_actions": _action_metrics(
            live_events,
            provider_diagnostics,
        ),
        "energyplus": {
            "baseline": {
                "exit_status": int(baseline_result["exit_code"]),
                "severe_count": int(baseline_result["severe_count"]),
                "fatal_count": int(baseline_result["fatal_count"]),
                "output_directory": str(
                    baseline_result["output_directory"]
                ),
            },
            "controlled": {
                "exit_status": int(controlled_result["exit_code"]),
                "severe_count": int(controlled_result["severe_count"]),
                "fatal_count": int(controlled_result["fatal_count"]),
                "output_directory": str(
                    controlled_result["output_directory"]
                ),
            },
        },
        "separate_existing_result": {
            "label": "Phase 1 deterministic seven-day comparison",
            "energy_reduction_percent": 6.7473173222510106,
            "included_in_this_comparison": False,
            "note": (
                "The seven-day 6.747% result is deterministic Phase 1 "
                "evidence and is not a real-LLM result."
            ),
        },
    }


def _csv_text(report: Mapping[str, Any]) -> str:
    rows: list[tuple[str, str, Any, str]] = []

    def add(section: str, metric: str, value: Any, unit: str = "") -> None:
        rows.append((section, metric, value, unit))

    electricity = report["electricity"]
    for metric, unit in (
        ("baseline_total_kwh", "kWh"),
        ("controlled_total_kwh", "kWh"),
        ("controlled_minus_baseline_kwh", "kWh"),
        ("absolute_difference_kwh", "kWh"),
        ("percentage_difference_from_baseline", "%"),
    ):
        add("electricity", metric, electricity[metric], unit)
    for case in ("baseline", "controlled"):
        comfort = report[f"{case}_comfort"]
        add(case, "peak_demand_w", report["peak_demand"][f"{case}_w"], "W")
        add(
            case,
            "zone_temperature_c_range",
            comfort["zone_temperature_c_range"],
            "degC",
        )
        add(case, "pmv_range", comfort["pmv_range"], "PMV")
        add(
            case,
            "occupied_pmv_compliance_percent",
            comfort["occupied_pmv_compliance_percent"],
            "%",
        )
        add(
            case,
            "occupied_comfort_violation_count",
            comfort["occupied_comfort_violation_count"],
            "zone-samples",
        )
    for metric, value in report["control_actions"].items():
        add(
            "control_actions",
            metric,
            value,
            "seconds" if "latency_seconds" in metric else "",
        )
    for case in ("baseline", "controlled"):
        for metric in ("exit_status", "severe_count", "fatal_count"):
            add("energyplus_" + case, metric, report["energyplus"][case][metric])

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("section", "metric", "value", "unit"))
    for section, metric, value, unit in rows:
        rendered = (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else "" if value is None else str(value)
        )
        writer.writerow((section, metric, rendered, unit))
    return stream.getvalue()


def _range_text(value: Mapping[str, float]) -> str:
    return f"{value['minimum']:.2f} to {value['maximum']:.2f}"


def _markdown_text(report: Mapping[str, Any]) -> str:
    electricity = report["electricity"]
    baseline = report["baseline_comfort"]
    controlled = report["controlled_comfort"]
    actions = report["control_actions"]
    peak = report["peak_demand"]
    energyplus = report["energyplus"]
    signed = electricity["percentage_difference_from_baseline"]
    return f"""# {report['comparison_label']}

This is a matched real-provider comparison using the same EnergyPlus model,
weather start, 15-minute timestep, and initial snapshot for both cases.

## Electricity and demand

- Baseline total electricity: **{electricity['baseline_total_kwh']:.3f} kWh**
- Controlled total electricity: **{electricity['controlled_total_kwh']:.3f} kWh**
- Controlled minus baseline: **{electricity['controlled_minus_baseline_kwh']:.3f} kWh**
- Absolute difference: **{electricity['absolute_difference_kwh']:.3f} kWh**
- Difference from baseline: **{signed:.3f}%** (negative means lower controlled use)
- Baseline/controlled peak demand: **{peak['baseline_w'] / 1000:.3f}/{peak['controlled_w'] / 1000:.3f} kW**

![Matched electricity comparison](../runs/final/ollama_24h_energy_peak.png)

## Comfort

- Baseline temperature range: **{_range_text(baseline['zone_temperature_c_range'])} °C**
- Controlled temperature range: **{_range_text(controlled['zone_temperature_c_range'])} °C**
- Baseline PMV range: **{_range_text(baseline['pmv_range'])}**
- Controlled PMV range: **{_range_text(controlled['pmv_range'])}**
- Occupied PMV compliance: baseline **{baseline['occupied_pmv_compliance_percent']:.2f}%**, controlled **{controlled['occupied_pmv_compliance_percent']:.2f}%**
- Occupied comfort violations: baseline **{baseline['occupied_comfort_violation_count']}**, controlled **{controlled['occupied_comfort_violation_count']}**

![Matched comfort comparison](../runs/final/ollama_24h_comfort.png)

## Ollama supervision and safety

- Hourly decisions: **{actions['hourly_decision_count']}**
- Real actions accepted: **{actions['real_actions_accepted']}**
- Actions rejected: **{actions['actions_rejected']}**
- Corrected action proposals: **{actions['corrected_action_proposals']}**
- Deterministic fallbacks: **{actions['fallback_count']}**
- Successful zone actuator writes: **{actions['successful_actuator_zone_writes']}**
- Median/maximum LLM latency: **{actions['median_llm_latency_seconds']:.3f}/{actions['maximum_llm_latency_seconds']:.3f} seconds**

![Action and latency evidence](../runs/final/ollama_24h_actions_latency.png)

## EnergyPlus status

- Baseline exit/severe/fatal: **{energyplus['baseline']['exit_status']}/{energyplus['baseline']['severe_count']}/{energyplus['baseline']['fatal_count']}**
- Controlled exit/severe/fatal: **{energyplus['controlled']['exit_status']}/{energyplus['controlled']['severe_count']}/{energyplus['controlled']['fatal_count']}**

## Separate deterministic evidence

The existing **seven-day deterministic Phase 1 reduction is 6.747%**. It is
kept separate and is not described as real-LLM-generated savings.
"""


def _energy_chart(report: Mapping[str, Any], path: Path) -> None:
    electricity = report["electricity"]
    peak = report["peak_demand"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = ("Baseline", "Ollama hybrid")
    axes[0].bar(
        labels,
        (
            electricity["baseline_total_kwh"],
            electricity["controlled_total_kwh"],
        ),
        color=("#6b7280", "#2563eb"),
    )
    axes[0].set_ylabel("Electricity (kWh)")
    axes[0].set_title("Matched total electricity")
    axes[1].bar(
        labels,
        (peak["baseline_w"] / 1000, peak["controlled_w"] / 1000),
        color=("#6b7280", "#2563eb"),
    )
    axes[1].set_ylabel("Peak demand (kW)")
    axes[1].set_title("Matched peak demand")
    figure.suptitle(report["comparison_label"])
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _comfort_chart(report: Mapping[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = ("Baseline", "Ollama hybrid")
    for axis, key, title, ylabel in (
        (axes[0], "zone_temperature_c_range", "Zone temperature", "°C"),
        (axes[1], "pmv_range", "Fanger PMV", "PMV"),
    ):
        ranges = [
            report[f"{case}_comfort"][key]
            for case in ("baseline", "controlled")
        ]
        centers = [
            (item["minimum"] + item["maximum"]) / 2
            for item in ranges
        ]
        errors = [
            (item["maximum"] - item["minimum"]) / 2
            for item in ranges
        ]
        axis.errorbar(
            labels,
            centers,
            yerr=errors,
            fmt="o",
            capsize=8,
            color="#2563eb",
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Matched 24-hour comfort ranges")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _action_chart(report: Mapping[str, Any], path: Path) -> None:
    actions = report["control_actions"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    action_labels = ("Accepted", "Rejected", "Corrected", "Fallback")
    axes[0].bar(
        action_labels,
        (
            actions["real_actions_accepted"],
            actions["actions_rejected"],
            actions["corrected_action_proposals"],
            actions["fallback_count"],
        ),
        color=("#16a34a", "#dc2626", "#f59e0b", "#6b7280"),
    )
    axes[0].set_ylabel("Hourly decisions")
    axes[0].set_title("Control outcomes")
    axes[1].bar(
        ("Median", "Maximum"),
        (
            actions["median_llm_latency_seconds"],
            actions["maximum_llm_latency_seconds"],
        ),
        color=("#60a5fa", "#1d4ed8"),
    )
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Ollama request latency")
    figure.suptitle("Real Ollama action and latency evidence")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_comparison_outputs(
    report: Mapping[str, Any],
    *,
    repository_root: Path,
    final_directory: Path | None = None,
    markdown_path: Path | None = None,
) -> tuple[Path, ...]:
    """Write JSON, CSV, Markdown, and PNG evidence."""

    root = repository_root.resolve()
    output = (
        final_directory
        if final_directory is not None
        else root / FINAL_DIRECTORY
    )
    docs = (
        markdown_path
        if markdown_path is not None
        else root / MARKDOWN_PATH
    )
    output.mkdir(parents=True, exist_ok=True)
    docs.parent.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{FILE_STEM}.json"
    csv_path = output / f"{FILE_STEM}.csv"
    energy_path = output / "ollama_24h_energy_peak.png"
    comfort_path = output / "ollama_24h_comfort.png"
    action_path = output / "ollama_24h_actions_latency.png"
    json_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    csv_path.write_text(
        _csv_text(report),
        encoding="utf-8",
        newline="\n",
    )
    docs.write_text(
        _markdown_text(report),
        encoding="utf-8",
        newline="\n",
    )
    _energy_chart(report, energy_path)
    _comfort_chart(report, comfort_path)
    _action_chart(report, action_path)
    return (
        json_path,
        csv_path,
        docs,
        energy_path,
        comfort_path,
        action_path,
    )


def run_comparison(
    phase1_config_path: str | Path = "config/phase1.yaml",
    phase2_config_path: str | Path = "config/phase2.yaml",
    *,
    hours: int = DEFAULT_HOURS,
) -> dict[str, Any]:
    """Run both matched EnergyPlus cases and write comparison evidence."""

    if hours not in {DEFAULT_HOURS, FALLBACK_HOURS}:
        raise ValueError("comparison duration must be 24 or 8 hours")
    config = Phase1Config.load(phase1_config_path)
    timesteps = hours * 60 // config.timestep_minutes
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    diagnostics: list[dict[str, Any]] = []

    def emit_diagnostic(payload: Mapping[str, Any]) -> None:
        diagnostic = dict(payload)
        diagnostics.append(diagnostic)
        print(
            "Ollama comparison diagnostic: "
            + json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    provider = OllamaToolProvider.from_phase2_config(
        phase2_config_path,
        scenario_directive=(
            "Use the current matched-comparison EnergyPlus snapshot. Choose "
            "conservative safe setpoints, hold for four timesteps, and rely "
            "on deterministic release fallback when uncertain."
        ),
        diagnostic_sink=emit_diagnostic,
    )
    wrapper = EnergyPlusWrapper(config)
    baseline = wrapper.run(
        run_id=f"ollama-{hours}h-baseline-{suffix}",
        mode="baseline-no-llm",
        policy=None,
        model_path=config.baseline_model,
        max_timesteps=timesteps,
    )
    artifact_directory = (
        config.output_root
        / f"ollama-{hours}h-comparison-artifacts-{suffix}"
    )
    controller = LiveScriptedAgentController(
        repository_root=config.repository_root,
        work_directory=artifact_directory,
        controlled_zones=config.controlled_zones,
        safety_limits=config.safety,
        provider_factory=lambda snapshot: provider,
    )
    controlled = wrapper.run(
        run_id=f"ollama-{hours}h-controlled-{suffix}",
        mode="live-ollama-supervised",
        policy=controller,
        model_path=config.baseline_model,
        max_timesteps=timesteps,
    )
    baseline_records = _read_jsonl(
        Path(baseline.output_directory) / config.timestep_log_name
    )
    controlled_records = _read_jsonl(
        Path(controlled.output_directory) / config.timestep_log_name
    )
    report = build_comparison_report(
        hours=hours,
        timestep_minutes=config.timestep_minutes,
        baseline_result=baseline.to_dict(),
        controlled_result=controlled.to_dict(),
        baseline_records=baseline_records,
        controlled_records=controlled_records,
        live_events=controller.events,
        provider_diagnostics=diagnostics,
        weather_path=str(config.weather_file),
        run_period_start=config.run_period_start,
        run_period_end=config.run_period_end,
    )
    report["source_artifacts"] = {
        "baseline_timesteps": str(
            Path(baseline.output_directory) / config.timestep_log_name
        ),
        "controlled_timesteps": str(
            Path(controlled.output_directory) / config.timestep_log_name
        ),
        "live_agent_timesteps": str(controller.log_path),
    }
    if (
        baseline.exit_code
        or baseline.severe_count
        or baseline.fatal_count
        or controlled.exit_code
        or controlled.severe_count
        or controlled.fatal_count
    ):
        raise RuntimeError("matched EnergyPlus comparison failed acceptance gates")
    outputs = write_comparison_outputs(
        report,
        repository_root=config.repository_root,
    )
    print(
        json.dumps(
            {
                "event": "ollama_energyplus_comparison_complete",
                "comparison_label": report["comparison_label"],
                "outputs": [str(path) for path in outputs],
                "electricity": report["electricity"],
                "control_actions": report["control_actions"],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> int:
    args = _parse_args()
    try:
        run_comparison(
            args.phase1_config,
            args.phase2_config,
            hours=args.hours,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "ollama_energyplus_comparison_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_LABEL",
    "build_comparison_report",
    "run_comparison",
    "write_comparison_outputs",
]
