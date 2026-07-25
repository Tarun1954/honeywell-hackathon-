"""Run baseline and deterministic-actuation EnergyPlus Phase 1 evidence cases."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.energyplus_wrapper import (
    ControlAction,
    EnergyPlusWrapper,
    Phase1Config,
    RunResult,
    SensorSnapshot,
    ZoneSetpoints,
)


LOGGER = logging.getLogger("eco_loop.phase1.runner")


@dataclass(frozen=True)
class DiagnosticSetpointPolicy:
    """Deterministic, occupancy-gated policy used only to prove live writeback."""

    zones: tuple[str, ...]
    heating_setpoint_c: float
    cooling_setpoint_c: float

    def __call__(self, snapshot: SensorSnapshot) -> ControlAction | None:
        """Relax occupied setpoints and reset control when the building is empty."""

        occupied = any(
            snapshot.zones[zone].occupant_count > 0.0 for zone in self.zones
        )
        if not occupied:
            return None
        return ControlAction(
            setpoints={
                zone: ZoneSetpoints(
                    heating_c=self.heating_setpoint_c,
                    cooling_c=self.cooling_setpoint_c,
                )
                for zone in self.zones
            },
            source="phase1_deterministic_diagnostic",
            reason="Occupied setpoint step proving EnergyPlus EMS/API writeback",
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load the compact Phase 1 timestep evidence log."""

    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _proof_metrics(
    baseline: RunResult,
    actuated: RunResult,
    limits_tolerance_c: float,
) -> dict[str, Any]:
    """Calculate setpoint writeback and downstream physical-response evidence."""

    baseline_records = _load_jsonl(
        Path(baseline.output_directory) / "timesteps.jsonl"
    )
    actuated_records = _load_jsonl(
        Path(actuated.output_directory) / "timesteps.jsonl"
    )
    if len(baseline_records) != len(actuated_records):
        raise RuntimeError(
            "Baseline and actuated runs have different timestep counts: "
            f"{len(baseline_records)} != {len(actuated_records)}"
        )

    applied_steps = 0
    setpoint_samples = 0
    maximum_setpoint_error_c = 0.0
    maximum_zone_temperature_delta_c = 0.0
    occupied_pmv_samples = 0
    occupied_pmv_within_07 = 0
    occupied_max_co2_ppm = 0.0

    for baseline_record, actuated_record in zip(
        baseline_records,
        actuated_records,
        strict=True,
    ):
        if (
            baseline_record["snapshot"]["sequence"]
            != actuated_record["snapshot"]["sequence"]
        ):
            raise RuntimeError("Baseline and actuated timestep sequences do not align")

        action = actuated_record["action_applied"]
        if action["status"] == "applied":
            applied_steps += 1
            for zone, applied in action["zones"].items():
                if applied["status"] != "applied":
                    continue
                reported = actuated_record["snapshot"]["zones"][zone]
                maximum_setpoint_error_c = max(
                    maximum_setpoint_error_c,
                    abs(
                        float(reported["heating_setpoint_c"])
                        - float(applied["heating_c"])
                    ),
                    abs(
                        float(reported["cooling_setpoint_c"])
                        - float(applied["cooling_c"])
                    ),
                )
                setpoint_samples += 2

        for zone, actuated_zone in actuated_record["snapshot"]["zones"].items():
            baseline_zone = baseline_record["snapshot"]["zones"][zone]
            maximum_zone_temperature_delta_c = max(
                maximum_zone_temperature_delta_c,
                abs(
                    float(actuated_zone["air_temperature_c"])
                    - float(baseline_zone["air_temperature_c"])
                ),
            )
            if float(actuated_zone["occupant_count"]) > 0.0:
                occupied_pmv_samples += 1
                occupied_pmv_within_07 += (
                    abs(float(actuated_zone["fanger_pmv"])) <= 0.7
                )
                occupied_max_co2_ppm = max(
                    occupied_max_co2_ppm,
                    float(actuated_zone["co2_ppm"]),
                )

    facility_delta_kwh = (
        actuated.facility_electricity_kwh - baseline.facility_electricity_kwh
    )
    physical_response_proven = (
        maximum_zone_temperature_delta_c > 0.01
        or abs(facility_delta_kwh) > 0.01
    )
    return {
        "applied_zone_timestep_count": applied_steps,
        "setpoint_comparison_sample_count": setpoint_samples,
        "maximum_reported_setpoint_error_c": maximum_setpoint_error_c,
        "writeback_within_tolerance": (
            setpoint_samples > 0
            and maximum_setpoint_error_c <= limits_tolerance_c
        ),
        "maximum_zone_temperature_delta_c": maximum_zone_temperature_delta_c,
        "facility_electricity_delta_kwh": facility_delta_kwh,
        "physical_response_proven": physical_response_proven,
        "occupied_pmv_sample_count": occupied_pmv_samples,
        "occupied_pmv_within_abs_0_7_fraction": (
            occupied_pmv_within_07 / occupied_pmv_samples
            if occupied_pmv_samples
            else None
        ),
        "occupied_max_co2_ppm": occupied_max_co2_ppm,
    }


def _comparison_report(
    baseline: RunResult | None,
    actuated: RunResult | None,
    repeatability: RunResult | None,
    config: Phase1Config,
) -> dict[str, Any]:
    """Build the final compact Phase 1 A/B evidence report."""

    report: dict[str, Any] = {
        "phase": 1,
        "energyplus_version": config.energyplus_version,
        "controlled_zones": list(config.controlled_zones),
        "timestep_minutes": config.timestep_minutes,
        "baseline": baseline.to_dict() if baseline else None,
        "actuated": actuated.to_dict() if actuated else None,
        "repeatability_run": repeatability.to_dict() if repeatability else None,
    }
    if baseline and actuated:
        baseline_kwh = baseline.facility_electricity_kwh
        report["comparison"] = {
            "facility_electricity_savings_kwh": (
                baseline_kwh - actuated.facility_electricity_kwh
            ),
            "facility_electricity_savings_percent": (
                100.0
                * (baseline_kwh - actuated.facility_electricity_kwh)
                / baseline_kwh
                if baseline_kwh
                else None
            ),
            "proof": _proof_metrics(
                baseline,
                actuated,
                config.safety.writeback_tolerance_c,
            ),
        }
    if actuated and repeatability:
        energy_delta_kwh = abs(
            actuated.facility_electricity_kwh
            - repeatability.facility_electricity_kwh
        )
        report["repeatability"] = {
            "facility_electricity_absolute_delta_kwh": energy_delta_kwh,
            "timestep_count_matches": (
                actuated.timestep_count == repeatability.timestep_count
            ),
            "deterministic_within_1e_9_kwh": energy_delta_kwh <= 1.0e-9,
            "repeat_run_zero_severe_fatal": (
                repeatability.severe_count == 0 and repeatability.fatal_count == 0
            ),
        }
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Write the final Phase 1 report as deterministic UTF-8 JSON."""

    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _report_path_for_mode(output_root: Path, mode: str) -> Path:
    """Keep partial-run reports separate from the canonical complete A/B report."""

    filenames = {
        "baseline": "phase1_report_baseline.json",
        "actuated": "phase1_report_actuated.json",
        "both": "phase1_report.json",
    }
    try:
        filename = filenames[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 1 report mode: {mode}") from exc
    return output_root / filename


def _parse_args() -> argparse.Namespace:
    """Parse Phase 1 runner arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/phase1.yaml",
        help="Path to the Phase 1 YAML configuration",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "actuated", "both"),
        default="both",
        help="Evidence case(s) to run",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested Phase 1 case(s) and emit the evidence report."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    config = Phase1Config.load(args.config)
    wrapper = EnergyPlusWrapper(config)

    baseline_result: RunResult | None = None
    actuated_result: RunResult | None = None
    repeatability_result: RunResult | None = None
    if args.mode in {"baseline", "both"}:
        baseline_result = wrapper.run(
            run_id="baseline",
            mode="baseline",
            policy=None,
            model_path=config.baseline_model,
        )
        LOGGER.info(
            json.dumps(
                {"event": "baseline_complete", "result": baseline_result.to_dict()},
                sort_keys=True,
            )
        )

    if args.mode in {"actuated", "both"}:
        config.runtime_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.baseline_model, config.runtime_model)
        policy = DiagnosticSetpointPolicy(
            zones=config.controlled_zones,
            heating_setpoint_c=config.diagnostic_heating_setpoint_c,
            cooling_setpoint_c=config.diagnostic_cooling_setpoint_c,
        )
        actuated_result = wrapper.run(
            run_id="actuated",
            mode="actuated",
            policy=policy,
            model_path=config.runtime_model,
        )
        LOGGER.info(
            json.dumps(
                {"event": "actuated_complete", "result": actuated_result.to_dict()},
                sort_keys=True,
            )
        )
        if args.mode == "both":
            repeatability_result = wrapper.run(
                run_id="actuated-repeat",
                mode="actuated-repeat",
                policy=policy,
                model_path=config.runtime_model,
            )
            LOGGER.info(
                json.dumps(
                    {
                        "event": "repeatability_complete",
                        "result": repeatability_result.to_dict(),
                    },
                    sort_keys=True,
                )
            )

    report = _comparison_report(
        baseline_result,
        actuated_result,
        repeatability_result,
        config,
    )
    report_path = _report_path_for_mode(config.output_root, args.mode)
    _write_report(report_path, report)
    LOGGER.info(
        json.dumps(
            {
                "event": "phase1_report_written",
                "mode": args.mode,
                "path": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
