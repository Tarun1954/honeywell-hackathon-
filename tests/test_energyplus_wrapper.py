"""Unit and opt-in real-engine checks for the Phase 1 EnergyPlus wrapper."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from scripts.run_phase1_smoke import _report_path_for_mode, _write_report
from src.energyplus_wrapper import (
    CallbackExecutionError,
    ControlAction,
    EnergyCrosscheckError,
    EnergyPlusWrapper,
    HandleResolutionError,
    InvalidControlAction,
    Phase1Config,
    RuntimeCleanupError,
    SafetyLimits,
    ZoneSetpoints,
    parse_energyplus_error_file,
    read_csv_facility_electricity_kwh,
    read_csv_hvac_electricity_kwh,
    read_csv_natural_gas_kwh,
    validate_control_action,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config" / "phase1.yaml"
PHASE1_EVIDENCE_ROOT = (REPOSITORY_ROOT / "runs" / "phase1").resolve()


def _resolve_report_output_directory(raw_path: str) -> Path:
    """Resolve a report's run directory without trusting a stored host path."""

    native_path = Path(raw_path)
    windows_path = PureWindowsPath(raw_path)
    posix_path = PurePosixPath(raw_path)
    if (
        native_path.is_absolute()
        or windows_path.is_absolute()
        or posix_path.is_absolute()
    ):
        if windows_path.is_absolute():
            basename = windows_path.name
        elif posix_path.is_absolute():
            basename = posix_path.name
        else:
            basename = native_path.name
        candidate = PHASE1_EVIDENCE_ROOT / basename
    else:
        candidate = REPOSITORY_ROOT / native_path

    resolved = candidate.resolve()
    try:
        resolved.relative_to(PHASE1_EVIDENCE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Report output directory escapes runs/phase1: {raw_path}"
        ) from exc
    return resolved


def _load_timestep_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Load a report result's raw timestep evidence from the current checkout."""

    output_directory = _resolve_report_output_directory(
        str(result["output_directory"])
    )
    log_path = output_directory / "timesteps.jsonl"
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ControlActionValidationTests(unittest.TestCase):
    """Exercise the pure actuator-safety boundary."""

    def setUp(self) -> None:
        self.limits = SafetyLimits(
            heating_minimum_c=16.0,
            heating_maximum_c=24.0,
            cooling_minimum_c=20.0,
            cooling_maximum_c=30.0,
            minimum_deadband_c=1.0,
            writeback_tolerance_c=0.1,
        )
        self.zones = ("SPACE1-1",)

    def test_valid_action_is_returned(self) -> None:
        action = ControlAction(
            setpoints={"SPACE1-1": ZoneSetpoints(20.0, 26.0)},
            source="test",
            reason="valid",
        )
        self.assertIs(
            validate_control_action(action, self.zones, self.limits),
            action,
        )

    def test_nan_is_rejected(self) -> None:
        action = ControlAction(
            setpoints={"SPACE1-1": ZoneSetpoints(math.nan, 26.0)},
            source="test",
            reason="invalid",
        )
        with self.assertRaisesRegex(InvalidControlAction, "finite"):
            validate_control_action(action, self.zones, self.limits)

    def test_out_of_range_value_is_rejected(self) -> None:
        action = ControlAction(
            setpoints={"SPACE1-1": ZoneSetpoints(25.0, 26.0)},
            source="test",
            reason="invalid",
        )
        with self.assertRaisesRegex(InvalidControlAction, "outside"):
            validate_control_action(action, self.zones, self.limits)

    def test_deadband_violation_is_rejected(self) -> None:
        action = ControlAction(
            setpoints={"SPACE1-1": ZoneSetpoints(22.0, 22.5)},
            source="test",
            reason="invalid",
        )
        with self.assertRaisesRegex(InvalidControlAction, "deadband"):
            validate_control_action(action, self.zones, self.limits)

    def test_unknown_zone_is_rejected(self) -> None:
        action = ControlAction(
            setpoints={"UNKNOWN": ZoneSetpoints(20.0, 26.0)},
            source="test",
            reason="invalid",
        )
        with self.assertRaisesRegex(InvalidControlAction, "unknown zones"):
            validate_control_action(action, self.zones, self.limits)


class ArtifactParsingTests(unittest.TestCase):
    """Verify deterministic error and native-energy parsing."""

    def test_absolute_report_path_is_rebased_to_current_checkout(self) -> None:
        resolved = _resolve_report_output_directory(
            r"C:\old-checkout\runs\phase1\actuated-3"
        )
        self.assertEqual(
            resolved,
            PHASE1_EVIDENCE_ROOT / "actuated-3",
        )

    def test_relative_report_path_cannot_escape_phase1_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes runs/phase1"):
            _resolve_report_output_directory("../outside")

    def test_error_file_severity_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            error_path = Path(temporary_directory) / "eplusout.err"
            error_path.write_text(
                "** Warning ** warning text\n"
                "** Severe  ** severe text\n"
                "**  Fatal  ** fatal text\n",
                encoding="utf-8",
            )
            summary = parse_energyplus_error_file(error_path)
        self.assertEqual(summary.warning_count, 1)
        self.assertEqual(summary.severe_count, 1)
        self.assertEqual(summary.fatal_count, 1)

    def test_csv_energy_columns_are_summed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "eplusout.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "Whole Building:Facility Total Purchased Electricity Energy [J](TimeStep)",
                        "Electricity:HVAC [J](TimeStep)",
                        "NaturalGas:Facility [J](TimeStep)",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Whole Building:Facility Total Purchased Electricity Energy [J](TimeStep)": 3_600_000,
                        "Electricity:HVAC [J](TimeStep)": 1_800_000,
                        "NaturalGas:Facility [J](TimeStep)": 900_000,
                    }
                )
                writer.writerow(
                    {
                        "Whole Building:Facility Total Purchased Electricity Energy [J](TimeStep)": 3_600_000,
                        "Electricity:HVAC [J](TimeStep)": 1_800_000,
                        "NaturalGas:Facility [J](TimeStep)": 900_000,
                    }
                )
            self.assertEqual(read_csv_facility_electricity_kwh(csv_path), 2.0)
            self.assertEqual(read_csv_hvac_electricity_kwh(csv_path), 1.0)
            self.assertEqual(read_csv_natural_gas_kwh(csv_path), 0.5)


class ConfigurationTests(unittest.TestCase):
    """Validate checked-in Phase 1 paths and mappings."""

    def test_project_configuration_resolves(self) -> None:
        config = Phase1Config.load(CONFIG_PATH)
        config.validate_paths()
        self.assertEqual(config.energyplus_version, "26.1.0")
        self.assertEqual(config.timestep_minutes, 15)
        self.assertEqual(len(config.controlled_zones), 5)
        self.assertEqual(set(config.controlled_zones), set(config.people_objects))


class WrapperRobustnessTests(unittest.TestCase):
    """Exercise retry classification and pre-run lifecycle cleanup."""

    def test_recoverable_crosscheck_failure_uses_fresh_attempt(self) -> None:
        config = Phase1Config.load(CONFIG_PATH)
        calls = 0
        sentinel = object()

        with tempfile.TemporaryDirectory() as temporary_directory:
            retry_config = replace(
                config,
                output_root=Path(temporary_directory),
                max_attempts=2,
                retry_delay_seconds=0.0,
            )
            wrapper = EnergyPlusWrapper(retry_config, api_factory=lambda: None)

            def fake_run_once(**_: object) -> object:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise EnergyCrosscheckError("transient crosscheck")
                return sentinel

            wrapper._run_once = fake_run_once  # type: ignore[method-assign]
            result = wrapper.run("retry-test", "test")

        self.assertIs(result, sentinel)
        self.assertEqual(calls, 2)

    def test_handle_callback_failure_is_not_retried(self) -> None:
        config = Phase1Config.load(CONFIG_PATH)
        calls = 0

        with tempfile.TemporaryDirectory() as temporary_directory:
            retry_config = replace(
                config,
                output_root=Path(temporary_directory),
                max_attempts=2,
                retry_delay_seconds=0.0,
            )
            wrapper = EnergyPlusWrapper(retry_config, api_factory=lambda: None)

            def fake_run_once(**_: object) -> object:
                nonlocal calls
                calls += 1
                try:
                    raise HandleResolutionError("missing handle")
                except HandleResolutionError as cause:
                    raise CallbackExecutionError("callback failed") from cause

            wrapper._run_once = fake_run_once  # type: ignore[method-assign]
            with self.assertRaises(CallbackExecutionError):
                wrapper.run("permanent-test", "test")

        self.assertEqual(calls, 1)

    def test_cleanup_failure_is_not_retried(self) -> None:
        config = Phase1Config.load(CONFIG_PATH)
        calls = 0

        with tempfile.TemporaryDirectory() as temporary_directory:
            retry_config = replace(
                config,
                output_root=Path(temporary_directory),
                max_attempts=2,
                retry_delay_seconds=0.0,
            )
            wrapper = EnergyPlusWrapper(retry_config, api_factory=lambda: None)

            def fake_run_once(**_: object) -> object:
                nonlocal calls
                calls += 1
                raise RuntimeCleanupError("state could not be released")

            wrapper._run_once = fake_run_once  # type: ignore[method-assign]
            with self.assertRaises(RuntimeCleanupError):
                wrapper.run("cleanup-test", "test")

        self.assertEqual(calls, 1)

    def test_setup_failure_releases_state_and_callbacks(self) -> None:
        config = Phase1Config.load(CONFIG_PATH)
        calls: list[str] = []

        class FakeStateManager:
            def new_state(self) -> object:
                calls.append("new_state")
                return object()

            def delete_state(self, _: object) -> None:
                calls.append("delete_state")

        class FakeRuntime:
            def set_console_output_status(self, _: object, __: bool) -> None:
                calls.append("set_console")
                raise OSError("setup failed")

            def clear_callbacks(self) -> None:
                calls.append("runtime_clear")

        class FakeFunctional:
            def clear_callbacks(self) -> None:
                calls.append("functional_clear")

        class FakeApi:
            state_manager = FakeStateManager()
            runtime = FakeRuntime()
            functional = FakeFunctional()

            def verify_api_version_match(self, _: object) -> None:
                calls.append("verify")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            wrapper = EnergyPlusWrapper(config, api_factory=FakeApi)
            with self.assertRaisesRegex(OSError, "setup failed"):
                wrapper._run_once(
                    run_id="setup-failure",
                    mode="test",
                    policy=None,
                    model_path=config.baseline_model,
                    output_directory=output_directory,
                    attempt=1,
                )
            failure_summary = json.loads(
                (output_directory / config.summary_name).read_text(encoding="utf-8")
            )
            self.assertEqual(failure_summary["status"], "failed")
            self.assertEqual(failure_summary["stage"], "api_setup")

        self.assertEqual(
            calls,
            [
                "new_state",
                "verify",
                "set_console",
                "runtime_clear",
                "functional_clear",
                "delete_state",
            ],
        )

    def test_runtime_and_cleanup_failure_is_not_retried(self) -> None:
        config = Phase1Config.load(CONFIG_PATH)
        factory_calls = 0

        class FakeExchange:
            def reset_api_error_flag(self, _: object) -> None:
                return None

            def request_variable(self, _: object, __: str, ___: str) -> None:
                return None

            def api_error_flag(self, _: object) -> bool:
                return False

        class FakeStateManager:
            def new_state(self) -> object:
                return object()

            def delete_state(self, _: object) -> None:
                raise OSError("delete failed")

        class FakeRuntime:
            def set_console_output_status(self, _: object, __: bool) -> None:
                return None

            def callback_begin_system_timestep_before_predictor(
                self,
                _: object,
                __: object,
            ) -> None:
                return None

            def callback_end_system_timestep_after_hvac_reporting(
                self,
                _: object,
                __: object,
            ) -> None:
                return None

            def callback_end_zone_timestep_after_zone_reporting(
                self,
                _: object,
                __: object,
            ) -> None:
                return None

            def run_energyplus(self, _: object, __: list[str]) -> int:
                raise OSError("runtime failed")

            def clear_callbacks(self) -> None:
                raise OSError("callback clear failed")

        class FakeFunctional:
            def callback_error(self, _: object, __: object) -> None:
                return None

            def clear_callbacks(self) -> None:
                return None

        class FakeApi:
            exchange = FakeExchange()
            state_manager = FakeStateManager()
            runtime = FakeRuntime()
            functional = FakeFunctional()

            def verify_api_version_match(self, _: object) -> None:
                return None

        def api_factory() -> FakeApi:
            nonlocal factory_calls
            factory_calls += 1
            return FakeApi()

        with tempfile.TemporaryDirectory() as temporary_directory:
            retry_config = replace(
                config,
                output_root=Path(temporary_directory),
                max_attempts=2,
                retry_delay_seconds=0.0,
            )
            wrapper = EnergyPlusWrapper(retry_config, api_factory=api_factory)
            with self.assertRaisesRegex(RuntimeCleanupError, "cleanup failed"):
                wrapper.run("combined-failure", "test")
            attempts = [
                path for path in Path(temporary_directory).iterdir() if path.is_dir()
            ]
            self.assertEqual(len(attempts), 1)
            failure_summary = json.loads(
                (attempts[0] / retry_config.summary_name).read_text(encoding="utf-8")
            )
            self.assertEqual(failure_summary["stage"], "runtime_cleanup")
            self.assertEqual(failure_summary["error_type"], "RuntimeCleanupError")

        self.assertEqual(factory_calls, 1)


class Phase1RunnerReportTests(unittest.TestCase):
    """Protect the canonical full A/B report from partial-run output."""

    def test_partial_report_does_not_overwrite_canonical_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            canonical_path = _report_path_for_mode(output_root, "both")
            canonical_report = {"scope": "complete"}
            _write_report(canonical_path, canonical_report)

            baseline_path = _report_path_for_mode(output_root, "baseline")
            _write_report(baseline_path, {"scope": "baseline"})

            self.assertEqual(
                json.loads(canonical_path.read_text(encoding="utf-8")),
                canonical_report,
            )
            self.assertEqual(baseline_path.name, "phase1_report_baseline.json")
            self.assertNotEqual(baseline_path, canonical_path)

    def test_actuated_report_has_mode_specific_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            actuated_path = _report_path_for_mode(output_root, "actuated")

            self.assertEqual(actuated_path.name, "phase1_report_actuated.json")
            self.assertNotEqual(
                actuated_path,
                _report_path_for_mode(output_root, "both"),
            )


class CompletedEvidenceTests(unittest.TestCase):
    """Assert the generated Phase 1 A/B evidence contract."""

    def test_phase1_report_meets_acceptance_gates(self) -> None:
        report_path = REPOSITORY_ROOT / "runs" / "phase1" / "phase1_report.json"
        self.assertTrue(report_path.is_file(), "Run scripts.run_phase1_smoke first")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for case in ("baseline", "actuated"):
            result = report[case]
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["warning_count"], 0)
            self.assertEqual(result["severe_count"], 0)
            self.assertEqual(result["fatal_count"], 0)
            self.assertEqual(result["timestep_count"], 7 * 24 * 4)
            self.assertLessEqual(result["energy_crosscheck_relative_error"], 0.005)
        proof = report["comparison"]["proof"]
        self.assertTrue(proof["writeback_within_tolerance"])
        self.assertTrue(proof["physical_response_proven"])
        self.assertGreater(proof["setpoint_comparison_sample_count"], 0)
        self.assertTrue(report["repeatability"]["timestep_count_matches"])
        self.assertTrue(report["repeatability"]["deterministic_within_1e_9_kwh"])
        self.assertTrue(report["repeatability"]["repeat_run_zero_severe_fatal"])

    def test_actuated_log_contains_apply_and_reset_records(self) -> None:
        report_path = REPOSITORY_ROOT / "runs" / "phase1" / "phase1_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        controlled_zones = set(report["controlled_zones"])
        baseline_records = _load_timestep_records(report["baseline"])
        actuated_records = _load_timestep_records(report["actuated"])
        statuses = {
            record["action_applied"]["status"] for record in actuated_records
        }
        self.assertIn("applied", statuses)
        self.assertIn("reset", statuses)
        self.assertEqual(
            {
                record["action_applied"]["status"]
                for record in baseline_records
            },
            {"reset"},
        )

        applied_steps = 0
        setpoint_samples = 0
        maximum_setpoint_error_c = 0.0
        for record in actuated_records:
            action = record["action_applied"]
            self.assertEqual(set(action["zones"]), controlled_zones)
            if action["status"] == "reset":
                self.assertTrue(
                    all(
                        zone_action["status"] == "reset"
                        for zone_action in action["zones"].values()
                    )
                )
                continue

            self.assertEqual(action["status"], "applied")
            applied_steps += 1
            for zone_name, zone_action in action["zones"].items():
                self.assertEqual(zone_action["status"], "applied")
                heating_c = float(zone_action["heating_c"])
                cooling_c = float(zone_action["cooling_c"])
                self.assertTrue(math.isfinite(heating_c))
                self.assertTrue(math.isfinite(cooling_c))
                self.assertGreaterEqual(cooling_c - heating_c, 1.0)

                reported = record["snapshot"]["zones"][zone_name]
                maximum_setpoint_error_c = max(
                    maximum_setpoint_error_c,
                    abs(float(reported["heating_setpoint_c"]) - heating_c),
                    abs(float(reported["cooling_setpoint_c"]) - cooling_c),
                )
                setpoint_samples += 2

        proof = report["comparison"]["proof"]
        self.assertGreater(applied_steps, 0)
        self.assertEqual(applied_steps, proof["applied_zone_timestep_count"])
        self.assertEqual(
            setpoint_samples,
            proof["setpoint_comparison_sample_count"],
        )
        self.assertLessEqual(maximum_setpoint_error_c, 0.1)
        self.assertAlmostEqual(
            maximum_setpoint_error_c,
            proof["maximum_reported_setpoint_error_c"],
        )

        maximum_temperature_delta_c = max(
            abs(
                float(actuated["air_temperature_c"])
                - float(baseline["air_temperature_c"])
            )
            for baseline_record, actuated_record in zip(
                baseline_records,
                actuated_records,
                strict=True,
            )
            for zone_name, actuated in actuated_record["snapshot"]["zones"].items()
            for baseline in (baseline_record["snapshot"]["zones"][zone_name],)
        )
        self.assertGreater(maximum_temperature_delta_c, 0.01)
        self.assertAlmostEqual(
            maximum_temperature_delta_c,
            proof["maximum_zone_temperature_delta_c"],
        )

    def test_timestep_evidence_is_complete_ordered_and_finite(self) -> None:
        report_path = REPOSITORY_ROOT / "runs" / "phase1" / "phase1_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        controlled_zones = set(report["controlled_zones"])

        for case in ("baseline", "actuated", "repeatability_run"):
            records = _load_timestep_records(report[case])
            self.assertEqual(len(records), 7 * 24 * 4)
            previous_time = -math.inf
            for expected_sequence, record in enumerate(records, start=1):
                snapshot = record["snapshot"]
                self.assertEqual(snapshot["sequence"], expected_sequence)
                parsed_timestamp = datetime.fromisoformat(record["timestamp"])
                self.assertEqual(parsed_timestamp.second, 0)
                self.assertNotIn(":60", record["timestamp"])
                simulation_time = float(snapshot["simulation_time_hours"])
                self.assertAlmostEqual(
                    simulation_time,
                    expected_sequence * report["timestep_minutes"] / 60.0,
                )
                self.assertGreater(simulation_time, previous_time)
                previous_time = simulation_time
                self.assertEqual(set(snapshot["zones"]), controlled_zones)
                for field in (
                    "outdoor_drybulb_c",
                    "facility_electricity_demand_w",
                    "facility_electricity_j",
                ):
                    self.assertTrue(math.isfinite(float(snapshot[field])))
                for zone in snapshot["zones"].values():
                    for field in (
                        "air_temperature_c",
                        "relative_humidity_pct",
                        "co2_ppm",
                        "occupant_count",
                        "fanger_pmv",
                        "heating_setpoint_c",
                        "cooling_setpoint_c",
                    ):
                        self.assertTrue(math.isfinite(float(zone[field])))

            raw_facility_kwh = math.fsum(
                float(record["snapshot"]["facility_electricity_j"])
                for record in records
            ) / 3_600_000.0
            self.assertAlmostEqual(
                raw_facility_kwh,
                float(report[case]["facility_electricity_kwh"]),
                places=9,
            )
            self.assertEqual(records[3]["timestamp"], "2015-07-21T01:00:00")
            self.assertEqual(records[-1]["timestamp"], "2015-07-28T00:00:00")

        actuated_records = _load_timestep_records(report["actuated"])
        repeatability_records = _load_timestep_records(
            report["repeatability_run"]
        )
        self.assertEqual(
            [record["snapshot"] for record in actuated_records],
            [record["snapshot"] for record in repeatability_records],
        )
        self.assertEqual(
            [record["action_applied"] for record in actuated_records],
            [record["action_applied"] for record in repeatability_records],
        )
        self.assertEqual(
            [
                record["action_selected_for_next_timestep"]
                for record in actuated_records
            ],
            [
                record["action_selected_for_next_timestep"]
                for record in repeatability_records
            ],
        )
        for record in actuated_records:
            applied = record["action_applied"]
            if applied["status"] == "applied":
                self.assertEqual(
                    applied["selected_snapshot_sequence"],
                    record["snapshot"]["sequence"] - 1,
                )
                self.assertEqual(
                    actuated_records[
                        applied["selected_snapshot_sequence"] - 1
                    ]["action_selected_for_next_timestep"]["status"],
                    "accepted",
                )


@unittest.skipUnless(
    os.environ.get("RUN_ENERGYPLUS_INTEGRATION") == "1",
    "Set RUN_ENERGYPLUS_INTEGRATION=1 for real-engine negative tests",
)
class RealEngineNegativeTests(unittest.TestCase):
    """Prove a bad exchange key terminates cleanly with an actionable error."""

    def test_unknown_zone_handle_fails_fast(self) -> None:
        config = Phase1Config.load(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            bad_config = replace(
                config,
                controlled_zones=("NOT-A-REAL-ZONE",),
                people_objects={"NOT-A-REAL-ZONE": "NOT-A-REAL-PEOPLE"},
                output_root=Path(temporary_directory),
                max_attempts=2,
            )
            wrapper = EnergyPlusWrapper(bad_config)
            with self.assertRaisesRegex(
                CallbackExecutionError,
                "exchange handles are missing",
            ):
                wrapper.run(
                    run_id="bad-handle",
                    mode="negative-test",
                    policy=None,
                    model_path=config.baseline_model,
                )

            attempts = [
                path for path in Path(temporary_directory).iterdir() if path.is_dir()
            ]
            self.assertEqual(len(attempts), 1, "bad handles must fail without retry")
            failure_summary = json.loads(
                (attempts[0] / bad_config.summary_name).read_text(encoding="utf-8")
            )
            self.assertEqual(failure_summary["status"], "failed")
            self.assertEqual(failure_summary["stage"], "runtime_callback")
            self.assertEqual(
                failure_summary["energyplus_errors"]["severe_count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
