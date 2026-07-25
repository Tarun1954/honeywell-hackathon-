"""Focused tests for deterministic quantitative evidence aggregation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.generate_results_summary import (
    aggregate_results,
    calculate_energy_reduction,
    generate_results_summary,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                record,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _phase1_record(
    sequence: int,
    *,
    temperature: float,
    pmv: float,
    occupancy: float,
    selected_status: str,
    applied: bool,
) -> dict[str, Any]:
    return {
        "snapshot": {
            "sequence": sequence,
            "zones": {
                "ZONE-1": {
                    "air_temperature_c": temperature,
                    "fanger_pmv": pmv,
                    "occupant_count": occupancy,
                }
            },
        },
        "action_selected_for_next_timestep": {
            "status": selected_status,
        },
        "action_applied": {
            "status": "applied" if applied else "reset",
            "zones": {
                "ZONE-1": {
                    "status": "applied" if applied else "reset",
                }
            },
        },
    }


def _live_record(
    sequence: int,
    *,
    action_status: str,
    tools: list[str],
    applied: bool,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "zone_temperatures_c": {"ZONE-1": 22.0 + sequence / 10},
        "pmv": {"ZONE-1": 0.2},
        "occupancy": {"ZONE-1": 1.0},
        "action_status": action_status,
        "fallback_used": False,
        "mcp_tools_called": tools,
        "actuator_write_result": {
            "status": "applied" if applied else "reset",
            "zones": {
                "ZONE-1": {
                    "status": "applied" if applied else "reset",
                }
            },
        },
    }


class ResultsSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        phase1_root = self.root / "runs" / "phase1"
        baseline_directory = phase1_root / "baseline-fixture"
        controlled_directory = phase1_root / "controlled-fixture"
        baseline_records = [
            _phase1_record(
                1,
                temperature=21.0,
                pmv=0.2,
                occupancy=1.0,
                selected_status="no_action",
                applied=False,
            ),
            _phase1_record(
                2,
                temperature=24.0,
                pmv=0.8,
                occupancy=1.0,
                selected_status="no_action",
                applied=False,
            ),
        ]
        controlled_records = [
            _phase1_record(
                1,
                temperature=21.5,
                pmv=0.1,
                occupancy=1.0,
                selected_status="accepted",
                applied=False,
            ),
            _phase1_record(
                2,
                temperature=22.5,
                pmv=0.6,
                occupancy=1.0,
                selected_status="no_action",
                applied=True,
            ),
        ]
        _write_jsonl(
            baseline_directory / "timesteps.jsonl",
            baseline_records,
        )
        _write_jsonl(
            controlled_directory / "timesteps.jsonl",
            controlled_records,
        )
        self.phase1_report = phase1_root / "phase1_report.json"
        _write_json(
            self.phase1_report,
            {
                "timestep_minutes": 15,
                "baseline": {
                    "output_directory": str(baseline_directory),
                    "facility_electricity_kwh": 100.0,
                    "timestep_count": 2,
                    "exit_code": 0,
                    "severe_count": 0,
                    "fatal_count": 0,
                },
                "actuated": {
                    "output_directory": str(controlled_directory),
                    "facility_electricity_kwh": 80.0,
                    "timestep_count": 2,
                    "exit_code": 0,
                    "severe_count": 0,
                    "fatal_count": 0,
                },
            },
        )

        live_directory = (
            phase1_root
            / "live-scripted-artifacts-20260101T000000000000Z"
        )
        live_output = phase1_root / "live-scripted-smoke"
        live_log = live_directory / "live_agent_timesteps.jsonl"
        _write_jsonl(
            live_log,
            [
                _live_record(
                    1,
                    action_status="accepted",
                    tools=[
                        "read_sensor_data",
                        "get_grid_carbon_intensity",
                        "log_reasoning",
                        "set_control_action",
                    ],
                    applied=False,
                ),
                _live_record(
                    2,
                    action_status="held",
                    tools=[],
                    applied=True,
                ),
            ],
        )
        _write_json(
            live_output / "summary.json",
            {
                "facility_electricity_kwh": 0.25,
                "timestep_count": 2,
                "exit_code": 0,
                "severe_count": 0,
                "fatal_count": 0,
            },
        )
        self.live_report = live_directory / "live_smoke_report.json"
        _write_json(
            self.live_report,
            {
                "status": "passed",
                "simulated_hours": 0.5,
                "integration_log": str(live_log),
                "energyplus": {
                    "output_directory": str(live_output),
                },
                "setpoint_changes": {
                    "ZONE-1": {
                        "previous": {
                            "heating_c": 18.0,
                            "cooling_c": 28.0,
                        },
                        "updated": {
                            "heating_c": 21.0,
                            "cooling_c": 25.0,
                        },
                    }
                },
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict[str, Any]:
        return aggregate_results(
            self.root,
            phase1_report_path=self.phase1_report,
            scripted_report_path=self.live_report,
        )

    def test_energy_reduction_arithmetic(self) -> None:
        absolute, percentage = calculate_energy_reduction(100.0, 80.0)

        self.assertEqual(absolute, 20.0)
        self.assertEqual(percentage, 20.0)
        with self.assertRaises(ValueError):
            calculate_energy_reduction(0.0, 0.0)

    def test_evidence_classes_remain_separate(self) -> None:
        summary = self.aggregate()

        self.assertEqual(
            summary["phase1_deterministic"]["evidence_class"],
            "deterministic_phase1",
        )
        self.assertEqual(
            summary["live_scripted_provider"]["evidence_class"],
            "live_scripted_provider",
        )
        self.assertEqual(
            summary["real_ollama"]["evidence_class"],
            "real_ollama_provider",
        )
        self.assertFalse(summary["phase1_deterministic"]["llm_generated"])
        self.assertFalse(
            summary["live_scripted_provider"]["llm_generated"]
        )

    def test_missing_real_provider_evidence_is_pending(self) -> None:
        summary = self.aggregate()
        ollama = summary["real_ollama"]

        self.assertEqual(ollama["status"], "pending")
        self.assertIsNone(ollama["facility_electricity_kwh"])
        self.assertIsNone(ollama["percentage_energy_reduction"])
        self.assertEqual(ollama["source_artifacts"], [])

    def test_result_labels_do_not_misrepresent_deterministic_evidence(self) -> None:
        summary = self.aggregate()
        phase1_label = summary["phase1_deterministic"]["evidence_label"]
        scripted_label = summary["live_scripted_provider"]["evidence_label"]

        self.assertIn("deterministic", phase1_label.lower())
        self.assertIn("not an LLM result", phase1_label)
        self.assertIn("ScriptedProvider", scripted_label)
        self.assertIn("not an LLM result", scripted_label)
        self.assertIsNone(
            summary["live_scripted_provider"][
                "percentage_energy_reduction"
            ]
        )

    def test_output_generation_is_deterministic(self) -> None:
        output_directory = self.root / "generated"
        docs_path = self.root / "docs" / "current_results.md"
        _, outputs = generate_results_summary(
            self.root,
            phase1_report_path=self.phase1_report,
            scripted_report_path=self.live_report,
            output_directory=output_directory,
            docs_path=docs_path,
        )
        first_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in outputs
        }

        _, repeated_outputs = generate_results_summary(
            self.root,
            phase1_report_path=self.phase1_report,
            scripted_report_path=self.live_report,
            output_directory=output_directory,
            docs_path=docs_path,
        )
        repeated_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in repeated_outputs
        }

        self.assertEqual(first_hashes, repeated_hashes)
        self.assertEqual(
            (output_directory / "results_summary.json").read_bytes()[:1],
            b"{",
        )
        for chart_name in (
            "energy_comparison.png",
            "zone_temperature_pmv.png",
            "action_setpoint_evidence.png",
        ):
            self.assertEqual(
                (output_directory / chart_name).read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )


if __name__ == "__main__":
    unittest.main()
