"""Focused tests for matched Ollama/EnergyPlus comparison reporting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.run_ollama_energyplus_comparison import (
    build_comparison_report,
    write_comparison_outputs,
)


ZONE_IDS = (
    "SPACE1-1",
    "SPACE2-1",
    "SPACE3-1",
    "SPACE4-1",
    "SPACE5-1",
)


def _records(
    *,
    timesteps: int,
    temperature: float,
    pmv: float,
    demand_w: float,
) -> list[dict[str, Any]]:
    return [
        {
            "snapshot": {
                "sequence": sequence,
                "facility_electricity_demand_w": (
                    demand_w + sequence
                ),
                "zones": {
                    zone_id: {
                        "air_temperature_c": temperature,
                        "fanger_pmv": pmv,
                        "occupant_count": 1.0,
                    }
                    for zone_id in ZONE_IDS
                },
            }
        }
        for sequence in range(1, timesteps + 1)
    ]


def _events(timesteps: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for sequence in range(1, timesteps + 1):
        decision = sequence % 4 == 0
        fallback = decision and sequence == timesteps
        events.append(
            {
                "sequence": sequence,
                "action_status": "accepted" if decision else "held",
                "fallback_used": fallback,
                "rejected_action_count": 1 if sequence == 4 else 0,
                "corrected_action_count": 1 if sequence == 4 else 0,
                "actuator_write_result": {
                    "zones": {
                        zone_id: {"status": "applied"}
                        for zone_id in ZONE_IDS
                    }
                },
            }
        )
    return events


def _result(
    *,
    kwh: float,
    timesteps: int,
    first_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_path": "models/baseline.idf",
        "output_directory": "runs/example",
        "timestep_count": timesteps,
        "facility_electricity_kwh": kwh,
        "exit_code": 0,
        "severe_count": 0,
        "fatal_count": 0,
        "first_snapshot": first_snapshot or {"sequence": 1},
    }


class OllamaEnergyPlusComparisonTests(unittest.TestCase):
    def build_report(self) -> dict[str, Any]:
        timesteps = 32
        return build_comparison_report(
            hours=8,
            timestep_minutes=15,
            baseline_result=_result(kwh=100.0, timesteps=timesteps),
            controlled_result=_result(kwh=90.0, timesteps=timesteps),
            baseline_records=_records(
                timesteps=timesteps,
                temperature=24.0,
                pmv=0.8,
                demand_w=10_000.0,
            ),
            controlled_records=_records(
                timesteps=timesteps,
                temperature=22.0,
                pmv=0.2,
                demand_w=9_000.0,
            ),
            live_events=_events(timesteps),
            provider_diagnostics=[
                {"latency_seconds": 1.0},
                {"latency_seconds": 3.0},
                {"latency_seconds": 2.0},
            ],
            weather_path="weather/test.epw",
            run_period_start=(7, 21),
            run_period_end=(7, 27),
        )

    def test_energy_comfort_action_and_latency_metrics(self) -> None:
        report = self.build_report()

        self.assertEqual(
            report["electricity"]["controlled_minus_baseline_kwh"],
            -10.0,
        )
        self.assertEqual(
            report["electricity"]["absolute_difference_kwh"],
            10.0,
        )
        self.assertEqual(
            report["electricity"]["percentage_difference_from_baseline"],
            -10.0,
        )
        self.assertEqual(
            report["baseline_comfort"][
                "occupied_comfort_violation_count"
            ],
            160,
        )
        self.assertEqual(
            report["controlled_comfort"][
                "occupied_pmv_compliance_percent"
            ],
            100.0,
        )
        self.assertEqual(
            report["control_actions"]["real_actions_accepted"],
            7,
        )
        self.assertEqual(report["control_actions"]["fallback_count"], 1)
        self.assertEqual(report["control_actions"]["actions_rejected"], 1)
        self.assertEqual(
            report["control_actions"]["corrected_action_proposals"],
            1,
        )
        self.assertEqual(
            report["control_actions"]["median_llm_latency_seconds"],
            2.0,
        )
        self.assertEqual(
            report["control_actions"]["maximum_llm_latency_seconds"],
            3.0,
        )

    def test_mismatched_initial_conditions_fail_closed(self) -> None:
        timesteps = 32
        with self.assertRaisesRegex(ValueError, "initial snapshots"):
            build_comparison_report(
                hours=8,
                timestep_minutes=15,
                baseline_result=_result(
                    kwh=100.0,
                    timesteps=timesteps,
                    first_snapshot={"sequence": 1},
                ),
                controlled_result=_result(
                    kwh=90.0,
                    timesteps=timesteps,
                    first_snapshot={"sequence": 2},
                ),
                baseline_records=_records(
                    timesteps=timesteps,
                    temperature=22.0,
                    pmv=0.2,
                    demand_w=10_000.0,
                ),
                controlled_records=_records(
                    timesteps=timesteps,
                    temperature=22.0,
                    pmv=0.2,
                    demand_w=9_000.0,
                ),
                live_events=_events(timesteps),
                provider_diagnostics=[],
                weather_path="weather/test.epw",
                run_period_start=(7, 21),
                run_period_end=(7, 27),
            )

    def test_outputs_keep_seven_day_deterministic_result_separate(
        self,
    ) -> None:
        report = self.build_report()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = write_comparison_outputs(
                report,
                repository_root=root,
                final_directory=root / "runs" / "final",
                markdown_path=root / "docs" / "comparison.md",
            )
            markdown = (root / "docs" / "comparison.md").read_text(
                encoding="utf-8"
            )
            json_text = (
                root / "runs" / "final" / "ollama_24h_comparison.json"
            ).read_text(encoding="utf-8")

            self.assertIn(
                "8-hour real Ollama-supervised hybrid comparison",
                markdown,
            )
            self.assertIn("seven-day deterministic Phase 1", markdown)
            self.assertIn(
                '"included_in_this_comparison": false',
                json_text,
            )
            self.assertEqual(len(outputs), 6)
            for path in outputs[-3:]:
                self.assertEqual(
                    path.read_bytes()[:8],
                    b"\x89PNG\r\n\x1a\n",
                )


if __name__ == "__main__":
    unittest.main()
