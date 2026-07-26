"""Focused safety and arithmetic tests for offline cost post-processing."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from scripts.generate_cost_estimate import (
    DEFAULT_INPUT,
    ESTIMATE_LABEL,
    RATE_ASSUMPTIONS,
    build_cost_estimate,
    generate_cost_estimate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CostEstimateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparison = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        cls.input_hash = _sha256(DEFAULT_INPUT)

    def _build(self) -> dict[str, object]:
        return build_cost_estimate(
            self.comparison,
            input_path="runs/final/ollama_24h_comparison.json",
            input_sha256=self.input_hash,
        )

    def test_energy_cost_arithmetic_uses_verified_values(self) -> None:
        report = self._build()
        electricity = self.comparison["electricity"]
        rate = Decimal(str(RATE_ASSUMPTIONS["energy_rate_usd_per_kwh"]))
        expected_baseline = Decimal(
            str(electricity["baseline_total_kwh"])
        ) * rate
        expected_controlled = Decimal(
            str(electricity["controlled_total_kwh"])
        ) * rate
        costs = report["energy_cost_estimate"]

        self.assertAlmostEqual(
            float(expected_baseline),
            costs["baseline_usd"],
            places=6,
        )
        self.assertAlmostEqual(
            float(expected_controlled),
            costs["controlled_usd"],
            places=6,
        )
        self.assertAlmostEqual(
            float(expected_baseline - expected_controlled),
            costs["savings_usd"],
            places=6,
        )
        self.assertAlmostEqual(
            abs(electricity["percentage_difference_from_baseline"]),
            costs["savings_percent"],
            places=6,
        )

    def test_estimate_is_clearly_labeled_and_demand_charge_is_omitted(
        self,
    ) -> None:
        report = self._build()
        self.assertEqual(ESTIMATE_LABEL, report["estimate_label"])
        self.assertFalse(report["simulation_rerun"])
        self.assertEqual(
            "offline_post_processing",
            report["evidence_class"],
        )
        self.assertIn("not an actual", report["disclaimer"])
        demand = report["demand_cost_estimate"]
        self.assertEqual("not_estimated", demand["status"])
        self.assertIsNone(demand["demand_charge_usd_per_kw"])
        self.assertIsNone(demand["savings_usd"])

    def test_outputs_are_additive_and_do_not_change_verified_input(self) -> None:
        before = _sha256(DEFAULT_INPUT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = generate_cost_estimate(
                DEFAULT_INPUT,
                json_output=root / "cost_estimate.json",
                csv_output=root / "cost_estimate.csv",
                markdown_output=root / "cost_estimate.md",
            )
            self.assertEqual(before, _sha256(DEFAULT_INPUT))
            self.assertEqual(
                before,
                report["input_evidence"]["sha256"],
            )
            self.assertEqual(
                "runs/final/ollama_24h_comparison.json",
                report["input_evidence"]["path"],
            )
            self.assertEqual(
                {
                    "cost_estimate.json",
                    "cost_estimate.csv",
                    "cost_estimate.md",
                },
                {path.name for path in root.iterdir()},
            )
            csv_text = (root / "cost_estimate.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn(RATE_ASSUMPTIONS["source_url"], csv_text)
            self.assertIn("not an actual utility bill", csv_text)

    def test_output_generation_is_deterministic(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_directory,
            tempfile.TemporaryDirectory() as second_directory,
        ):
            first = Path(first_directory)
            second = Path(second_directory)
            for root in (first, second):
                generate_cost_estimate(
                    DEFAULT_INPUT,
                    json_output=root / "cost_estimate.json",
                    csv_output=root / "cost_estimate.csv",
                    markdown_output=root / "cost_estimate.md",
                )
            for name in (
                "cost_estimate.json",
                "cost_estimate.csv",
                "cost_estimate.md",
            ):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )

    def test_source_assumption_is_complete(self) -> None:
        self.assertEqual("Illinois", RATE_ASSUMPTIONS["geography"])
        self.assertEqual("commercial", RATE_ASSUMPTIONS["customer_sector"])
        self.assertEqual(
            "May 2026",
            RATE_ASSUMPTIONS["observation_period"],
        )
        self.assertEqual(
            "2026-07-23",
            RATE_ASSUMPTIONS["source_publication_date"],
        )
        self.assertEqual(15.36, RATE_ASSUMPTIONS["energy_rate_cents_per_kwh"])
        self.assertTrue(
            str(RATE_ASSUMPTIONS["source_url"]).startswith(
                "https://www.eia.gov/"
            )
        )


if __name__ == "__main__":
    unittest.main()
