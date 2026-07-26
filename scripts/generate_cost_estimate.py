"""Generate a separate cost estimate from verified 24-hour result artifacts.

This command is offline post-processing. It does not import or invoke
EnergyPlus, Ollama, MCP, the agent loop, or safety validation.

Usage:
    python -m scripts.generate_cost_estimate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs" / "final" / "ollama_24h_comparison.json"
DEFAULT_JSON_OUTPUT = ROOT / "runs" / "final" / "cost_estimate.json"
DEFAULT_CSV_OUTPUT = ROOT / "runs" / "final" / "cost_estimate.csv"
DEFAULT_MARKDOWN_OUTPUT = ROOT / "docs" / "cost_estimate.md"

ESTIMATE_LABEL = (
    "Simulated 24-hour electricity cost estimate; not an actual utility bill"
)
RATE_CENTS_PER_KWH = Decimal("15.36")
RATE_USD_PER_KWH = RATE_CENTS_PER_KWH / Decimal("100")
RATE_SOURCE_URL = (
    "https://www.eia.gov/electricity/monthly/"
    "epm_table_grapher.php?t=epmt_5_6_a"
)
RATE_ASSUMPTIONS: dict[str, Any] = {
    "currency": "USD",
    "geography": "Illinois",
    "customer_sector": "commercial",
    "energy_rate_cents_per_kwh": float(RATE_CENTS_PER_KWH),
    "energy_rate_usd_per_kwh": float(RATE_USD_PER_KWH),
    "rate_type": "statewide_monthly_average_retail_price_proxy",
    "observation_period": "May 2026",
    "source_name": "U.S. Energy Information Administration",
    "source_table": (
        "Electric Power Monthly Table 5.6.A, Average Price of Electricity "
        "to Ultimate Customers by End-Use Sector, by State"
    ),
    "source_publication_date": "2026-07-23",
    "source_url": RATE_SOURCE_URL,
    "demand_charge_usd_per_kw": None,
    "demand_charge_status": "not_estimated",
    "demand_charge_reason": (
        "A 24-hour simulated peak is not a utility billing demand and no "
        "specific monthly tariff was selected."
    ),
}


class CostEstimateError(ValueError):
    """The verified comparison cannot support the requested estimate."""


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Verified 24-hour comparison JSON",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Separate cost-estimate JSON output",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Separate cost-estimate CSV output",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="Separate cost-estimate Markdown output",
    )
    return parser.parse_args(arguments)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CostEstimateError(f"comparison input is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CostEstimateError("comparison input is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise CostEstimateError("comparison input must contain one JSON object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CostEstimateError(f"comparison is missing object {key!r}")
    return value


def _decimal_number(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostEstimateError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise CostEstimateError(f"{label} must be finite")
    return Decimal(str(value))


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _quantity(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def build_cost_estimate(
    comparison: Mapping[str, Any],
    *,
    input_path: str,
    input_sha256: str,
    assumptions: Mapping[str, Any] = RATE_ASSUMPTIONS,
) -> dict[str, Any]:
    """Calculate an energy-only simulated cost estimate."""

    if comparison.get("status") != "completed":
        raise CostEstimateError("comparison status must be 'completed'")
    if (
        comparison.get("comparison_label")
        != "24-hour real Ollama-supervised hybrid comparison"
    ):
        raise CostEstimateError("unexpected comparison evidence label")

    electricity = _mapping(comparison, "electricity")
    peak = _mapping(comparison, "peak_demand")
    baseline_kwh = _decimal_number(
        electricity.get("baseline_total_kwh"),
        label="baseline_total_kwh",
    )
    controlled_kwh = _decimal_number(
        electricity.get("controlled_total_kwh"),
        label="controlled_total_kwh",
    )
    baseline_peak_w = _decimal_number(
        peak.get("baseline_w"),
        label="baseline_w",
    )
    controlled_peak_w = _decimal_number(
        peak.get("controlled_w"),
        label="controlled_w",
    )
    rate = _decimal_number(
        assumptions.get("energy_rate_usd_per_kwh"),
        label="energy_rate_usd_per_kwh",
    )
    if baseline_kwh <= 0 or controlled_kwh < 0 or rate <= 0:
        raise CostEstimateError("energy and rate values must be positive")

    energy_reduction_kwh = baseline_kwh - controlled_kwh
    baseline_energy_cost = baseline_kwh * rate
    controlled_energy_cost = controlled_kwh * rate
    energy_cost_savings = baseline_energy_cost - controlled_energy_cost
    percentage_savings = (
        energy_cost_savings / baseline_energy_cost * Decimal("100")
    )

    return {
        "schema_version": "eco-loop.cost-estimate.v1",
        "estimate_label": ESTIMATE_LABEL,
        "status": "completed",
        "evidence_class": "offline_post_processing",
        "simulation_rerun": False,
        "input_evidence": {
            "comparison_label": comparison["comparison_label"],
            "path": input_path,
            "sha256": input_sha256,
            "duration_hours": comparison.get("duration_hours"),
            "timestep_count": comparison.get("timestep_count"),
        },
        "assumptions": dict(assumptions),
        "verified_energy_and_demand": {
            "baseline_electricity_kwh": _quantity(baseline_kwh),
            "controlled_electricity_kwh": _quantity(controlled_kwh),
            "electricity_reduction_kwh": _quantity(energy_reduction_kwh),
            "baseline_peak_kw": _quantity(baseline_peak_w / Decimal("1000")),
            "controlled_peak_kw": _quantity(
                controlled_peak_w / Decimal("1000")
            ),
        },
        "energy_cost_estimate": {
            "baseline_usd": _money(baseline_energy_cost),
            "controlled_usd": _money(controlled_energy_cost),
            "savings_usd": _money(energy_cost_savings),
            "savings_percent": _quantity(percentage_savings),
        },
        "demand_cost_estimate": {
            "status": assumptions.get("demand_charge_status"),
            "demand_charge_usd_per_kw": assumptions.get(
                "demand_charge_usd_per_kw"
            ),
            "baseline_usd": None,
            "controlled_usd": None,
            "savings_usd": None,
            "reason": assumptions.get("demand_charge_reason"),
        },
        "total_cost_estimate": {
            "scope": "energy_only; demand charges excluded",
            "baseline_usd": _money(baseline_energy_cost),
            "controlled_usd": _money(controlled_energy_cost),
            "savings_usd": _money(energy_cost_savings),
            "savings_percent": _quantity(percentage_savings),
        },
        "disclaimer": (
            "This is a simulated cost estimate based on a statewide average "
            "commercial retail-price proxy and verified 24-hour EnergyPlus "
            "results. It is not an actual tariff calculation or utility bill."
        ),
    }


def _csv_text(report: Mapping[str, Any]) -> str:
    assumptions = _mapping(report, "assumptions")
    energy = _mapping(report, "verified_energy_and_demand")
    costs = _mapping(report, "energy_cost_estimate")
    demand = _mapping(report, "demand_cost_estimate")
    rows = (
        ("metadata", "estimate_label", report["estimate_label"], ""),
        ("metadata", "simulation_rerun", report["simulation_rerun"], ""),
        ("assumption", "energy_rate_cents_per_kwh", assumptions["energy_rate_cents_per_kwh"], "cents/kWh"),
        ("assumption", "energy_rate_usd_per_kwh", assumptions["energy_rate_usd_per_kwh"], "USD/kWh"),
        ("assumption", "geography", assumptions["geography"], ""),
        ("assumption", "customer_sector", assumptions["customer_sector"], ""),
        ("assumption", "observation_period", assumptions["observation_period"], ""),
        ("assumption", "source_publication_date", assumptions["source_publication_date"], ""),
        ("assumption", "source_name", assumptions["source_name"], ""),
        ("assumption", "source_url", assumptions["source_url"], ""),
        ("energy", "baseline_electricity_kwh", energy["baseline_electricity_kwh"], "kWh"),
        ("energy", "controlled_electricity_kwh", energy["controlled_electricity_kwh"], "kWh"),
        ("energy", "electricity_reduction_kwh", energy["electricity_reduction_kwh"], "kWh"),
        ("peak", "baseline_peak_kw", energy["baseline_peak_kw"], "kW"),
        ("peak", "controlled_peak_kw", energy["controlled_peak_kw"], "kW"),
        ("energy_cost", "baseline_usd", costs["baseline_usd"], "USD"),
        ("energy_cost", "controlled_usd", costs["controlled_usd"], "USD"),
        ("energy_cost", "savings_usd", costs["savings_usd"], "USD"),
        ("energy_cost", "savings_percent", costs["savings_percent"], "%"),
        ("demand_cost", "status", demand["status"], ""),
        ("demand_cost", "demand_charge_usd_per_kw", "", "USD/kW"),
        ("demand_cost", "baseline_usd", "", "USD"),
        ("demand_cost", "controlled_usd", "", "USD"),
        ("demand_cost", "savings_usd", "", "USD"),
        ("disclaimer", "text", report["disclaimer"], ""),
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("section", "metric", "value", "unit"))
    writer.writerows(rows)
    return stream.getvalue()


def _markdown_text(report: Mapping[str, Any]) -> str:
    assumptions = _mapping(report, "assumptions")
    energy = _mapping(report, "verified_energy_and_demand")
    costs = _mapping(report, "energy_cost_estimate")
    demand = _mapping(report, "demand_cost_estimate")
    return f"""# Simulated 24-hour electricity cost estimate

This is **offline post-processing of the existing verified comparison**. It did
not rerun or modify EnergyPlus, Ollama, MCP, the agent loop, safety validation,
or the existing comparison artifacts.

## Assumption

- Geography/sector: **{assumptions['geography']} commercial**
- Energy-price proxy: **{assumptions['energy_rate_cents_per_kwh']:.2f} cents/kWh**
- Observation period: **{assumptions['observation_period']}**
- Source publication date: **{assumptions['source_publication_date']}**
- Source: [{assumptions['source_name']} - {assumptions['source_table']}]({assumptions['source_url']})

The EIA value is a statewide monthly average retail price, not a utility rate
schedule for this simulated building.

## Energy-only estimate

| Metric | Baseline | Ollama hybrid |
| --- | ---: | ---: |
| Verified electricity | {energy['baseline_electricity_kwh']:.3f} kWh | {energy['controlled_electricity_kwh']:.3f} kWh |
| Simulated energy cost | ${costs['baseline_usd']:.3f} | ${costs['controlled_usd']:.3f} |

- Electricity reduction: **{energy['electricity_reduction_kwh']:.3f} kWh**
- Simulated energy-cost savings: **${costs['savings_usd']:.3f} ({costs['savings_percent']:.3f}%)**

## Demand charge

Demand-cost status: **{demand['status']}**.

{demand['reason']}

The verified baseline/controlled peaks remain
**{energy['baseline_peak_kw']:.3f}/{energy['controlled_peak_kw']:.3f} kW**, but
no currency value is assigned to them.

## Required disclaimer

**{report['disclaimer']}**
"""


def write_cost_outputs(
    report: Mapping[str, Any],
    *,
    json_output: Path,
    csv_output: Path,
    markdown_output: Path,
) -> tuple[Path, Path, Path]:
    """Write only the three dedicated cost-estimate artifacts."""

    resolved = {
        json_output.resolve(),
        csv_output.resolve(),
        markdown_output.resolve(),
    }
    if len(resolved) != 3:
        raise CostEstimateError("cost output paths must be distinct")
    for path in (json_output, csv_output, markdown_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    csv_output.write_text(
        _csv_text(report),
        encoding="utf-8",
        newline="\n",
    )
    markdown_output.write_text(
        _markdown_text(report),
        encoding="utf-8",
        newline="\n",
    )
    return json_output, csv_output, markdown_output


def generate_cost_estimate(
    input_path: Path = DEFAULT_INPUT,
    *,
    json_output: Path = DEFAULT_JSON_OUTPUT,
    csv_output: Path = DEFAULT_CSV_OUTPUT,
    markdown_output: Path = DEFAULT_MARKDOWN_OUTPUT,
) -> dict[str, Any]:
    """Read verified evidence and create separate cost artifacts."""

    input_resolved = input_path.resolve()
    outputs = {
        json_output.resolve(),
        csv_output.resolve(),
        markdown_output.resolve(),
    }
    if input_resolved in outputs:
        raise CostEstimateError("cost output must not overwrite comparison input")
    before_hash = _sha256(input_path)
    comparison = _load_object(input_path)
    report = build_cost_estimate(
        comparison,
        input_path=_display_path(input_path),
        input_sha256=before_hash,
    )
    written = write_cost_outputs(
        report,
        json_output=json_output,
        csv_output=csv_output,
        markdown_output=markdown_output,
    )
    after_hash = _sha256(input_path)
    if after_hash != before_hash:
        raise RuntimeError("verified comparison input changed during cost estimate")
    print(
        json.dumps(
            {
                "event": "cost_estimate_complete",
                "simulation_rerun": False,
                "input_sha256_unchanged": True,
                "outputs": [str(path) for path in written],
                "energy_cost_estimate": report["energy_cost_estimate"],
                "demand_cost_status": report["demand_cost_estimate"]["status"],
            },
            sort_keys=True,
        )
    )
    return report


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    try:
        generate_cost_estimate(
            args.input,
            json_output=args.json_output,
            csv_output=args.csv_output,
            markdown_output=args.markdown_output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "cost_estimate_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CSV_OUTPUT",
    "DEFAULT_INPUT",
    "DEFAULT_JSON_OUTPUT",
    "DEFAULT_MARKDOWN_OUTPUT",
    "ESTIMATE_LABEL",
    "RATE_ASSUMPTIONS",
    "build_cost_estimate",
    "generate_cost_estimate",
    "write_cost_outputs",
]
