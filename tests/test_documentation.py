"""Focused checks for final documentation and the static results dashboard."""

from __future__ import annotations

import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "system_architecture.md"
DASHBOARD = ROOT / "docs" / "dashboard" / "index.html"
DELIVERABLES = ROOT / "docs" / "final_deliverables.md"
COMPARISON = ROOT / "runs" / "final" / "ollama_24h_comparison.json"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")


class _LocalAssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        for name in ("href", "src"):
            value = attributes.get(name)
            if value:
                self.references.append(value)


def _assert_markdown_paths_exist(
    case: unittest.TestCase,
    path: Path,
) -> None:
    text = path.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.split("#", 1)[0].strip("<>")
        if not target or "://" in target or target.startswith("#"):
            continue
        resolved = (path.parent / target).resolve()
        case.assertTrue(
            resolved.exists(),
            f"{path.relative_to(ROOT)} references missing path {target}",
        )


class FinalDocumentationTests(unittest.TestCase):
    def test_required_commands_and_evidence_are_in_readme(self) -> None:
        text = README.read_text(encoding="utf-8")
        commands = (
            "python -m scripts.run_phase1_smoke --mode both",
            "python -m scripts.run_phase2_mcp_smoke",
            "python -m scripts.run_live_scripted_energyplus_smoke",
            "python -u -m scripts.run_live_ollama_energyplus_smoke",
            "python -u -m scripts.run_ollama_energyplus_comparison",
        )
        for command in commands:
            self.assertIn(command, text)
        for tool in (
            "read_sensor_data",
            "get_grid_carbon_intensity",
            "log_reasoning",
            "set_control_action",
            "parse_runtime_errors",
        ):
            self.assertIn(tool, text)
        for value in (
            "180.556 kWh",
            "168.703 kWh",
            "11.853 kWh",
            "6.565%",
            "17.258 kW",
            "15.566 kW",
            "96.82%",
            "95.45%",
        ):
            self.assertIn(value, text)

    def test_readme_labels_deterministic_result_separately(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertIn(
            "Seven-day deterministic-controller comparison",
            text,
        )
        self.assertIn("was not produced by Ollama", text)
        self.assertIn(
            "The Ollama-supervised run reduced energy and peak demand, "
            "while occupied PMV compliance decreased slightly from 96.82% "
            "to 95.45%, with comfort violations increasing from 7 to 10.",
            text.replace("\n", " "),
        )

    def test_architecture_contains_required_flows_and_mermaid(self) -> None:
        text = ARCHITECTURE.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("```mermaid"), 2)
        for term in (
            "EnergyPlus Python API integration",
            "Sensor and actuator flow",
            "MCP server and client",
            "Ollama provider",
            "Agent tool-call sequence",
            "Validation, rejection, and fallback",
            "Latency management",
            "Evidence and logging pipeline",
        ):
            self.assertIn(term, text)

    def test_dashboard_values_match_verified_json(self) -> None:
        report = json.loads(COMPARISON.read_text(encoding="utf-8"))
        html = DASHBOARD.read_text(encoding="utf-8")
        expected = {
            "180.556": report["electricity"]["baseline_total_kwh"],
            "168.703": report["electricity"]["controlled_total_kwh"],
            "11.853": report["electricity"]["absolute_difference_kwh"],
            "6.565%": abs(
                report["electricity"]["percentage_difference_from_baseline"]
            ),
            "17.258": report["peak_demand"]["baseline_w"] / 1000,
            "15.566": report["peak_demand"]["controlled_w"] / 1000,
            "19.457": report["control_actions"][
                "median_llm_latency_seconds"
            ],
            "28.487": report["control_actions"][
                "maximum_llm_latency_seconds"
            ],
        }
        for rendered, value in expected.items():
            numeric = rendered.removesuffix("%")
            self.assertEqual(numeric, f"{value:.3f}")
            self.assertIn(rendered, html)
        self.assertIn("96.82%", html)
        self.assertIn("95.45%", html)
        self.assertIn("This result was produced by a deterministic controller", html)

    def test_dashboard_local_links_and_images_exist(self) -> None:
        parser = _LocalAssetParser()
        parser.feed(DASHBOARD.read_text(encoding="utf-8"))
        images = [
            reference
            for reference in parser.references
            if reference.endswith(".png")
        ]
        self.assertEqual(3, len(images))
        for reference in parser.references:
            if "://" in reference or reference.startswith("#"):
                continue
            resolved = (DASHBOARD.parent / reference).resolve()
            self.assertTrue(
                resolved.exists(),
                f"dashboard references missing path {reference}",
            )

    def test_documentation_markdown_links_resolve(self) -> None:
        for path in (
            README,
            ARCHITECTURE,
            ROOT / "docs" / "current_results.md",
            DELIVERABLES,
        ):
            _assert_markdown_paths_exist(self, path)

    def test_unproduced_media_are_marked_pending(self) -> None:
        text = DELIVERABLES.read_text(encoding="utf-8")
        self.assertIn("- [ ] Video: pending", text)
        self.assertIn("- [ ] Presentation: pending", text)


if __name__ == "__main__":
    unittest.main()
