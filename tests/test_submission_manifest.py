"""Focused integrity, path, and hygiene checks for the submission bundle."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
MANIFEST = SUBMISSION / "SUBMISSION_MANIFEST.md"
MODEL_NOTES = SUBMISSION / "models" / "MODEL_NOTES.md"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")

MODEL_COPIES = {
    "upstream_reference_building.idf": (
        "models/upstream/5ZoneAirCooled-v26.1.0.idf",
        "0187CF7F2CA9C27C43D435A68A8C66A557A43678846813A7E21463A0B0C716CD",
    ),
    "baseline_building.idf": (
        "models/baseline.idf",
        "4A18BAA3FCDAB2A3968D3AD46FEAF5AED63DF93D4ED1FB2F4610D9827F3E36D2",
    ),
    "api_ready_runtime.idf": (
        "models/optimized_runtime.idf",
        "4A18BAA3FCDAB2A3968D3AD46FEAF5AED63DF93D4ED1FB2F4610D9827F3E36D2",
    ),
}

RESULT_FILES = (
    "ollama_24h_comparison.json",
    "ollama_24h_comparison.csv",
    "ollama_24h_energy_peak.png",
    "ollama_24h_comfort.png",
    "ollama_24h_actions_latency.png",
    "results_summary.json",
    "results_summary.csv",
    "energy_comparison.png",
    "zone_temperature_pmv.png",
    "action_setpoint_evidence.png",
)

HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(rb"xox[baprs]-[0-9A-Za-z-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_markdown_links_resolve(
    case: unittest.TestCase,
    path: Path,
) -> None:
    for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
        local = target.split("#", 1)[0].strip("<>")
        if not local or "://" in local or local.startswith("#"):
            continue
        case.assertTrue(
            (path.parent / local).resolve().exists(),
            f"{path.relative_to(ROOT)} references missing path {local}",
        )


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


class SubmissionManifestTests(unittest.TestCase):
    def test_model_copies_are_exact_and_honestly_named(self) -> None:
        for submitted_name, (source_name, expected_hash) in MODEL_COPIES.items():
            source = ROOT / source_name
            submitted = SUBMISSION / "models" / submitted_name
            self.assertTrue(source.is_file())
            self.assertTrue(submitted.is_file())
            self.assertEqual(expected_hash, _sha256(source))
            self.assertEqual(expected_hash, _sha256(submitted))
            self.assertEqual(source.read_bytes(), submitted.read_bytes())

        self.assertFalse(
            (SUBMISSION / "models" / "controlled_building.idf").exists()
        )
        notes = MODEL_NOTES.read_text(encoding="utf-8")
        self.assertIn("no permanently modified controlled IDF", notes)
        self.assertIn("runtime-control experiment", notes)

    def test_weather_and_required_source_paths_exist(self) -> None:
        required = (
            "weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
            "src/energyplus_wrapper.py",
            "src/mcp_server.py",
            "src/mcp_client.py",
            "src/phase2_agent.py",
            "src/ollama_provider.py",
            "src/live_energyplus_integration.py",
            "scripts/run_ollama_energyplus_comparison.py",
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

        weather = ROOT / required[0]
        self.assertEqual(
            "C7D4EFCF93BA316A1D874352E743DF5CF137BA5C0E3459EB2DC4B5442D5B7F5C",
            _sha256(weather),
        )

    def test_result_copies_match_verified_artifacts(self) -> None:
        for name in RESULT_FILES:
            source = ROOT / "runs" / "final" / name
            submitted = SUBMISSION / "results" / name
            self.assertTrue(source.is_file(), name)
            self.assertTrue(submitted.is_file(), name)
            self.assertEqual(source.read_bytes(), submitted.read_bytes(), name)

        report = json.loads(
            (SUBMISSION / "results" / "ollama_24h_comparison.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            "24-hour real Ollama-supervised hybrid comparison",
            report["comparison_label"],
        )
        self.assertAlmostEqual(
            180.55570880622525,
            report["electricity"]["baseline_total_kwh"],
        )
        self.assertAlmostEqual(
            168.70276911481415,
            report["electricity"]["controlled_total_kwh"],
        )
        self.assertFalse(
            report["separate_existing_result"]["included_in_this_comparison"]
        )

    def test_manifest_maps_every_required_deliverable(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8")
        expected_labels = (
            "Functional source code",
            "Original reference building model",
            "Matched-run baseline building model",
            "Modified/controlled building model",
            "Quantitative savings dashboard",
            "Architecture document",
            "Demo video",
            "Presentation",
            "GitHub repository",
        )
        for label in expected_labels:
            self.assertIn(label, text)

        expected_paths = (
            "src/energyplus_wrapper.py",
            "src/mcp_server.py",
            "src/mcp_client.py",
            "src/ollama_provider.py",
            "submission/models/baseline_building.idf",
            "submission/models/api_ready_runtime.idf",
            "submission/models/MODEL_NOTES.md",
            "submission/dashboard/index.html",
            "submission/docs/system_architecture.md",
            "submission/results/ollama_24h_comparison.json",
            "submission/video/VIDEO_PENDING.md",
            "submission/presentation/PRESENTATION_PENDING.md",
        )
        for relative in expected_paths:
            self.assertIn(relative, text)
            self.assertTrue((ROOT / relative).exists(), relative)

        self.assertIn(
            "https://github.com/Tarun1954/honeywell-hackathon-",
            text,
        )
        self.assertIn("Do not create the final ZIP yet", text)
        self.assertEqual([], list(SUBMISSION.rglob("*.zip")))

    def test_submission_document_and_dashboard_links_resolve(self) -> None:
        markdown_files = (
            MANIFEST,
            MODEL_NOTES,
            SUBMISSION / "results" / "README.md",
            SUBMISSION / "docs" / "current_results.md",
            SUBMISSION / "docs" / "ollama_24h_comparison.md",
            SUBMISSION / "docs" / "system_architecture.md",
            SUBMISSION / "docs" / "phase1_report.md",
        )
        for path in markdown_files:
            self.assertTrue(path.is_file())
            _assert_markdown_links_resolve(self, path)

        dashboard = SUBMISSION / "dashboard" / "index.html"
        parser = _LocalAssetParser()
        parser.feed(dashboard.read_text(encoding="utf-8"))
        for reference in parser.references:
            if "://" in reference or reference.startswith("#"):
                continue
            self.assertTrue(
                (dashboard.parent / reference).resolve().exists(),
                reference,
            )

    def test_repository_excludes_secrets_models_caches_and_logs(self) -> None:
        self.assertEqual("", _git("ls-files", "--", ".env").stdout.strip())
        self.assertEqual(0, _git("check-ignore", "-q", ".env").returncode)
        ignored_examples = (
            ".tools/engine.bin",
            "__pycache__/module.pyc",
            ".pytest_cache/state",
            "runs/phase1/timesteps.jsonl",
            "local-model.gguf",
            "models/downloads/model.safetensors",
            "debug.log",
        )
        for example in ignored_examples:
            self.assertEqual(
                0,
                _git("check-ignore", "-q", example).returncode,
                example,
            )

        tracked = {
            ROOT / line
            for line in _git("ls-files").stdout.splitlines()
            if line
        }
        tracked.update(path for path in SUBMISSION.rglob("*") if path.is_file())
        tracked.add(Path(__file__))
        findings: list[str] = []
        for path in tracked:
            if not path.is_file():
                continue
            payload = path.read_bytes()
            if any(pattern.search(payload) for pattern in HIGH_CONFIDENCE_SECRET_PATTERNS):
                findings.append(str(path.relative_to(ROOT)))
        self.assertEqual([], findings, f"credential signatures found in {findings}")


if __name__ == "__main__":
    unittest.main()
