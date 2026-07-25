"""Focused reliability and documentation checks for the final demo."""

from __future__ import annotations

import contextlib
import io
import json
import re
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_final_demo import (
    DEFAULT_COMPARISON_PATH,
    DEFAULT_EVIDENCE_PATH,
    main,
    run_demo,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "demo_script.md"
CHECKLIST = ROOT / "docs" / "demo_recording_checklist.md"
TIMED_HEADING = re.compile(
    r"^## (?P<start>\d):(?P<start_seconds>\d{2})-"
    r"(?P<end>\d):(?P<end_seconds>\d{2})",
    re.MULTILINE,
)


def _seconds(minutes: str, seconds: str) -> int:
    return int(minutes) * 60 + int(seconds)


class FinalDemoTests(unittest.TestCase):
    def test_saved_replay_command_prints_required_evidence(self) -> None:
        output: list[str] = []
        actual_mode = run_demo(mode="replay", emit=output.append)
        text = "\n".join(output)

        self.assertEqual("replay", actual_mode)
        self.assertIn("previously captured verified evidence", text)
        self.assertIn("this is not a live run", text)
        for phrase in (
            "Simulated timestamp:",
            "temperature=",
            "PMV=",
            "occupancy=",
            "Facility electricity:",
            "DISCOVERED: read_sensor_data",
            "MCP CALL 4: set_control_action",
            "Concise LLM decision summary",
            "Safety validation: ACCEPTED",
            "fallback_used=false",
            "heat 16.7 -> 20.0 C",
            "cool 29.4 -> 26.0 C",
            "Actuator-write success: status=applied | zones_applied=5/5",
            "severe=0 | fatal=0",
            "baseline=180.556 kWh",
            "Ollama hybrid=168.703 kWh",
            "reduction=11.853 kWh (6.565%)",
            "Occupied PMV compliance: baseline=96.82% | controlled=95.45%",
        ):
            self.assertIn(phrase, text)
        self.assertNotIn("chain-of-thought:", text.lower())

    def test_auto_mode_falls_back_without_exposing_exception_secrets(self) -> None:
        secret = "sk-this-value-must-never-appear-1234567890"

        def unavailable() -> dict[str, object]:
            raise RuntimeError(f"provider failed with {secret}")

        output: list[str] = []
        actual_mode = run_demo(
            mode="auto",
            live_runner=unavailable,
            emit=output.append,
        )
        text = "\n".join(output)
        self.assertEqual("replay", actual_mode)
        self.assertIn("Live demo unavailable: RuntimeError", text)
        self.assertIn("FALLBACK MODE: SAVED-EVIDENCE REPLAY", text)
        self.assertNotIn(secret, text)
        self.assertNotIn("PHASE2_LLM_MODEL", text)

    def test_live_mode_startup_uses_injected_proven_workflow(self) -> None:
        report = json.loads(DEFAULT_EVIDENCE_PATH.read_text(encoding="utf-8"))
        calls = 0

        def live_runner() -> dict[str, object]:
            nonlocal calls
            calls += 1
            return report

        output: list[str] = []
        actual_mode = run_demo(
            mode="live",
            live_runner=live_runner,
            emit=output.append,
        )
        self.assertEqual(1, calls)
        self.assertEqual("live", actual_mode)
        self.assertIn(
            "EnergyPlus starting; waiting for live sensors",
            "\n".join(output),
        )

    def test_cli_help_and_replay_do_not_require_environment_values(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            help_exit = None
            try:
                main(["--help"])
            except SystemExit as exc:
                help_exit = exc.code
        self.assertEqual(0, help_exit)
        self.assertNotIn(".env contents", stdout.getvalue())

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exit_code = main(["--mode", "replay"])
        self.assertEqual(0, exit_code)
        self.assertNotIn("PHASE2_LLM_PROVIDER=", stdout.getvalue())
        self.assertNotIn("OLLAMA_BASE_URL=", stdout.getvalue())

    def test_auto_cli_sanitizes_live_startup_failure_then_replays(self) -> None:
        secret = "http://user:password@127.0.0.1:11434"
        with mock.patch(
            "scripts.run_final_demo._run_live_workflow",
            side_effect=RuntimeError(secret),
        ):
            with (
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                exit_code = main(["--mode", "auto"])
        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertEqual(0, exit_code)
        self.assertIn("FALLBACK MODE: SAVED-EVIDENCE REPLAY", rendered)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("password", rendered)

    def test_all_demo_evidence_and_document_paths_exist(self) -> None:
        paths = (
            DEFAULT_EVIDENCE_PATH,
            DEFAULT_COMPARISON_PATH,
            SCRIPT,
            CHECKLIST,
            ROOT / "docs" / "system_architecture.md",
            ROOT / "docs" / "dashboard" / "index.html",
            ROOT / "runs" / "final" / "ollama_24h_energy_peak.png",
            ROOT / "runs" / "final" / "ollama_24h_comfort.png",
            ROOT / "runs" / "final" / "ollama_24h_actions_latency.png",
        )
        for path in paths:
            self.assertTrue(path.is_file(), path)

    def test_narration_timeline_and_word_estimate_are_under_three_minutes(
        self,
    ) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        headings = list(TIMED_HEADING.finditer(text))
        self.assertGreaterEqual(len(headings), 8)
        starts = [
            _seconds(match["start"], match["start_seconds"])
            for match in headings
        ]
        ends = [
            _seconds(match["end"], match["end_seconds"])
            for match in headings
        ]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(
            all(left_end == right_start for left_end, right_start in zip(ends, starts[1:]))
        )
        self.assertLessEqual(ends[-1], 180)

        narration_lines = [
            line.removeprefix("> ").strip()
            for line in text.splitlines()
            if line.startswith("> ")
        ]
        words = re.findall(r"\b[\w.-]+\b", " ".join(narration_lines))
        self.assertLessEqual(len(words), 390)
        estimated_seconds_at_125_wpm = len(words) / 125 * 60
        self.assertLessEqual(estimated_seconds_at_125_wpm, 180)

    def test_manifest_uses_planned_video_path_without_creating_video(self) -> None:
        manifest = (
            ROOT / "submission" / "SUBMISSION_MANIFEST.md"
        ).read_text(encoding="utf-8")
        planned = ROOT / "submission" / "video" / "eco_loop_demo.mp4"
        self.assertIn("submission/video/eco_loop_demo.mp4", manifest)
        self.assertIn("planned", manifest.lower())
        self.assertFalse(planned.exists())


if __name__ == "__main__":
    unittest.main()
