from __future__ import annotations

import hashlib
import json
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.run_readonly_dashboard import ROOT, create_server


@contextmanager
def running_dashboard() -> Iterator[str]:
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request_json(base_url: str, path: str) -> dict:
    with urlopen(f"{base_url}{path}", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReadOnlyDashboardTests(unittest.TestCase):
    def test_frontend_contains_required_read_only_pages_and_controls(self) -> None:
        index = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
        for phrase in (
            "Overview",
            "Live Loop",
            "Results",
            "Cost Estimate",
            "Evidence",
            "Saved evidence",
            "Live monitoring",
            "Evidence date",
            "All available dates",
            "Replay verified evidence",
            "Start demo",
            "No control or setpoint input is exposed",
        ):
            self.assertIn(phrase, index)
        self.assertNotIn('type="number"', index)
        self.assertNotIn("setpoint-input", index)

    def test_result_routes_return_verified_values(self) -> None:
        with running_dashboard() as base_url:
            comparison = request_json(base_url, "/api/results/ollama-24h")
            cost = request_json(base_url, "/api/results/cost")
            summary = request_json(base_url, "/api/results/summary")

        self.assertAlmostEqual(
            180.55570880622525,
            comparison["electricity"]["baseline_total_kwh"],
        )
        self.assertAlmostEqual(
            168.70276911481415,
            comparison["electricity"]["controlled_total_kwh"],
        )
        self.assertEqual(
            "Simulated 24-hour electricity cost estimate; not an actual utility bill",
            cost["estimate_label"],
        )
        self.assertEqual(
            "deterministic_phase1",
            summary["phase1_deterministic"]["evidence_class"],
        )

    def test_live_replay_returns_five_zone_mcp_evidence(self) -> None:
        with running_dashboard() as base_url:
            payload = request_json(
                base_url,
                "/api/live/events"
                "?run_id=ollama_24h_comparison&mode=replay",
            )

        self.assertEqual("verified_replay", payload["source_mode"])
        self.assertEqual(96, payload["event_count"])
        accepted = [
            event
            for event in payload["events"]
            if event["action_status"] == "accepted"
        ]
        self.assertTrue(accepted)
        event = accepted[-1]
        self.assertEqual(5, len(event["zone_temperatures_c"]))
        self.assertEqual(
            [
                "read_sensor_data",
                "get_grid_carbon_intensity",
                "log_reasoning",
                "set_control_action",
            ],
            event["mcp_tools_called"],
        )
        self.assertEqual("applied", event["actuator_write_result"]["status"])

    def test_live_monitoring_is_read_only_latest_recorded_evidence(self) -> None:
        with running_dashboard() as base_url:
            payload = request_json(
                base_url,
                "/api/live/latest"
                "?run_id=ollama_24h_comparison&mode=live",
            )

        self.assertIn(
            payload["source_mode"],
            {"latest_recorded_evidence", "verified_replay"},
        )
        self.assertLessEqual(payload["event_count"], 1)
        self.assertNotIn("control_endpoint", payload)

    def test_available_dates_come_only_from_verified_artifacts(self) -> None:
        with running_dashboard() as base_url:
            ollama_dates = request_json(
                base_url,
                "/api/live/dates"
                "?run_id=ollama_24h_comparison&mode=replay",
            )
            deterministic_dates = request_json(
                base_url,
                "/api/live/dates"
                "?run_id=phase1_deterministic&mode=replay",
            )
            july_25 = request_json(
                base_url,
                "/api/live/events"
                "?run_id=phase1_deterministic&mode=replay&date=2015-07-25",
            )

        self.assertEqual(
            ["2015-07-21", "2015-07-22"],
            ollama_dates["available_dates"],
        )
        self.assertEqual(
            [
                "2015-07-21",
                "2015-07-22",
                "2015-07-23",
                "2015-07-24",
                "2015-07-25",
                "2015-07-26",
                "2015-07-27",
                "2015-07-28",
            ],
            deterministic_dates["available_dates"],
        )
        self.assertEqual(
            "verified_deterministic_replay",
            july_25["source_mode"],
        )
        self.assertEqual("2015-07-25", july_25["selected_date"])
        self.assertEqual(96, july_25["event_count"])
        self.assertTrue(
            all(
                event["simulated_timestamp"].startswith("2015-07-25")
                for event in july_25["events"]
            )
        )
        self.assertTrue(
            all(
                event["provider_name"] == "deterministic_phase1_policy"
                for event in july_25["events"]
            )
        )

    def test_invalid_date_filter_is_rejected(self) -> None:
        with running_dashboard() as base_url:
            with self.assertRaises(HTTPError) as error:
                urlopen(
                    f"{base_url}/api/live/events"
                    "?run_id=ollama_24h_comparison"
                    "&mode=replay&date=../../.env",
                    timeout=5,
                )
        self.assertEqual(400, error.exception.code)

    def test_mutations_arbitrary_paths_and_env_are_not_exposed(self) -> None:
        with running_dashboard() as base_url:
            mutation = Request(
                f"{base_url}/api/live/events",
                data=b'{"heating_c":30}',
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as mutation_error:
                urlopen(mutation, timeout=5)
            self.assertEqual(405, mutation_error.exception.code)

            blocked_paths = (
                "/download/../../.env",
                "/.env",
                "/api/results/../../.env",
                "/download/not-approved",
            )
            for path in blocked_paths:
                with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}{path}", timeout=5)
                self.assertEqual(404, error.exception.code)

    def test_api_reads_do_not_modify_verified_artifacts(self) -> None:
        protected = (
            ROOT / "runs/final/ollama_24h_comparison.json",
            ROOT / "runs/final/results_summary.json",
            ROOT / "runs/final/cost_estimate.json",
            ROOT / "models/baseline.idf",
        )
        before = {path: sha256(path) for path in protected}

        with running_dashboard() as base_url:
            request_json(base_url, "/api/results/ollama-24h")
            request_json(base_url, "/api/results/cost")
            request_json(
                base_url,
                "/api/live/latest"
                "?run_id=ollama_24h_comparison&mode=live",
            )

        self.assertEqual(before, {path: sha256(path) for path in protected})

    def test_evidence_downloads_use_a_fixed_allowlist(self) -> None:
        with running_dashboard() as base_url:
            evidence = request_json(base_url, "/api/evidence")
            available = [entry for entry in evidence["entries"] if entry["available"]]
            self.assertTrue(evidence["read_only"])
            self.assertGreaterEqual(len(available), 4)
            for entry in available:
                self.assertTrue(entry["download_url"].startswith("/download/"))
                with urlopen(
                    f"{base_url}{entry['download_url']}",
                    timeout=5,
                ) as response:
                    self.assertGreater(len(response.read()), 0)


if __name__ == "__main__":
    unittest.main()
