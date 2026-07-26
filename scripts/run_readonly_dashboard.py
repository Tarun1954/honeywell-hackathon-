from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = ROOT / "frontend"
RUNS_ROOT = ROOT / "runs"

STATIC_FILES = {
    "/": FRONTEND_ROOT / "index.html",
    "/index.html": FRONTEND_ROOT / "index.html",
    "/styles.css": FRONTEND_ROOT / "styles.css",
    "/app.js": FRONTEND_ROOT / "app.js",
}

RESULT_FILES = {
    "/api/results/ollama-24h": ROOT / "runs/final/ollama_24h_comparison.json",
    "/api/results/cost": ROOT / "runs/final/cost_estimate.json",
    "/api/results/summary": ROOT / "runs/final/results_summary.json",
}

DOWNLOAD_FILES = {
    "ollama-24h-json": ROOT / "runs/final/ollama_24h_comparison.json",
    "ollama-24h-csv": ROOT / "runs/final/ollama_24h_comparison.csv",
    "cost-json": ROOT / "runs/final/cost_estimate.json",
    "cost-csv": ROOT / "runs/final/cost_estimate.csv",
    "results-summary-json": ROOT / "runs/final/results_summary.json",
    "results-summary-csv": ROOT / "runs/final/results_summary.csv",
}

CHART_FILES = {
    "/assets/ollama-energy-peak.png": ROOT
    / "runs/final/ollama_24h_energy_peak.png",
    "/assets/ollama-comfort.png": ROOT / "runs/final/ollama_24h_comfort.png",
    "/assets/ollama-actions-latency.png": ROOT
    / "runs/final/ollama_24h_actions_latency.png",
}

RUN_IDS = {
    "ollama_24h_comparison",
    "phase1_deterministic",
    "cost_estimate",
}


@dataclass(frozen=True)
class EvidenceSource:
    path: Path | None
    source_mode: str
    label: str


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return payload


def _resolve_run_artifact(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        resolved = candidate.resolve()
    else:
        lowered_parts = [part.lower() for part in candidate.parts]
        if "runs" not in lowered_parts:
            raise FileNotFoundError("Evidence path is outside the configured run tree.")
        runs_index = lowered_parts.index("runs")
        resolved = (ROOT / Path(*candidate.parts[runs_index:])).resolve()
    if (
        not _is_within(resolved, RUNS_ROOT)
        or resolved.suffix.lower() != ".jsonl"
        or resolved.name != "live_agent_timesteps.jsonl"
    ):
        raise PermissionError("Evidence path is not an approved live JSONL artifact.")
    return resolved


def _saved_ollama_source() -> EvidenceSource:
    comparison = _read_json(RESULT_FILES["/api/results/ollama-24h"])
    raw_path = comparison.get("source_artifacts", {}).get("live_agent_timesteps")
    if not isinstance(raw_path, str):
        return EvidenceSource(None, "unavailable", "No saved Ollama evidence")
    return EvidenceSource(
        _resolve_run_artifact(raw_path),
        "verified_replay",
        "Verified 24-hour Ollama evidence",
    )


def _latest_ollama_source() -> EvidenceSource:
    candidates = list(
        (RUNS_ROOT / "phase1").glob(
            "live-ollama-artifacts-*/live_agent_timesteps.jsonl"
        )
    )
    if not candidates:
        return _saved_ollama_source()
    latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    if not _is_within(latest, RUNS_ROOT):
        raise PermissionError("Latest live evidence escaped the configured run tree.")
    return EvidenceSource(
        latest,
        "latest_recorded_evidence",
        "Latest recorded Ollama smoke evidence",
    )


def _deterministic_source() -> EvidenceSource:
    summary = _read_json(RESULT_FILES["/api/results/summary"])
    sources = summary.get("phase1_deterministic", {}).get("source_artifacts", [])
    for raw_path in sources:
        if isinstance(raw_path, str) and raw_path.endswith(
            "actuated-5/timesteps.jsonl"
        ):
            candidate = (ROOT / raw_path).resolve()
            if (
                _is_within(candidate, RUNS_ROOT)
                and candidate.name == "timesteps.jsonl"
            ):
                return EvidenceSource(
                    candidate,
                    "verified_deterministic_replay",
                    "Verified seven-day deterministic evidence",
                )
    return EvidenceSource(None, "unavailable", "No seven-day deterministic evidence")


def _source_for(run_id: str, view_mode: str) -> EvidenceSource:
    if run_id == "ollama_24h_comparison":
        return _latest_ollama_source() if view_mode == "live" else _saved_ollama_source()
    if run_id == "phase1_deterministic":
        return _deterministic_source()
    return EvidenceSource(None, "not_applicable", "Cost estimate has no live events")


def _normalize_deterministic_record(
    payload: dict[str, Any],
    *,
    cumulative_energy_kwh: float,
) -> dict[str, Any]:
    snapshot = payload.get("snapshot", {})
    zones = snapshot.get("zones", {})
    action = payload.get("action_applied", {})
    action_status = action.get("status", "no_action")
    chosen_setpoints: dict[str, dict[str, Any]] = {}
    previous_setpoints: dict[str, dict[str, Any]] = {}
    for zone_name, zone in zones.items():
        setpoints = {
            "heating_c": zone.get("heating_setpoint_c"),
            "cooling_c": zone.get("cooling_setpoint_c"),
        }
        previous_setpoints[zone_name] = setpoints
        if action_status == "applied":
            chosen_setpoints[zone_name] = {**setpoints, "mode": "set"}
    return {
        "action_status": "held" if action_status == "applied" else "no_action",
        "actuator_write_result": action,
        "chosen_setpoints": chosen_setpoints,
        "corrected_action_count": 0,
        "cycle_id": f"deterministic-{snapshot.get('sequence', 0):06d}",
        "event": "live_phase2_timestep",
        "facility_electricity_demand_w": snapshot.get(
            "facility_electricity_demand_w", 0.0
        ),
        "facility_electricity_kwh_since_start": cumulative_energy_kwh,
        "fallback_used": False,
        "mcp_tools_called": [],
        "occupancy": {
            zone_name: zone.get("occupant_count", 0.0)
            for zone_name, zone in zones.items()
        },
        "pmv": {
            zone_name: zone.get("fanger_pmv")
            for zone_name, zone in zones.items()
        },
        "previous_setpoints": previous_setpoints,
        "provider_name": "deterministic_phase1_policy",
        "rejected_action_count": 0,
        "sequence": snapshot.get("sequence"),
        "simulated_timestamp": payload.get("timestamp"),
        "snapshot_id": f"phase1-{snapshot.get('sequence', 0):06d}",
        "zone_temperatures_c": {
            zone_name: zone.get("air_temperature_c")
            for zone_name, zone in zones.items()
        },
    }


def _read_jsonl(path: Path, *, limit: int = 2000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cumulative_energy_kwh = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("event") == "live_phase2_timestep":
                records.append(payload)
            elif isinstance(payload, dict) and payload.get("event") == "zone_timestep":
                snapshot = payload.get("snapshot", {})
                timestep_energy_j = snapshot.get("facility_electricity_j", 0.0)
                if isinstance(timestep_energy_j, (int, float)):
                    cumulative_energy_kwh += float(timestep_energy_j) / 3_600_000.0
                records.append(
                    _normalize_deterministic_record(
                        payload,
                        cumulative_energy_kwh=cumulative_energy_kwh,
                    )
                )
    return records[-limit:]


def _event_date(record: dict[str, Any]) -> str | None:
    timestamp = record.get("simulated_timestamp")
    if isinstance(timestamp, str) and len(timestamp) >= 10:
        return timestamp[:10]
    return None


def _live_payload(
    run_id: str,
    view_mode: str,
    *,
    latest_only: bool,
    selected_date: str | None = None,
) -> dict[str, Any]:
    source = _source_for(run_id, view_mode)
    if source.path is None or not source.path.exists():
        return {
            "run_id": run_id,
            "view_mode": view_mode,
            "source_mode": source.source_mode,
            "source_label": source.label,
            "available_dates": [],
            "selected_date": selected_date,
            "events": [],
            "event_count": 0,
        }
    records = _read_jsonl(source.path)
    available_dates = sorted(
        {date for record in records if (date := _event_date(record)) is not None}
    )
    if selected_date is not None:
        records = [
            record for record in records if _event_date(record) == selected_date
        ]
    if latest_only and records:
        records = [records[-1]]
    return {
        "run_id": run_id,
        "view_mode": view_mode,
        "source_mode": source.source_mode,
        "source_label": source.label,
        "source_file": source.path.name,
        "available_dates": available_dates,
        "selected_date": selected_date,
        "event_count": len(records),
        "events": records,
    }


def _evidence_index() -> dict[str, Any]:
    entries = []
    labels = {
        "ollama-24h-json": "24-hour Ollama comparison (JSON)",
        "ollama-24h-csv": "24-hour Ollama comparison (CSV)",
        "cost-json": "Offline cost estimate (JSON)",
        "cost-csv": "Offline cost estimate (CSV)",
        "results-summary-json": "Separated results summary (JSON)",
        "results-summary-csv": "Separated results summary (CSV)",
    }
    for evidence_id, path in DOWNLOAD_FILES.items():
        entries.append(
            {
                "id": evidence_id,
                "label": labels[evidence_id],
                "available": path.is_file(),
                "download_url": f"/download/{evidence_id}",
            }
        )
    return {
        "read_only": True,
        "entries": entries,
        "notice": "Downloads are restricted to approved verified artifacts.",
    }


class ReadOnlyDashboardHandler(BaseHTTPRequestHandler):
    server_version = "EcoLoopReadOnly/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _security_headers(self, *, api: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; connect-src 'self'",
        )
        self.send_header("Cache-Control", "no-store" if api else "no-cache")

    def _send_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        api: bool = False,
        download_name: str | None = None,
        head_only: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if download_name:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{download_name}"'
            )
        self._security_headers(api=api)
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
        head_only: bool = False,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(
            encoded,
            content_type="application/json; charset=utf-8",
            status=status,
            api=True,
            head_only=head_only,
        )

    def _send_file(
        self,
        path: Path,
        *,
        download: bool = False,
        head_only: bool = False,
    ) -> None:
        if not path.is_file():
            self._send_json(
                {"error": "approved_artifact_unavailable"},
                status=HTTPStatus.NOT_FOUND,
                head_only=head_only,
            )
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(
            path.read_bytes(),
            content_type=content_type,
            download_name=path.name if download else None,
            head_only=head_only,
        )

    def _handle_get(self, *, head_only: bool = False) -> None:
        request = urlsplit(self.path)
        path = request.path
        query = parse_qs(request.query)

        if path in STATIC_FILES:
            self._send_file(STATIC_FILES[path], head_only=head_only)
            return
        if path in CHART_FILES:
            self._send_file(CHART_FILES[path], head_only=head_only)
            return
        if path in RESULT_FILES:
            if not RESULT_FILES[path].is_file():
                self._send_json(
                    {"error": "approved_result_unavailable"},
                    status=HTTPStatus.NOT_FOUND,
                    head_only=head_only,
                )
                return
            self._send_json(_read_json(RESULT_FILES[path]), head_only=head_only)
            return
        if path == "/api/evidence":
            self._send_json(_evidence_index(), head_only=head_only)
            return
        if path in {"/api/live/events", "/api/live/latest", "/api/live/dates"}:
            run_id = query.get("run_id", ["ollama_24h_comparison"])[0]
            view_mode = query.get("mode", ["replay"])[0]
            if run_id not in RUN_IDS or view_mode not in {"replay", "live"}:
                self._send_json(
                    {"error": "invalid_dashboard_selection"},
                    status=HTTPStatus.BAD_REQUEST,
                    head_only=head_only,
                )
                return
            selected_date = query.get("date", [None])[0]
            if selected_date is not None and not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", selected_date
            ):
                self._send_json(
                    {"error": "invalid_evidence_date"},
                    status=HTTPStatus.BAD_REQUEST,
                    head_only=head_only,
                )
                return
            payload = _live_payload(
                run_id,
                view_mode,
                latest_only=path == "/api/live/latest",
                selected_date=selected_date,
            )
            if path == "/api/live/dates":
                payload = {
                    "run_id": run_id,
                    "view_mode": view_mode,
                    "source_mode": payload["source_mode"],
                    "source_label": payload["source_label"],
                    "available_dates": payload["available_dates"],
                }
            self._send_json(
                payload,
                head_only=head_only,
            )
            return
        if path.startswith("/download/"):
            evidence_id = path.removeprefix("/download/")
            approved = DOWNLOAD_FILES.get(evidence_id)
            if approved is None:
                self._send_json(
                    {"error": "download_not_approved"},
                    status=HTTPStatus.NOT_FOUND,
                    head_only=head_only,
                )
                return
            self._send_file(approved, download=True, head_only=head_only)
            return
        self._send_json(
            {"error": "route_not_found"},
            status=HTTPStatus.NOT_FOUND,
            head_only=head_only,
        )

    def do_GET(self) -> None:
        self._handle_get()

    def do_HEAD(self) -> None:
        self._handle_get(head_only=True)

    def _reject_mutation(self) -> None:
        self._send_json(
            {
                "error": "read_only_dashboard",
                "message": "This server exposes no control or mutation endpoints.",
            },
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    do_POST = _reject_mutation
    do_PUT = _reject_mutation
    do_PATCH = _reject_mutation
    do_DELETE = _reject_mutation


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(host: str, port: int) -> DashboardServer:
    return DashboardServer((host, port), ReadOnlyDashboardHandler)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the isolated Eco-Loop read-only monitoring dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the dashboard in the default browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    server = create_server(args.host, args.port)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    print(
        json.dumps(
            {
                "event": "readonly_dashboard_started",
                "url": url,
                "read_only": True,
                "energyplus_started": False,
                "ollama_contacted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.no_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
