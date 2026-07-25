"""Run the four-hour live EnergyPlus loop with the configured Ollama model.

This command performs Ollama endpoint and installed-model validation before
EnergyPlus starts. It is intended to be run only after the configured model is
available locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.run_live_scripted_energyplus_smoke import run_live_smoke
from src.ollama_provider import OllamaToolProvider
from src.scripted_provider import ScriptedProviderError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-config",
        default="config/phase1.yaml",
        help="Path to the proven Phase 1 EnergyPlus configuration",
    )
    parser.add_argument(
        "--phase2-config",
        default="config/phase2.yaml",
        help="Path whose provider/model placeholders resolve from .env",
    )
    return parser.parse_args()


def create_live_ollama_provider(
    config_path: str | Path,
) -> OllamaToolProvider:
    """Load `.env` through the provider and validate Ollama startup."""

    return OllamaToolProvider.from_phase2_config(
        config_path,
        scenario_directive=(
            "Use the current live EnergyPlus snapshot. Prefer conservative "
            "thermostat changes, hold accepted commands for four timesteps, "
            "and release controls when safe action is uncertain."
        ),
    )


def run_live_ollama_smoke(
    phase1_config_path: str | Path = "config/phase1.yaml",
    phase2_config_path: str | Path = "config/phase2.yaml",
) -> dict[str, Any]:
    """Validate Ollama first, then run the existing four-hour live path."""

    provider = create_live_ollama_provider(phase2_config_path)
    return run_live_smoke(
        phase1_config_path,
        provider_factory=lambda snapshot: provider,
        artifact_label="live-ollama",
    )


def main() -> int:
    args = _parse_args()
    try:
        report = run_live_ollama_smoke(
            args.phase1_config,
            args.phase2_config,
        )
    except ScriptedProviderError as exc:
        print(
            json.dumps(
                {
                    "event": "live_ollama_startup_failed",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                allow_nan=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(report, allow_nan=False, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_live_ollama_provider",
    "run_live_ollama_smoke",
]
