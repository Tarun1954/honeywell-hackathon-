"""Mocked tests for the live EnergyPlus/Ollama preparation path."""

from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import ANY, Mock, patch

from scripts.run_live_ollama_energyplus_smoke import (
    main,
    run_live_ollama_smoke,
)
from src.energyplus_wrapper import ControlAction, ZoneSetpoints
from src.live_energyplus_integration import (
    LiveAgentOutcome,
    LiveScriptedAgentController,
)
from src.ollama_provider import (
    OllamaProviderConfigurationError,
    OllamaProviderModelUnavailableError,
    OllamaProviderResponseError,
    OllamaProviderTimeoutError,
)
from src.phase2_mock_services import PHASE1_ZONE_IDS
from tests.test_live_energyplus_integration import (
    SAFETY_LIMITS,
    phase1_snapshot,
)


def accepted_ollama_outcome(
    *,
    heating_c: float = 21.0,
    cooling_c: float = 25.0,
) -> LiveAgentOutcome:
    return LiveAgentOutcome(
        action=ControlAction(
            setpoints={
                zone_id: ZoneSetpoints(
                    heating_c=heating_c,
                    cooling_c=cooling_c,
                )
                for zone_id in PHASE1_ZONE_IDS
            },
            source="phase2-ollama-llama3.2:3b-agent",
            reason="Mocked MCP-validated Ollama action",
        ),
        action_status="accepted",
        fallback_used=False,
        terminal_status="accepted",
        mcp_tools_called=(
            "read_sensor_data",
            "get_grid_carbon_intensity",
            "log_reasoning",
            "set_control_action",
        ),
        action_id="mock-ollama-action",
        provider_name="ollama-llama3.2:3b",
    )


class LiveOllamaIntegrationTests(unittest.TestCase):
    """Exercise live-provider behavior without starting Ollama or EnergyPlus."""

    def make_controller(
        self,
        temporary_root: Path,
        runner: object,
    ) -> LiveScriptedAgentController:
        return LiveScriptedAgentController(
            repository_root=Path(__file__).resolve().parents[1],
            work_directory=temporary_root / "live-ollama-work",
            controlled_zones=PHASE1_ZONE_IDS,
            safety_limits=SAFETY_LIMITS,
            cycle_runner=runner,  # type: ignore[arg-type]
        )

    @staticmethod
    def advance_to_hour(
        controller: LiveScriptedAgentController,
    ) -> ControlAction:
        selected: ControlAction | None = None
        for sequence in range(1, 5):
            selected = controller(phase1_snapshot(sequence))
        if selected is None:
            raise AssertionError("hourly decision returned no safe action")
        return selected

    def test_live_integration_selects_configured_ollama_provider(self) -> None:
        provider = Mock(name="ollama-provider")
        provider.name = "ollama-llama3.2:3b"
        expected = {"status": "mocked"}
        with patch(
            "scripts.run_live_ollama_energyplus_smoke."
            "OllamaToolProvider.from_phase2_config",
            return_value=provider,
        ) as create_provider:
            with patch(
                "scripts.run_live_ollama_energyplus_smoke.run_live_smoke",
                return_value=expected,
            ) as run_smoke:
                result = run_live_ollama_smoke(
                    "config/phase1.yaml",
                    "config/phase2.yaml",
                )

        self.assertEqual(result, expected)
        create_provider.assert_called_once_with(
            "config/phase2.yaml",
            scenario_directive=ANY,
            diagnostic_sink=ANY,
        )
        provider_factory = run_smoke.call_args.kwargs["provider_factory"]
        self.assertIs(provider_factory(Mock()), provider)
        self.assertEqual(
            run_smoke.call_args.kwargs["artifact_label"],
            "live-ollama",
        )

    def test_model_unavailable_during_cycle_uses_fallback(self) -> None:
        def unavailable(*_: object) -> LiveAgentOutcome:
            raise OllamaProviderModelUnavailableError("model disappeared")

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary), unavailable)
            selected = self.advance_to_hour(controller)
            event = dict(controller.events[-1])

        self.assertEqual(selected.setpoints, {})
        self.assertEqual(selected.source, "phase2-scripted-fallback")
        self.assertTrue(event["fallback_used"])
        self.assertEqual(
            event["provider_name"],
            "deterministic-fallback",
        )

    def test_timeout_during_cycle_uses_fallback(self) -> None:
        def timeout(*_: object) -> LiveAgentOutcome:
            raise OllamaProviderTimeoutError("mock timeout")

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary), timeout)
            selected = self.advance_to_hour(controller)
            event = dict(controller.events[-1])

        self.assertEqual(selected.setpoints, {})
        self.assertTrue(event["fallback_used"])
        self.assertEqual(event["action_status"], "fallback_release")

    def test_malformed_tool_call_uses_safe_fallback(self) -> None:
        def malformed(*_: object) -> LiveAgentOutcome:
            raise OllamaProviderResponseError(
                "tool-call arguments were not valid JSON"
            )

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary), malformed)
            selected = self.advance_to_hour(controller)

        self.assertEqual(selected.setpoints, {})
        self.assertEqual(selected.source, "phase2-scripted-fallback")

    def test_accepted_ollama_action_reaches_actuator_write_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary),
                lambda *_: accepted_ollama_outcome(),
            )
            selected = self.advance_to_hour(controller)
            controller.record_actuator_write(
                {
                    "status": "applied",
                    "source": selected.source,
                    "selected_snapshot_sequence": 4,
                    "zones": {
                        zone_id: {
                            "status": "applied",
                            "heating_c": 21.0,
                            "cooling_c": 25.0,
                        }
                        for zone_id in PHASE1_ZONE_IDS
                    },
                }
            )
            held = controller(phase1_snapshot(5))
            event = dict(controller.events[-1])

        self.assertIsNotNone(held)
        self.assertEqual(
            event["actuator_write_result"]["status"],
            "applied",
        )
        self.assertEqual(
            set(event["actuator_write_result"]["zones"]),
            set(PHASE1_ZONE_IDS),
        )
        self.assertEqual(event["provider_name"], "ollama-llama3.2:3b")

    def test_unsafe_ollama_action_never_reaches_energyplus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary),
                lambda *_: accepted_ollama_outcome(
                    heating_c=15.0,
                    cooling_c=31.0,
                ),
            )
            selected = self.advance_to_hour(controller)
            event = dict(controller.events[-1])

        self.assertEqual(selected.setpoints, {})
        self.assertNotEqual(
            selected.source,
            "phase2-ollama-llama3.2:3b-agent",
        )
        self.assertTrue(event["fallback_used"])

    def test_command_reports_provider_startup_failures(self) -> None:
        failures = (
            OllamaProviderConfigurationError(
                "PHASE2_LLM_PROVIDER=ollama is required"
            ),
            OllamaProviderConfigurationError(
                "PHASE2_LLM_MODEL must name an installed Ollama model"
            ),
            OllamaProviderConfigurationError(
                "Ollama is unavailable at http://127.0.0.1:11434"
            ),
            OllamaProviderModelUnavailableError(
                "Ollama model 'llama3.2:3b' is not installed"
            ),
        )
        for failure in failures:
            with self.subTest(error_type=type(failure).__name__, text=str(failure)):
                stderr = io.StringIO()
                with patch(
                    "scripts.run_live_ollama_energyplus_smoke._parse_args",
                    return_value=argparse.Namespace(
                        phase1_config="config/phase1.yaml",
                        phase2_config="config/phase2.yaml",
                    ),
                ):
                    with patch(
                        "scripts.run_live_ollama_energyplus_smoke."
                        "run_live_ollama_smoke",
                        side_effect=failure,
                    ):
                        with redirect_stderr(stderr):
                            exit_code = main()

                self.assertEqual(exit_code, 2)
                output = stderr.getvalue()
                self.assertIn("live_ollama_startup_failed", output)
                self.assertIn(type(failure).__name__, output)
                self.assertIn(str(failure), output)


if __name__ == "__main__":
    unittest.main()
