"""Provider-independent tests for the Ollama Phase 2 adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from pydantic import BaseModel

from src.mcp_client import PHASE2_TOOL_NAMES
from src.ollama_provider import (
    DEFAULT_OLLAMA_BASE_URL,
    OllamaProviderConfig,
    OllamaProviderConfigurationError,
    OllamaProviderModelUnavailableError,
    OllamaProviderResponseError,
    OllamaProviderTimeoutError,
    OllamaToolProvider,
    _normalize_argument_format,
    load_ollama_provider_config,
)
from src.phase2_agent import AgentTerminalStatus, Phase2AgentOrchestrator
from src.phase2_contracts import (
    ControlActionStatus,
    GridCarbonIntensityRequest,
    LogReasoningRequest,
    ParseRuntimeErrorsRequest,
    ReadSensorDataRequest,
    SetControlActionRequest,
)
from src.phase2_mock_services import Phase2Services
from src.scripted_provider import ProviderObservation, ScriptedToolCall


class StaticOllamaProvider(OllamaToolProvider):
    """Return mocked Ollama responses without opening a network connection."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(
            config=OllamaProviderConfig(
                provider="ollama",
                model="qwen3:4b-instruct",
            )
        )
        self._responses = list(responses)

    def validate_startup(self) -> None:
        return None

    def _chat_once(self, observation: ProviderObservation) -> dict[str, Any]:
        if not self._responses:
            raise OllamaProviderResponseError("mock responses exhausted")
        response = self._responses.pop(0)
        if response.get("raise_timeout"):
            raise TimeoutError("mock timeout")
        return response


class FailingAfterReadProvider:
    """Read once, then fail so the agent can exercise fallback."""

    name = "failing-provider"

    def __init__(self) -> None:
        self._cursor = 0

    def start_cycle(self) -> None:
        self._cursor = 0

    async def next_tool_call(
        self,
        observation: ProviderObservation,
    ) -> ScriptedToolCall:
        self._cursor += 1
        if self._cursor == 1:
            return ScriptedToolCall(
                call_id="provider-read",
                tool_name="read_sensor_data",
                arguments={"history_steps": 1},
            )
        raise OllamaProviderResponseError("mock malformed provider response")


class FakeMCPClient:
    """Small in-memory MCP client implementing the agent's used surface."""

    def __init__(self) -> None:
        self.services = Phase2Services.deterministic()
        self._tools = {
            name: SimpleNamespace(
                inputSchema={
                    "properties": {"request": {}},
                    "required": ["request"],
                    "additionalProperties": False,
                }
            )
            for name in PHASE2_TOOL_NAMES
        }

    async def __aenter__(self) -> "FakeMCPClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        return None

    @property
    def tool_names(self) -> tuple[str, ...]:
        return PHASE2_TOOL_NAMES

    @property
    def tools(self) -> dict[str, Any]:
        return self._tools

    async def call_request(
        self,
        tool_name: str,
        request: BaseModel,
    ) -> dict[str, Any]:
        if tool_name == "read_sensor_data":
            response = self.services.sensor_store.read(
                ReadSensorDataRequest.model_validate(request)
            )
        elif tool_name == "get_grid_carbon_intensity":
            response = self.services.grid_carbon_store.read(
                GridCarbonIntensityRequest.model_validate(request)
            )
        elif tool_name == "log_reasoning":
            response = self.services.log_reasoning(
                LogReasoningRequest.model_validate(request)
            )
        elif tool_name == "set_control_action":
            response = self.services.submit_action(
                SetControlActionRequest.model_validate(request)
            )
        elif tool_name == "parse_runtime_errors":
            response = self.services.runtime_error_store.retrieve(
                ParseRuntimeErrorsRequest.model_validate(request)
            )
        else:
            raise AssertionError(f"unexpected tool {tool_name}")
        return response.model_dump(mode="json")


def make_observation() -> ProviderObservation:
    return ProviderObservation(
        run_id="ollama-test",
        round_number=1,
        discovered_tool_names=PHASE2_TOOL_NAMES,
        discovered_tool_schemas={
            name: {"type": "object", "properties": {"request": {}}}
            for name in PHASE2_TOOL_NAMES
        },
        tool_calls_remaining=12,
        corrected_action_proposals_remaining=2,
    )


def tool_response(
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "message": {
            "tool_calls": [
                {
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }
            ]
        }
    }


class OllamaProviderToolCallTests(unittest.IsolatedAsyncioTestCase):
    """Exercise response conversion without reaching real Ollama."""

    def test_chat_request_exposes_only_the_next_required_tool(self) -> None:
        provider = OllamaToolProvider(
            config=OllamaProviderConfig(
                provider="ollama",
                model="llama3.2:3b",
                context_tokens=4096,
            )
        )
        response = tool_response("read_sensor_data", {"history_steps": 0})
        with patch.object(
            OllamaToolProvider,
            "_request_json",
            return_value=response,
        ) as request_json:
            provider._chat_once(make_observation())

        payload = request_json.call_args.args[2]
        self.assertEqual(
            [
                tool["function"]["name"]
                for tool in payload["tools"]
            ],
            ["read_sensor_data"],
        )
        self.assertEqual(payload["options"]["num_ctx"], 1024)
        self.assertEqual(payload["options"]["num_predict"], 96)

    def test_nested_ollama_argument_strings_are_normalized_not_clamped(
        self,
    ) -> None:
        reasoning = _normalize_argument_format(
            "log_reasoning",
            {
                "objective_tags": '["thermal_comfort","safety"]',
                "confidence": "0.8",
            },
        )
        action = _normalize_argument_format(
            "set_control_action",
            {
                "commands": (
                    '{"all_zones":{"mode":"set",'
                    '"heating_c":"15.0","cooling_c":"31.0"}}'
                ),
                "hold_steps": "4",
            },
        )

        self.assertEqual(
            reasoning["objective_tags"],
            ["thermal_comfort", "safety"],
        )
        self.assertEqual(reasoning["confidence"], 0.8)
        self.assertEqual(action["hold_steps"], 4)
        self.assertEqual(len(action["commands"]), 5)
        self.assertEqual(
            {item["zone_id"] for item in action["commands"]},
            {
                "SPACE1-1",
                "SPACE2-1",
                "SPACE3-1",
                "SPACE4-1",
                "SPACE5-1",
            },
        )
        self.assertEqual(action["commands"][0]["heating_c"], 15.0)
        self.assertEqual(action["commands"][0]["cooling_c"], 31.0)

    async def test_valid_structured_tool_call_is_converted(self) -> None:
        provider = StaticOllamaProvider(
            [tool_response("read_sensor_data", {"history_steps": 2})]
        )
        provider.start_cycle()

        call = await provider.next_tool_call(make_observation())

        self.assertEqual(call.tool_name, "read_sensor_data")
        self.assertEqual(call.arguments, {"history_steps": 2})
        self.assertEqual(len(provider.evidence), 1)
        self.assertEqual(provider.evidence[0].argument_keys, ("history_steps",))

    async def test_multiple_calls_use_stable_incrementing_call_ids(self) -> None:
        provider = StaticOllamaProvider(
            [
                tool_response("read_sensor_data", {"history_steps": 0}),
                tool_response("get_grid_carbon_intensity", {"forecast_steps": 4}),
            ]
        )
        provider.start_cycle()
        observation = make_observation()
        sensor = Phase2Services.deterministic().sensor_store.read(
            ReadSensorDataRequest(
                request_id="ollama-test-read",
                history_steps=0,
            )
        )

        first = await provider.next_tool_call(observation)
        second = await provider.next_tool_call(
            observation.model_copy(
                update={
                    "round_number": 2,
                    "cycle_id": sensor.snapshot.cycle_id,
                    "snapshot_id": sensor.snapshot.snapshot_id,
                    "sensor_snapshot": sensor.snapshot,
                }
            )
        )

        self.assertEqual(first.call_id, "ollama-test:ollama:01:01")
        self.assertEqual(second.call_id, "ollama-test:ollama:02:02")
        self.assertEqual(
            tuple(item.tool_name for item in provider.evidence),
            ("read_sensor_data", "get_grid_carbon_intensity"),
        )

    async def test_text_only_response_is_rejected(self) -> None:
        provider = StaticOllamaProvider(
            [{"message": {"content": "I would read the sensors first."}}]
        )
        diagnostics: list[dict[str, Any]] = []
        provider.diagnostic_sink = lambda payload: diagnostics.append(
            dict(payload)
        )
        provider.start_cycle()

        with self.assertRaisesRegex(
            OllamaProviderResponseError,
            "text instead of a tool call",
        ):
            await provider.next_tool_call(make_observation())
        self.assertEqual(provider.evidence[0].response_type, "text_only")
        self.assertEqual(
            provider.evidence[0].validation_code,
            "text_only_response",
        )
        self.assertEqual(
            diagnostics[0]["model_response_type"],
            "text_only",
        )
        self.assertEqual(
            diagnostics[0]["validation_code"],
            "text_only_response",
        )
        self.assertFalse(diagnostics[0]["fallback_used"])

    async def test_malformed_tool_arguments_are_rejected(self) -> None:
        provider = StaticOllamaProvider(
            [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_sensor_data",
                                    "arguments": "{not-json",
                                }
                            }
                        ]
                    }
                }
            ]
        )
        provider.start_cycle()

        with self.assertRaisesRegex(
            OllamaProviderResponseError,
            "arguments were not valid JSON",
        ):
            await provider.next_tool_call(make_observation())

    async def test_out_of_sequence_tool_is_rejected(self) -> None:
        provider = StaticOllamaProvider(
            [
                tool_response(
                    "get_grid_carbon_intensity",
                    {"forecast_steps": 4},
                )
            ]
        )
        provider.start_cycle()

        with self.assertRaisesRegex(
            OllamaProviderResponseError,
            "out-of-sequence",
        ):
            await provider.next_tool_call(make_observation())
        self.assertEqual(
            provider.evidence[0].validation_code,
            "wrong_tool_sequence",
        )

    async def test_connection_timeout_is_classified(self) -> None:
        provider = StaticOllamaProvider([{"raise_timeout": True}])
        provider.start_cycle()

        with self.assertRaises(OllamaProviderTimeoutError):
            await provider.next_tool_call(make_observation())
        self.assertEqual(provider.evidence[0].response_type, "provider_timeout")
        self.assertEqual(
            provider.evidence[0].validation_code,
            "provider_timeout",
        )


class OllamaStartupValidationTests(unittest.TestCase):
    """Validate config and startup without loading a real `.env` file."""

    def write_config(self, llm: dict[str, str]) -> Path:
        directory = Path(tempfile.mkdtemp())
        config = directory / "config" / "phase2.yaml"
        config.parent.mkdir()
        config.write_text(
            "llm:\n"
            f"  provider: {llm.get('provider', '')}\n"
            f"  model: {llm.get('model', '')}\n",
            encoding="utf-8",
        )
        return config

    def load_with_env(
        self,
        llm: dict[str, str],
        env: dict[str, str],
    ) -> OllamaProviderConfig:
        with patch("src.ollama_provider.load_dotenv", return_value=False):
            with patch.dict("os.environ", env, clear=True):
                return load_ollama_provider_config(self.write_config(llm))

    def test_missing_provider_is_rejected_by_startup_validation(self) -> None:
        config = self.load_with_env(
            {"provider": "${PHASE2_LLM_PROVIDER}", "model": "qwen3:4b-instruct"},
            {},
        )
        provider = OllamaToolProvider(config=config)

        with self.assertRaisesRegex(
            OllamaProviderConfigurationError,
            "PHASE2_LLM_PROVIDER=ollama",
        ):
            provider.validate_startup()

    def test_missing_model_is_rejected_by_startup_validation(self) -> None:
        config = self.load_with_env(
            {"provider": "ollama", "model": "${PHASE2_LLM_MODEL}"},
            {},
        )
        provider = OllamaToolProvider(config=config)

        with self.assertRaisesRegex(
            OllamaProviderConfigurationError,
            "PHASE2_LLM_MODEL",
        ):
            provider.validate_startup()

    def test_unsupported_provider_is_rejected(self) -> None:
        config = self.load_with_env(
            {"provider": "groq", "model": "qwen3:4b-instruct"},
            {},
        )
        provider = OllamaToolProvider(config=config)

        with self.assertRaisesRegex(
            OllamaProviderConfigurationError,
            "PHASE2_LLM_PROVIDER=ollama",
        ):
            provider.validate_startup()

    def test_valid_ollama_configuration_resolves_without_server_access(self) -> None:
        config = self.load_with_env(
            {
                "provider": "${PHASE2_LLM_PROVIDER}",
                "model": "${PHASE2_LLM_MODEL}",
            },
            {
                "PHASE2_LLM_PROVIDER": "ollama",
                "PHASE2_LLM_MODEL": "qwen3:4b-instruct",
                "OLLAMA_BASE_URL": "http://localhost:11434",
            },
        )

        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "qwen3:4b-instruct")
        self.assertEqual(config.base_url, "http://localhost:11434")

    def test_unavailable_model_is_rejected_from_mocked_tags(self) -> None:
        provider = OllamaToolProvider(
            config=OllamaProviderConfig(
                provider="ollama",
                model="qwen3:4b-instruct",
            )
        )
        with patch.object(
            OllamaToolProvider,
            "_request_json",
            return_value={"models": [{"name": "llama3.1:8b"}]},
        ):
            with self.assertRaises(OllamaProviderModelUnavailableError):
                provider.validate_startup()

    def test_ollama_http_server_error_is_classified(self) -> None:
        provider = OllamaToolProvider(
            config=OllamaProviderConfig(provider="ollama", model="qwen3")
        )
        error = urllib.error.HTTPError(
            url="http://localhost:11434/api/tags",
            code=500,
            msg="server error",
            hdrs=None,
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                OllamaProviderConfigurationError,
                "status 500",
            ):
                provider._request_json("GET", "/api/tags", None)

    def test_connection_error_message_redacts_url_secrets(self) -> None:
        provider = OllamaToolProvider(
            config=OllamaProviderConfig(
                provider="ollama",
                model="qwen3",
                base_url="http://user:secret-token@localhost:11434?api_key=SECRET",
            )
        )
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with self.assertRaises(OllamaProviderConfigurationError) as raised:
                provider._request_json("GET", "/api/tags", None)

        message = str(raised.exception)
        self.assertNotIn("secret-token", message)
        self.assertNotIn("SECRET", message)
        self.assertIn("<redacted>", message)


class ProviderFailureFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Provider failure after a snapshot should use deterministic fallback."""

    async def test_provider_failure_after_sensor_read_applies_fallback(self) -> None:
        client = FakeMCPClient()
        result = await Phase2AgentOrchestrator(
            client_factory=lambda: client,
        ).run_cycle(
            FailingAfterReadProvider(),
            run_id="provider-fallback",
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.FALLBACK_ACCEPTED,
        )
        self.assertEqual(
            result.record.action_status,
            ControlActionStatus.ACCEPTED,
        )
        self.assertTrue(result.record.fallback_used)
        self.assertEqual(
            result.record.tool_sequence,
            ("read_sensor_data", "log_reasoning", "set_control_action"),
        )


if __name__ == "__main__":
    unittest.main()
