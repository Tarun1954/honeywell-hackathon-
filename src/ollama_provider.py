"""Ollama-backed provider adapter for the Phase 2 MCP agent loop.

The adapter is intentionally small: it proposes semantic tool calls only.
The existing agent remains responsible for correlation IDs, MCP dispatch,
validation, idempotency, and fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import yaml
from dotenv import load_dotenv

from src.phase2_contracts import ToolError
from src.phase2_mock_services import PHASE1_ZONE_IDS
from src.scripted_provider import (
    ProviderObservation,
    ScriptIdentifier,
    ScriptedProviderError,
    ScriptedToolCall,
)


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_CONTEXT_TOKENS = 4096
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "phase2.yaml"
_REASONING_TAGS = ("thermal_comfort", "energy_reduction", "safety")


class OllamaProviderConfigurationError(ScriptedProviderError):
    """The configured Ollama provider cannot be used."""


class OllamaProviderAuthenticationError(ScriptedProviderError):
    """The local Ollama endpoint refused access."""


class OllamaProviderModelUnavailableError(ScriptedProviderError):
    """The configured model is not installed in Ollama."""


class OllamaProviderTimeoutError(ScriptedProviderError):
    """The Ollama request exceeded the configured timeout."""


class OllamaProviderResponseError(ScriptedProviderError):
    """Ollama returned no usable tool call."""


@dataclass(frozen=True, slots=True)
class OllamaProviderConfig:
    """Resolved one-provider Phase 2 LLM configuration."""

    provider: str
    model: str
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    context_tokens: int = DEFAULT_CONTEXT_TOKENS


@dataclass(frozen=True, slots=True)
class ProviderCallEvidence:
    """One provider round's latency and selected tool-call summary."""

    round_number: int
    tool_name: str
    latency_seconds: float
    argument_keys: tuple[str, ...]


@dataclass(slots=True)
class OllamaToolProvider:
    """Real Ollama provider compatible with the existing agent interface."""

    config: OllamaProviderConfig
    scenario_directive: str = ""
    name: ScriptIdentifier = "ollama"
    _call_sequence: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)
    _evidence: list[ProviderCallEvidence] = field(
        default_factory=list,
        init=False,
    )

    @classmethod
    def from_phase2_config(
        cls,
        config_path: str | Path = _DEFAULT_CONFIG_PATH,
        *,
        scenario_directive: str = "",
    ) -> "OllamaToolProvider":
        """Load `.env`, resolve config placeholders, and validate startup."""

        config = load_ollama_provider_config(config_path)
        provider = cls(
            config=config,
            scenario_directive=scenario_directive,
            name=_provider_name_for_model(config.model),
        )
        provider.validate_startup()
        return provider

    @property
    def evidence(self) -> tuple[ProviderCallEvidence, ...]:
        """Return provider latency/tool-call evidence for the active cycle."""

        return tuple(self._evidence)

    def validate_startup(self) -> None:
        """Validate provider endpoint and model availability once."""

        if self.config.provider.lower() != "ollama":
            raise OllamaProviderConfigurationError(
                "this checkpoint implements only PHASE2_LLM_PROVIDER=ollama"
            )
        if not self.config.model:
            raise OllamaProviderConfigurationError(
                "PHASE2_LLM_MODEL must name an installed Ollama model"
            )
        try:
            payload = self._request_json("GET", "/api/tags", None)
        except TimeoutError as exc:
            raise OllamaProviderTimeoutError(
                "Ollama startup validation timed out"
            ) from exc
        models = {
            str(item.get("name", ""))
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        model_roots = {name.split(":", 1)[0] for name in models}
        if self.config.model not in models and self.config.model not in model_roots:
            raise OllamaProviderModelUnavailableError(
                f"Ollama model {self.config.model!r} is not installed"
            )

    def start_cycle(self) -> None:
        """Reset per-cycle call IDs and evidence."""

        self._call_sequence = 0
        self._evidence.clear()
        self._started = True

    async def next_tool_call(
        self,
        observation: ProviderObservation,
    ) -> ScriptedToolCall:
        """Ask Ollama for one bounded semantic MCP tool call."""

        if not self._started:
            raise ScriptedProviderError(
                "start_cycle() must be called before next_tool_call()"
            )
        started = time.perf_counter()
        try:
            response = await anyio.to_thread.run_sync(
                self._chat_once,
                observation,
            )
        except TimeoutError as exc:
            raise OllamaProviderTimeoutError(
                "Ollama request timed out"
            ) from exc
        latency = time.perf_counter() - started
        call = self._extract_tool_call(response, observation)
        self._call_sequence += 1
        stable_call = ScriptedToolCall(
            call_id=(
                f"{observation.run_id}:ollama:"
                f"{observation.round_number:02d}:{self._call_sequence:02d}"
            ),
            tool_name=call["tool_name"],
            arguments=call["arguments"],
        )
        self._evidence.append(
            ProviderCallEvidence(
                round_number=observation.round_number,
                tool_name=stable_call.tool_name,
                latency_seconds=latency,
                argument_keys=tuple(sorted(stable_call.arguments)),
            )
        )
        return stable_call

    def _chat_once(self, observation: ProviderObservation) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "stream": False,
            "tools": _tool_specs(observation.discovered_tool_names),
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _observation_prompt(observation, self.scenario_directive)},
            ],
            "options": {
                "temperature": 0,
                "num_ctx": self.config.context_tokens,
            },
        }
        return self._request_json("POST", "/api/chat", payload)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise OllamaProviderAuthenticationError(
                    "Ollama rejected the request"
                ) from exc
            raise OllamaProviderConfigurationError(
                f"Ollama HTTP request failed with status {exc.code}"
            ) from exc
        except TimeoutError:
            raise
        except OSError as exc:
            raise OllamaProviderConfigurationError(
                f"Ollama is unavailable at {_safe_base_url(self.config.base_url)}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaProviderResponseError(
                "Ollama returned non-JSON data"
            ) from exc

    def _extract_tool_call(
        self,
        response: dict[str, Any],
        observation: ProviderObservation,
    ) -> dict[str, Any]:
        message = response.get("message")
        if not isinstance(message, dict):
            raise OllamaProviderResponseError("Ollama response has no message")

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first = tool_calls[0]
            function = first.get("function") if isinstance(first, dict) else None
            if isinstance(function, dict):
                return _normalize_call(
                    function.get("name"),
                    function.get("arguments"),
                    observation,
                )

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaProviderResponseError(
                "Ollama returned no tool call and no JSON content"
            )
        try:
            decoded = json.loads(_strip_json_fence(content))
        except json.JSONDecodeError as exc:
            raise OllamaProviderResponseError(
                "Ollama returned text instead of a tool call"
            ) from exc
        if not isinstance(decoded, dict):
            raise OllamaProviderResponseError(
                "Ollama JSON response must be an object"
            )
        return _normalize_call(
            decoded.get("tool_name") or decoded.get("name"),
            decoded.get("arguments", {}),
            observation,
        )


def load_ollama_provider_config(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> OllamaProviderConfig:
    """Resolve the single supported real provider from Phase 2 config."""

    path = Path(config_path)
    load_dotenv(path.resolve().parents[0].parent / ".env", override=False)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("llm"), dict):
        raise OllamaProviderConfigurationError(
            f"Phase 2 config has no llm section: {path}"
        )
    provider = _resolve_env(str(payload["llm"].get("provider", "")))
    model = _resolve_env(str(payload["llm"].get("model", "")))
    return OllamaProviderConfig(
        provider=provider,
        model=model,
        base_url=os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
    )


def _resolve_env(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "").strip()
    return os.path.expandvars(value).strip()


def _provider_name_for_model(model: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "-", model).strip("-")
    return f"ollama-{cleaned}"[:48] or "ollama"


def _safe_base_url(value: str) -> str:
    sanitized = re.sub(r"//([^/@]+)@", "//<redacted>@", value)
    sanitized = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)=([^&#]+)",
        r"\1=<redacted>",
        sanitized,
    )
    return sanitized


def _normalize_call(
    raw_name: Any,
    raw_arguments: Any,
    observation: ProviderObservation,
) -> dict[str, Any]:
    if not isinstance(raw_name, str) or raw_name not in observation.discovered_tool_names:
        raise OllamaProviderResponseError("provider proposed an unknown tool")
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise OllamaProviderResponseError(
                "tool-call arguments were not valid JSON"
            ) from exc
    if not isinstance(raw_arguments, dict):
        raise OllamaProviderResponseError(
            "tool-call arguments must be a JSON object"
        )
    return {"tool_name": raw_name, "arguments": raw_arguments}


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _system_prompt() -> str:
    return (
        "You are a building-control tool caller. Return exactly one tool call. "
        "Do not reveal hidden chain-of-thought. Use concise summaries only. "
        "Call tools in this order when information is missing: "
        "read_sensor_data, get_grid_carbon_intensity, log_reasoning, "
        "set_control_action. If a control action is rejected, correct it using "
        "the provided errors. Safe default setpoints are heating 20.0 C and "
        "cooling 26.0 C for all five zones."
    )


def _observation_prompt(
    observation: ProviderObservation,
    scenario_directive: str,
) -> str:
    payload = {
        "run_id": observation.run_id,
        "round_number": observation.round_number,
        "available_tools": observation.discovered_tool_names,
        "discovered_tool_schemas": observation.discovered_tool_schemas,
        "cycle_id": observation.cycle_id,
        "snapshot_id": observation.snapshot_id,
        "sensor_snapshot": (
            observation.sensor_snapshot.model_dump(mode="json")
            if observation.sensor_snapshot is not None
            else None
        ),
        "carbon_signal": (
            observation.carbon_signal.model_dump(mode="json")
            if observation.carbon_signal is not None
            else None
        ),
        "reasoning_log_id": observation.reasoning_log_id,
        "last_action_status": observation.last_action_status,
        "last_error_codes": observation.last_error_codes,
        "last_action_errors": [
            error.model_dump(mode="json")
            for error in observation.last_action_errors
        ],
        "runtime_errors": [
            error.model_dump(mode="json") for error in observation.runtime_errors
        ],
        "tool_calls_remaining": observation.tool_calls_remaining,
        "corrected_action_proposals_remaining": (
            observation.corrected_action_proposals_remaining
        ),
        "scenario_directive": scenario_directive,
        "response_shape": {
            "tool_name": "one available tool name",
            "arguments": "semantic arguments only; omit request_id/cycle_id/snapshot_id",
        },
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def _tool_specs(tool_names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [_tool_spec(name) for name in tool_names]


def _tool_spec(name: str) -> dict[str, Any]:
    specs = {
        "read_sensor_data": {
            "history_steps": {"type": "integer", "minimum": 0, "maximum": 4}
        },
        "get_grid_carbon_intensity": {
            "forecast_steps": {"type": "integer", "minimum": 1, "maximum": 16}
        },
        "log_reasoning": {
            "decision_summary": {"type": "string", "maxLength": 256},
            "objective_tags": {
                "type": "array",
                "items": {"type": "string", "enum": list(_REASONING_TAGS)},
                "minItems": 1,
                "maxItems": 3,
            },
            "tradeoff_summary": {"type": "string", "maxLength": 256},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "set_control_action": {
            "commands": {
                "type": "array",
                "minItems": len(PHASE1_ZONE_IDS),
                "maxItems": len(PHASE1_ZONE_IDS),
                "items": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["set", "release"]},
                        "zone_id": {"type": "string", "enum": list(PHASE1_ZONE_IDS)},
                        "heating_c": {"type": "number"},
                        "cooling_c": {"type": "number"},
                    },
                    "required": ["mode", "zone_id"],
                    "additionalProperties": False,
                },
            },
            "hold_steps": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "parse_runtime_errors": {
            "after_error_id": {"type": ["string", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    }
    properties = specs.get(name, {})
    required = [
        key for key in properties if key not in {"after_error_id", "limit"}
    ]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Propose a semantic {name} call for the Phase 2 MCP agent.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def summarize_tool_errors(errors: tuple[ToolError, ...]) -> tuple[str, ...]:
    """Expose concise validation evidence without hidden reasoning."""

    return tuple(f"{error.code.value}:{error.field}:{error.message}" for error in errors)


__all__ = [
    "OllamaProviderAuthenticationError",
    "OllamaProviderConfig",
    "OllamaProviderConfigurationError",
    "OllamaProviderModelUnavailableError",
    "OllamaProviderResponseError",
    "OllamaProviderTimeoutError",
    "OllamaToolProvider",
    "ProviderCallEvidence",
    "load_ollama_provider_config",
    "summarize_tool_errors",
]
