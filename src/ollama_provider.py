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
from collections.abc import Callable, Mapping
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
DEFAULT_TIMEOUT_SECONDS = 30.0
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
    tool_name: str | None
    latency_seconds: float
    argument_keys: tuple[str, ...]
    response_type: str = "structured_tool_call"
    validation_code: str = "accepted_tool_call"
    correction_attempt: int = 0
    argument_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OllamaToolProvider:
    """Real Ollama provider compatible with the existing agent interface."""

    config: OllamaProviderConfig
    scenario_directive: str = ""
    diagnostic_sink: Callable[[Mapping[str, Any]], None] | None = None
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
        diagnostic_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> "OllamaToolProvider":
        """Load `.env`, resolve config placeholders, and validate startup."""

        config = load_ollama_provider_config(config_path)
        provider = cls(
            config=config,
            scenario_directive=scenario_directive,
            diagnostic_sink=diagnostic_sink,
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
            self._record_evidence(
                observation,
                started=started,
                tool_name=None,
                argument_keys=(),
                response_type="provider_timeout",
                validation_code="provider_timeout",
                argument_summary={},
            )
            raise OllamaProviderTimeoutError(
                "Ollama request timed out"
            ) from exc
        except Exception as exc:
            self._record_evidence(
                observation,
                started=started,
                tool_name=None,
                argument_keys=(),
                response_type="provider_error",
                validation_code=_exception_code(exc),
                argument_summary={},
            )
            raise
        latency = time.perf_counter() - started
        response_type = _model_response_type(response)
        try:
            call = self._extract_tool_call(response, observation)
        except OllamaProviderResponseError as exc:
            self._record_evidence(
                observation,
                started=started,
                tool_name=None,
                argument_keys=(),
                response_type=response_type,
                validation_code=_response_error_code(exc),
                argument_summary={},
            )
            raise
        self._call_sequence += 1
        stable_call = ScriptedToolCall(
            call_id=(
                f"{observation.run_id}:ollama:"
                f"{observation.round_number:02d}:{self._call_sequence:02d}"
            ),
            tool_name=call["tool_name"],
            arguments=call["arguments"],
        )
        self._record_evidence(
            observation,
            started=started,
            tool_name=stable_call.tool_name,
            argument_keys=tuple(sorted(stable_call.arguments)),
            response_type=response_type,
            validation_code="accepted_tool_call",
            latency_seconds=latency,
            argument_summary=_safe_argument_summary(
                stable_call.tool_name,
                stable_call.arguments,
            ),
        )
        return stable_call

    def _record_evidence(
        self,
        observation: ProviderObservation,
        *,
        started: float,
        tool_name: str | None,
        argument_keys: tuple[str, ...],
        response_type: str,
        validation_code: str,
        latency_seconds: float | None = None,
        argument_summary: dict[str, Any],
    ) -> None:
        evidence = ProviderCallEvidence(
            round_number=observation.round_number,
            tool_name=tool_name,
            latency_seconds=(
                time.perf_counter() - started
                if latency_seconds is None
                else latency_seconds
            ),
            argument_keys=argument_keys,
            response_type=response_type,
            validation_code=validation_code,
            correction_attempt=_correction_attempt(observation),
            argument_summary=argument_summary,
        )
        self._evidence.append(evidence)
        if self.diagnostic_sink is not None:
            diagnostic = {
                "event": "ollama_provider_round",
                "round": evidence.round_number,
                "model_response_type": evidence.response_type,
                "mcp_tool_requested": evidence.tool_name,
                "validation_code": evidence.validation_code,
                "correction_attempt": evidence.correction_attempt,
                "fallback_used": False,
                "latency_seconds": round(evidence.latency_seconds, 3),
                "argument_summary": evidence.argument_summary,
            }
            try:
                self.diagnostic_sink(diagnostic)
            except Exception:
                # Diagnostics must never change control behavior.
                pass

    def _chat_once(self, observation: ProviderObservation) -> dict[str, Any]:
        next_tool = _next_required_tool(observation)
        payload = {
            "model": self.config.model,
            "stream": False,
            "keep_alive": "15m",
            # Expose only the tool allowed at this mandatory sequence step.
            # This keeps the local 3B-model request small and prevents skips.
            "tools": _tool_specs((next_tool,)),
            "messages": [
                {"role": "system", "content": _system_prompt(next_tool)},
                {"role": "user", "content": _observation_prompt(observation, self.scenario_directive)},
            ],
            "options": {
                "temperature": 0,
                "num_ctx": min(
                    self.config.context_tokens,
                    1024
                    if next_tool
                    in {
                        "read_sensor_data",
                        "get_grid_carbon_intensity",
                        "log_reasoning",
                    }
                    else 1024,
                ),
                "num_predict": (
                    96
                    if next_tool
                    in {
                        "read_sensor_data",
                        "get_grid_carbon_intensity",
                    }
                    else 128
                    if next_tool == "log_reasoning"
                    else 128
                ),
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
    if raw_name != _next_required_tool(observation):
        raise OllamaProviderResponseError(
            "provider proposed an out-of-sequence tool"
        )
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
    return {
        "tool_name": raw_name,
        "arguments": _normalize_argument_format(raw_name, raw_arguments),
    }


def _normalize_argument_format(
    tool_name: str,
    raw_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize Ollama's JSON-encoded nested values without changing them."""

    arguments = dict(raw_arguments)
    if tool_name == "read_sensor_data":
        arguments["history_steps"] = _normalize_integer(
            arguments.get("history_steps")
        )
    elif tool_name == "get_grid_carbon_intensity":
        arguments["forecast_steps"] = _normalize_integer(
            arguments.get("forecast_steps")
        )
    elif tool_name == "log_reasoning":
        arguments["objective_tags"] = _normalize_json_list(
            arguments.get("objective_tags")
        )
        arguments["confidence"] = _normalize_number(
            arguments.get("confidence")
        )
    elif tool_name == "set_control_action":
        arguments["hold_steps"] = _normalize_integer(
            arguments.get("hold_steps")
        )
        commands = _normalize_json_value(arguments.get("commands"))
        if (
            isinstance(commands, Mapping)
            and set(commands) == {"all_zones"}
            and isinstance(commands["all_zones"], Mapping)
        ):
            template = dict(commands["all_zones"])
            commands = [
                {"zone_id": zone_id, **template}
                for zone_id in PHASE1_ZONE_IDS
            ]
        if isinstance(commands, list):
            normalized_commands: list[Any] = []
            for command in commands:
                if not isinstance(command, Mapping):
                    normalized_commands.append(command)
                    continue
                normalized = dict(command)
                for field_name in ("heating_c", "cooling_c"):
                    if field_name in normalized:
                        normalized[field_name] = _normalize_number(
                            normalized[field_name]
                        )
                normalized_commands.append(normalized)
            arguments["commands"] = normalized_commands
        else:
            arguments["commands"] = commands
    return arguments


def _normalize_json_list(value: Any) -> Any:
    decoded = _normalize_json_value(value)
    return decoded if isinstance(decoded, list) else value


def _normalize_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_integer(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return value


def _normalize_number(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
        value.strip(),
    ):
        return float(value)
    return value


def _model_response_type(response: Mapping[str, Any]) -> str:
    message = response.get("message")
    if not isinstance(message, Mapping):
        return "missing_message"
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return "structured_tool_call"
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return "no_tool_call"
    try:
        decoded = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError:
        return "text_only"
    return "json_content" if isinstance(decoded, dict) else "malformed_json_content"


def _response_error_code(exc: OllamaProviderResponseError) -> str:
    message = str(exc)
    if "text instead of a tool call" in message:
        return "text_only_response"
    if "no tool call" in message:
        return "no_tool_call"
    if "arguments were not valid JSON" in message:
        return "malformed_tool_arguments"
    if "arguments must be a JSON object" in message:
        return "malformed_tool_arguments"
    if "unknown tool" in message:
        return "unknown_tool"
    if "out-of-sequence tool" in message:
        return "wrong_tool_sequence"
    if "no message" in message:
        return "missing_message"
    return "malformed_provider_response"


def _exception_code(exc: Exception) -> str:
    if isinstance(exc, OllamaProviderTimeoutError):
        return "provider_timeout"
    if isinstance(exc, OllamaProviderResponseError):
        return _response_error_code(exc)
    return type(exc).__name__


def _correction_attempt(observation: ProviderObservation) -> int:
    if observation.last_action_status != "rejected":
        return 0
    return 3 - observation.corrected_action_proposals_remaining


def _safe_argument_summary(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if tool_name in {"read_sensor_data", "get_grid_carbon_intensity"}:
        return dict(arguments)
    if tool_name == "log_reasoning":
        return {
            "argument_keys": sorted(arguments),
            "objective_tags": arguments.get("objective_tags"),
            "confidence": arguments.get("confidence"),
        }
    if tool_name == "set_control_action":
        commands = arguments.get("commands")
        return {
            "hold_steps": arguments.get("hold_steps"),
            "commands": commands if isinstance(commands, list) else commands,
        }
    return {"argument_keys": sorted(arguments)}


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


def _system_prompt(next_tool: str) -> str:
    argument_hint = {
        "read_sensor_data": '{"history_steps":2}',
        "get_grid_carbon_intensity": '{"forecast_steps":4}',
        "log_reasoning": (
            "decision_summary, objective_tags, tradeoff_summary, and numeric "
            "confidence"
        ),
        "set_control_action": (
            'commands={"all_zones":{"mode":"set","heating_c":NUMBER,'
            '"cooling_c":NUMBER}} and hold_steps=4'
        ),
    }[next_tool]
    prompt = (
        f"Call {next_tool} now. It is the only permitted tool in this round. "
        f"Required semantic arguments: {argument_hint}. "
        "Return exactly one native tool call and no prose. Do not restart the "
        "sequence and do not reveal hidden chain-of-thought. The controller "
        "enforces this cross-round order: read_sensor_data, "
        "get_grid_carbon_intensity, log_reasoning, set_control_action."
    )
    if next_tool == "log_reasoning":
        prompt += (
            " Keep both summaries under 12 words and use concise objective "
            "tags; no private reasoning."
        )
    if next_tool == "set_control_action":
        prompt += (
            " After a rejection, correct the action using the compact "
            "feedback. The all_zones command is expanded into exactly one "
            "command for SPACE1-1, SPACE2-1, SPACE3-1, SPACE4-1, and "
            "SPACE5-1 before validation. Choose numeric heating_c and "
            "cooling_c; use hold_steps=4. Safe defaults are heating 20.0 C "
            "and cooling 26.0 C. Safety validation is authoritative."
        )
    return prompt


def _observation_prompt(
    observation: ProviderObservation,
    scenario_directive: str,
) -> str:
    next_tool = _next_required_tool(observation)
    snapshot = observation.sensor_snapshot
    compact_zones = (
        [
            {
                "zone_id": zone.zone_id,
                "temperature_c": zone.air_temperature_c,
                "pmv": zone.fanger_pmv,
                "occupancy": zone.occupant_count,
                "heating_setpoint_c": zone.heating_setpoint_c,
                "cooling_setpoint_c": zone.cooling_setpoint_c,
            }
            for zone in snapshot.zones
        ]
        if snapshot is not None
        and next_tool == "log_reasoning"
        else None
    )
    action_state = (
        {
            "zone_ids": [zone.zone_id for zone in snapshot.zones],
            "temperature_c_range": [
                min(zone.air_temperature_c for zone in snapshot.zones),
                max(zone.air_temperature_c for zone in snapshot.zones),
            ],
            "pmv_range": [
                min(zone.fanger_pmv for zone in snapshot.zones),
                max(zone.fanger_pmv for zone in snapshot.zones),
            ],
            "total_occupancy": sum(
                zone.occupant_count for zone in snapshot.zones
            ),
        }
        if snapshot is not None and next_tool == "set_control_action"
        else None
    )
    carbon = observation.carbon_signal
    payload = {
        "round_number": observation.round_number,
        "next_required_tool": next_tool,
        "snapshot_id": (
            observation.snapshot_id
            if next_tool != "read_sensor_data"
            else None
        ),
        "zones": compact_zones,
        "action_state": action_state,
        "carbon_g_co2_per_kwh": (
            carbon.current.g_co2_per_kwh
            if carbon is not None
            and next_tool in {"log_reasoning", "set_control_action"}
            else None
        ),
        "reasoning_logged": observation.reasoning_log_id is not None,
        "last_action_status": observation.last_action_status,
        "rejection_feedback": [
            {
                "code": error.code.value,
                "field": error.field,
                "message": error.message,
                "retryable": error.retryable,
            }
            for error in observation.last_action_errors[:10]
        ],
        "tool_calls_remaining": observation.tool_calls_remaining,
        "correction_attempt": _correction_attempt(observation),
        "scenario_directive": (
            scenario_directive
            if next_tool in {"log_reasoning", "set_control_action"}
            else ""
        ),
        "instruction": f"Call {next_tool} now.",
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def _next_required_tool(observation: ProviderObservation) -> str:
    if observation.sensor_snapshot is None:
        return "read_sensor_data"
    if observation.carbon_signal is None:
        return "get_grid_carbon_intensity"
    if observation.reasoning_log_id is None:
        return "log_reasoning"
    return "set_control_action"


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
                "type": "object",
                "description": (
                    "A compact all_zones command expanded into the exact five "
                    "Phase 1 zone commands before MCP validation."
                ),
                "properties": {
                    "all_zones": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["set", "release"],
                            },
                            "heating_c": {"type": "number"},
                            "cooling_c": {"type": "number"},
                        },
                        "required": [
                            "mode",
                            "heating_c",
                            "cooling_c",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["all_zones"],
                "additionalProperties": False,
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
