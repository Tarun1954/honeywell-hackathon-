"""Deterministic, dependency-free tool-call provider for Phase 2 tests.

The provider returns a fixed sequence of semantic tool-call arguments. It never
uses a model, network API, environment configuration, prompt, or hidden
chain-of-thought.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

from src.phase2_contracts import (
    GridCarbonIntensityResponse,
    SensorSnapshot,
)


ScriptIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=48,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
RuntimeIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
RuntimeSummary = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
]
_MAX_ARGUMENT_DEPTH = 8
_MAX_ARGUMENT_NODES = 256
_MAX_SERIALIZED_ARGUMENT_BYTES = 16_384
_SCRIPT_IDENTIFIER_ADAPTER = TypeAdapter(ScriptIdentifier)


class ScriptedProviderError(RuntimeError):
    """Base error for deterministic provider misuse."""


class ScriptedProviderExhausted(ScriptedProviderError):
    """The configured call sequence has no remaining response."""


def _validate_json_value(
    value: Any,
    *,
    depth: int = 0,
    node_counter: list[int] | None = None,
) -> None:
    """Reject non-JSON, nonfinite, overly deep, or oversized call arguments."""

    counter = node_counter if node_counter is not None else [0]
    counter[0] += 1
    if counter[0] > _MAX_ARGUMENT_NODES:
        raise ValueError("scripted arguments exceed the node limit")
    if depth > _MAX_ARGUMENT_DEPTH:
        raise ValueError("scripted arguments exceed the nesting limit")

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scripted arguments require finite numbers")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("scripted argument object keys must be strings")
        for child in value.values():
            _validate_json_value(
                child,
                depth=depth + 1,
                node_counter=counter,
            )
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_json_value(
                child,
                depth=depth + 1,
                node_counter=counter,
            )
        return
    raise ValueError(
        f"scripted arguments contain unsupported {type(value).__name__}"
    )


class ScriptedToolCall(BaseModel):
    """One immutable, deterministic semantic tool-call proposal."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    call_id: ScriptIdentifier
    tool_name: ScriptIdentifier
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def validate_arguments(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("scripted tool arguments must be an object")
        _validate_json_value(value)
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > _MAX_SERIALIZED_ARGUMENT_BYTES:
            raise ValueError("scripted arguments exceed the serialized limit")
        return copy.deepcopy(value)


class RuntimeErrorObservation(BaseModel):
    """Bounded, auditable runtime feedback without raw logs or hidden reasoning."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    error_id: RuntimeIdentifier
    severity: Literal["warning", "severe", "fatal"]
    code: RuntimeIdentifier
    summary: RuntimeSummary
    retryable: StrictBool
    correction_fields: Annotated[
        tuple[RuntimeIdentifier, ...],
        Field(max_length=10),
    ] = ()
    correction_hint: RuntimeSummary | None = None


class ProviderObservation(BaseModel):
    """Fresh bounded state shown to the scripted provider each round."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    run_id: ScriptIdentifier
    round_number: Annotated[int, Field(ge=1, le=6)]
    discovered_tool_names: tuple[str, ...]
    cycle_id: str | None = None
    snapshot_id: str | None = None
    sensor_snapshot: SensorSnapshot | None = None
    carbon_signal: GridCarbonIntensityResponse | None = None
    reasoning_log_id: str | None = None
    last_action_status: str | None = None
    last_error_codes: tuple[str, ...] = ()
    last_feedback: Literal[
        "proposal_rejected",
        "mcp_error",
        "action_rejected",
    ] | None = None
    runtime_errors: Annotated[
        tuple[RuntimeErrorObservation, ...],
        Field(max_length=20),
    ] = ()
    runtime_errors_has_more: StrictBool = False
    next_runtime_error_id: RuntimeIdentifier | None = None
    tool_calls_remaining: Annotated[int, Field(ge=0, le=12)]
    corrected_action_proposals_remaining: Annotated[
        int,
        Field(ge=0, le=2),
    ]


class ScriptedProvider:
    """Return a configured call sequence with no nondeterministic behavior."""

    def __init__(
        self,
        calls: tuple[ScriptedToolCall, ...] | list[ScriptedToolCall],
        *,
        name: ScriptIdentifier = "scripted",
    ) -> None:
        self.name = _SCRIPT_IDENTIFIER_ADAPTER.validate_python(name)
        self._calls = tuple(call.model_copy(deep=True) for call in calls)
        self._cursor = 0
        self._started = False

    @property
    def calls(self) -> tuple[ScriptedToolCall, ...]:
        """Return isolated copies of the configured deterministic sequence."""

        return tuple(call.model_copy(deep=True) for call in self._calls)

    @property
    def responses_emitted(self) -> int:
        """Return the number of calls emitted in the active cycle."""

        return self._cursor

    def start_cycle(self) -> None:
        """Reset the cursor so each agent cycle gets fresh provider state."""

        self._cursor = 0
        self._started = True

    async def next_tool_call(
        self,
        observation: ProviderObservation,
    ) -> ScriptedToolCall:
        """Return the next call; the observation is intentionally read-only."""

        if not self._started:
            raise ScriptedProviderError(
                "start_cycle() must be called before next_tool_call()"
            )
        if self._cursor >= len(self._calls):
            raise ScriptedProviderExhausted(
                "scripted provider response sequence is exhausted"
            )
        call = self._calls[self._cursor].model_copy(deep=True)
        self._cursor += 1
        return call


__all__ = [
    "ProviderObservation",
    "RuntimeErrorObservation",
    "ScriptedProvider",
    "ScriptedProviderError",
    "ScriptedProviderExhausted",
    "ScriptedToolCall",
]
