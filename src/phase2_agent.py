"""Bounded deterministic Phase 2 agent loop over the real MCP client.

The loop accepts only scripted semantic tool-call proposals. It owns all
correlation and idempotency identifiers, validates every proposal, dispatches
through an exact tool table, and emits one concise terminal JSONL record.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal, TextIO, TypeVar

import anyio
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    TypeAdapter,
)

from src.mcp_client import (
    MCPClientBridgeError,
    MCPIndeterminateOutcomeError,
    MCPProtocolError,
    MCPToolInvocationError,
    PHASE2_TOOL_NAMES,
    Phase2MCPClient,
)
from src.phase2_contracts import (
    ControlActionStatus,
    GridCarbonIntensityRequest,
    GridCarbonIntensityResponse,
    LogReasoningRequest,
    LogReasoningResponse,
    ParseRuntimeErrorsRequest,
    ParseRuntimeErrorsResponse,
    ReadSensorDataRequest,
    ReadSensorDataResponse,
    ReleaseZoneCommand,
    RuntimeErrorRecord,
    RuntimeErrorSeverity,
    SafetyErrorCode,
    SCHEMA_VERSION,
    SensorSnapshot,
    SetControlActionRequest,
    SetControlActionResponse,
)
from src.phase2_mock_services import PHASE1_ZONE_IDS
from src.scripted_provider import (
    ProviderObservation,
    RuntimeErrorObservation,
    ScriptedProvider,
    ScriptedProviderExhausted,
    ScriptedToolCall,
)


MAX_AGENT_ROUNDS = 6
MAX_MCP_TOOL_CALLS = 12
MAX_CORRECTED_ACTION_PROPOSALS = 2
AGENT_RECORD_SCHEMA_VERSION = "phase2.agent.v1"

AgentIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=48,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
ConciseText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
_RUN_ID_ADAPTER = TypeAdapter(AgentIdentifier)


class AgentLoopError(RuntimeError):
    """Base error for deterministic agent-loop failures."""


class AgentProposalError(AgentLoopError):
    """A provider proposal failed local validation and was not executed."""


class AgentToolCallLimitError(AgentLoopError):
    """The cycle has no remaining physical MCP tool-call budget."""


class AgentTerminalStatus(StrEnum):
    """Stable terminal outcomes for one bounded agent cycle."""

    ACCEPTED = "accepted"
    FALLBACK_ACCEPTED = "fallback_accepted"
    RUNTIME_BLOCKED = "runtime_blocked"
    LIMIT_EXHAUSTED = "limit_exhausted"
    INDETERMINATE = "indeterminate"
    FAILED = "failed"


class AgentTraceKind(StrEnum):
    """Concise event categories safe to expose in smoke traces."""

    PROVIDER_CALL = "provider_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PROPOSAL_REJECTED = "proposal_rejected"
    DUPLICATE_PREVENTED = "duplicate_prevented"
    FALLBACK = "fallback"
    TERMINAL = "terminal"


class AgentModel(BaseModel):
    """Strict frozen base for externally visible agent records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class AgentLoopLimits(AgentModel):
    """Configurable limits that can only tighten the Phase 2 hard bounds."""

    max_rounds: Annotated[
        StrictInt,
        Field(ge=1, le=MAX_AGENT_ROUNDS),
    ] = MAX_AGENT_ROUNDS
    max_tool_calls: Annotated[
        StrictInt,
        Field(ge=1, le=MAX_MCP_TOOL_CALLS),
    ] = MAX_MCP_TOOL_CALLS
    max_corrected_action_proposals: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CORRECTED_ACTION_PROPOSALS),
    ] = MAX_CORRECTED_ACTION_PROPOSALS
    provider_response_timeout_seconds: Annotated[
        StrictFloat,
        Field(gt=0.0, le=30.0, allow_inf_nan=False),
    ] = 5.0


class AgentTraceEvent(AgentModel):
    """One bounded trace event without raw prompts or reasoning."""

    sequence: Annotated[StrictInt, Field(ge=1)]
    round_number: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_AGENT_ROUNDS),
    ]
    kind: AgentTraceKind
    tool_name: str | None = None
    status: ConciseText
    detail: ConciseText


class AgentCycleRecord(AgentModel):
    """One JSONL-safe terminal record emitted after MCP cleanup."""

    schema_version: Literal["phase2.agent.v1"] = (
        AGENT_RECORD_SCHEMA_VERSION
    )
    contract_schema_version: Literal["phase2.v1"] = SCHEMA_VERSION
    run_id: AgentIdentifier
    provider_name: AgentIdentifier
    terminal_status: AgentTerminalStatus
    terminal_summary: ConciseText
    cycle_id: str | None = None
    snapshot_id: str | None = None
    rounds_used: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_AGENT_ROUNDS),
    ]
    tool_calls_used: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_MCP_TOOL_CALLS),
    ]
    corrected_action_proposals: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CORRECTED_ACTION_PROPOSALS),
    ]
    fallback_used: StrictBool
    action_status: ControlActionStatus | None = None
    action_id: str | None = None
    reasoning_log_ids: tuple[str, ...] = ()
    runtime_error_ids: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    discovered_tool_names: tuple[str, ...]
    tool_sequence: tuple[str, ...]

    def to_jsonl(self) -> str:
        """Serialize this terminal record as exactly one compact JSONL line."""

        return self.model_dump_json() + "\n"


@dataclass(frozen=True, slots=True)
class AgentCycleResult:
    """Terminal record, concise trace, and its single JSONL line."""

    record: AgentCycleRecord
    trace: tuple[AgentTraceEvent, ...]
    jsonl: str


@dataclass(slots=True)
class _CycleState:
    run_id: str
    provider_name: str
    limits: AgentLoopLimits
    rounds_used: int = 0
    tool_calls_used: int = 0
    action_proposals: int = 0
    request_sequence: int = 0
    trace_sequence: int = 0
    snapshot: SensorSnapshot | None = None
    carbon: GridCarbonIntensityResponse | None = None
    reasoning: LogReasoningResponse | None = None
    runtime_errors: ParseRuntimeErrorsResponse | None = None
    last_action: SetControlActionResponse | None = None
    last_error_codes: tuple[str, ...] = ()
    observed_error_codes: list[str] = field(default_factory=list)
    reasoning_log_ids: list[str] = field(default_factory=list)
    runtime_error_ids: list[str] = field(default_factory=list)
    runtime_error_records: list[RuntimeErrorRecord] = field(
        default_factory=list
    )
    discovered_tool_names: tuple[str, ...] = ()
    discovered_tools: dict[str, Any] = field(default_factory=dict)
    tool_sequence: list[str] = field(default_factory=list)
    trace: list[AgentTraceEvent] = field(default_factory=list)
    call_id_fingerprints: dict[str, str] = field(default_factory=dict)
    side_effect_cache: dict[str, BaseModel] = field(default_factory=dict)
    fallback_used: bool = False
    runtime_blocking_seen: bool = False
    last_feedback: str | None = None
    terminal_status: AgentTerminalStatus | None = None
    terminal_summary: str = "cycle did not reach a terminal outcome"

    @property
    def corrected_action_proposals(self) -> int:
        return max(0, self.action_proposals - 1)

    @property
    def terminal(self) -> bool:
        return self.terminal_status is not None


ResponseT = TypeVar("ResponseT", bound=BaseModel)
ClientFactory = Callable[[], Phase2MCPClient]

_REQUEST_RESPONSE_MODELS: Mapping[
    str,
    tuple[type[BaseModel], type[BaseModel]],
] = {
    "read_sensor_data": (
        ReadSensorDataRequest,
        ReadSensorDataResponse,
    ),
    "get_grid_carbon_intensity": (
        GridCarbonIntensityRequest,
        GridCarbonIntensityResponse,
    ),
    "log_reasoning": (
        LogReasoningRequest,
        LogReasoningResponse,
    ),
    "set_control_action": (
        SetControlActionRequest,
        SetControlActionResponse,
    ),
    "parse_runtime_errors": (
        ParseRuntimeErrorsRequest,
        ParseRuntimeErrorsResponse,
    ),
}

_PROVIDER_ARGUMENT_FIELDS: Mapping[str, frozenset[str]] = {
    "read_sensor_data": frozenset({"history_steps"}),
    "get_grid_carbon_intensity": frozenset({"forecast_steps"}),
    "log_reasoning": frozenset(
        {
            "decision_summary",
            "objective_tags",
            "tradeoff_summary",
            "confidence",
        }
    ),
    "set_control_action": frozenset({"commands", "hold_steps"}),
    "parse_runtime_errors": frozenset({"after_error_id", "limit"}),
}

_SIDE_EFFECTING_TOOLS = frozenset(
    {"log_reasoning", "set_control_action"}
)


def _default_client_factory() -> Phase2MCPClient:
    """Disable bridge replay so the agent's physical call bound is exact."""

    return Phase2MCPClient(allow_read_reconnect=False)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


class Phase2AgentOrchestrator:
    """Execute one fresh, bounded scripted cycle over a real MCP session."""

    def __init__(
        self,
        *,
        client_factory: ClientFactory = _default_client_factory,
        limits: AgentLoopLimits | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._limits = limits or AgentLoopLimits()

    async def run_cycle(
        self,
        provider: ScriptedProvider,
        *,
        run_id: str,
        record_stream: TextIO | None = None,
    ) -> AgentCycleResult:
        """Run one provider cycle and emit one terminal JSONL record."""

        canonical_run_id = _RUN_ID_ADAPTER.validate_python(run_id)
        state = _CycleState(
            run_id=canonical_run_id,
            provider_name=provider.name,
            limits=self._limits,
        )
        provider.start_cycle()

        try:
            async with self._client_factory() as client:
                state.discovered_tool_names = client.tool_names
                state.discovered_tools = dict(client.tools)
                self._validate_discovered_tools(state)
                await self._run_provider_rounds(client, provider, state)
        except MCPIndeterminateOutcomeError:
            self._set_terminal(
                state,
                AgentTerminalStatus.INDETERMINATE,
                "A side-effecting MCP outcome was indeterminate; no retry "
                "or fallback was attempted.",
                force=True,
            )
        except MCPClientBridgeError:
            self._set_terminal(
                state,
                AgentTerminalStatus.FAILED,
                "The MCP session failed and the cycle stopped safely.",
                force=True,
            )
        except Exception:
            self._set_terminal(
                state,
                AgentTerminalStatus.FAILED,
                "The deterministic provider or agent cycle failed safely.",
                force=True,
            )

        if not state.terminal:
            self._set_terminal(
                state,
                AgentTerminalStatus.FAILED,
                "The cycle ended without an accepted action.",
            )
        self._trace(
            state,
            0,
            AgentTraceKind.TERMINAL,
            None,
            state.terminal_status.value,
            state.terminal_summary,
        )
        record = self._build_record(state)
        jsonl = record.to_jsonl()
        if record_stream is not None:
            record_stream.write(jsonl)
            record_stream.flush()
        return AgentCycleResult(
            record=record,
            trace=tuple(state.trace),
            jsonl=jsonl,
        )

    async def _run_provider_rounds(
        self,
        client: Phase2MCPClient,
        provider: ScriptedProvider,
        state: _CycleState,
    ) -> None:
        while not state.terminal and (
            state.rounds_used < state.limits.max_rounds
        ):
            round_number = state.rounds_used + 1
            observation = self._build_observation(state, round_number)
            try:
                with anyio.fail_after(
                    state.limits.provider_response_timeout_seconds
                ):
                    call = await provider.next_tool_call(observation)
            except ScriptedProviderExhausted:
                await self._apply_fallback(
                    client,
                    state,
                    round_number=max(1, state.rounds_used),
                    reason="The scripted provider exhausted its responses.",
                )
                return
            except TimeoutError:
                state.last_feedback = "proposal_rejected"
                await self._apply_fallback(
                    client,
                    state,
                    round_number=max(1, state.rounds_used),
                    reason="The scripted provider response timed out.",
                )
                return
            except Exception:
                state.last_feedback = "proposal_rejected"
                await self._apply_fallback(
                    client,
                    state,
                    round_number=max(1, state.rounds_used),
                    reason="The provider failed to return a valid tool call.",
                )
                return

            state.rounds_used = round_number
            self._trace(
                state,
                round_number,
                AgentTraceKind.PROVIDER_CALL,
                call.tool_name,
                "proposed",
                f"Provider proposed call {call.call_id}.",
            )

            if (
                call.tool_name == "set_control_action"
                and state.action_proposals
                >= 1 + state.limits.max_corrected_action_proposals
            ):
                await self._apply_fallback(
                    client,
                    state,
                    round_number=round_number,
                    reason="Corrected action proposal limit was exhausted.",
                )
                return

            try:
                outcome = await self._process_provider_call(
                    client,
                    state,
                    call,
                    round_number,
                )
            except AgentProposalError as exc:
                state.last_feedback = "proposal_rejected"
                self._trace(
                    state,
                    round_number,
                    AgentTraceKind.PROPOSAL_REJECTED,
                    call.tool_name,
                    "rejected_locally",
                    str(exc),
                )
                continue
            except AgentToolCallLimitError:
                await self._apply_fallback(
                    client,
                    state,
                    round_number=round_number,
                    reason="MCP tool-call budget was exhausted.",
                )
                return
            except MCPToolInvocationError:
                state.last_feedback = "mcp_error"
                self._trace(
                    state,
                    round_number,
                    AgentTraceKind.PROPOSAL_REJECTED,
                    call.tool_name,
                    "mcp_error",
                    "The MCP server rejected the validated tool request.",
                )
                if call.tool_name in _SIDE_EFFECTING_TOOLS:
                    self._set_terminal(
                        state,
                        AgentTerminalStatus.INDETERMINATE,
                        "A side-effecting MCP tool returned an execution "
                        "error after it may have written state; no retry or "
                        "fallback was attempted.",
                    )
                    return
                continue

            if state.terminal or outcome == "accepted":
                return
            if outcome == "fallback":
                await self._apply_fallback(
                    client,
                    state,
                    round_number=round_number,
                    reason="Corrected action proposals were exhausted.",
                )
                return
            if outcome == "runtime_blocked":
                self._set_terminal(
                    state,
                    AgentTerminalStatus.RUNTIME_BLOCKED,
                    "Runtime errors remained blocking after bounded "
                    "correction; baseline control remains active.",
                )
                return

        if not state.terminal:
            await self._apply_fallback(
                client,
                state,
                round_number=max(1, state.rounds_used),
                reason="Maximum agent rounds were exhausted.",
            )

    async def _process_provider_call(
        self,
        client: Phase2MCPClient,
        state: _CycleState,
        call: ScriptedToolCall,
        round_number: int,
    ) -> str:
        if call.tool_name not in state.discovered_tools:
            raise AgentProposalError(
                f"Unknown tool {call.tool_name!r}; nothing was executed."
            )
        if call.tool_name not in _REQUEST_RESPONSE_MODELS:
            raise AgentProposalError(
                f"Unsupported tool {call.tool_name!r}; nothing was executed."
            )

        raw_fingerprint = _canonical_json(
            {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
        )
        existing_call = state.call_id_fingerprints.get(call.call_id)
        if existing_call is not None and existing_call != raw_fingerprint:
            raise AgentProposalError(
                f"call_id {call.call_id!r} was reused with different arguments"
            )
        state.call_id_fingerprints[call.call_id] = raw_fingerprint

        if call.tool_name == "set_control_action":
            state.action_proposals += 1

        request = self._build_request(state, call)
        response_type = _REQUEST_RESPONSE_MODELS[call.tool_name][1]

        side_effect_fingerprint: str | None = None
        if call.tool_name in _SIDE_EFFECTING_TOOLS:
            side_effect_fingerprint = self._side_effect_fingerprint(
                call.tool_name,
                request,
            )
            cached = state.side_effect_cache.get(side_effect_fingerprint)
            if cached is not None:
                self._trace(
                    state,
                    round_number,
                    AgentTraceKind.DUPLICATE_PREVENTED,
                    call.tool_name,
                    "cached",
                    "An identical side effect was not executed twice.",
                )
                response = cached.model_copy(
                    deep=True,
                    update={"request_id": request.request_id},
                )
                return await self._apply_response(
                    client,
                    state,
                    call.tool_name,
                    response,
                    round_number,
                )

        response = await self._invoke(
            client,
            state,
            call.tool_name,
            request,
            response_type,
            round_number,
        )
        if (
            side_effect_fingerprint is not None
            and self._is_cacheable_side_effect(call.tool_name, response)
        ):
            state.side_effect_cache[side_effect_fingerprint] = (
                response.model_copy(deep=True)
            )
        return await self._apply_response(
            client,
            state,
            call.tool_name,
            response,
            round_number,
        )

    def _build_request(
        self,
        state: _CycleState,
        call: ScriptedToolCall,
    ) -> BaseModel:
        allowed_fields = _PROVIDER_ARGUMENT_FIELDS[call.tool_name]
        unexpected = set(call.arguments) - allowed_fields
        if unexpected:
            raise AgentProposalError(
                f"{call.tool_name} arguments contain unsupported fields"
            )

        request_id = self._next_request_id(
            state,
            call.tool_name,
        )
        arguments = copy.deepcopy(call.arguments)

        try:
            if call.tool_name == "read_sensor_data":
                return ReadSensorDataRequest.model_validate(
                    {"request_id": request_id, **arguments}
                )

            snapshot = state.snapshot
            if snapshot is None:
                raise AgentProposalError(
                    f"{call.tool_name} requires read_sensor_data first"
                )

            if call.tool_name == "get_grid_carbon_intensity":
                return GridCarbonIntensityRequest.model_validate(
                    {
                        "request_id": request_id,
                        "snapshot_id": snapshot.snapshot_id,
                        **arguments,
                    }
                )
            if call.tool_name == "parse_runtime_errors":
                return ParseRuntimeErrorsRequest.model_validate(
                    {
                        "request_id": request_id,
                        "cycle_id": snapshot.cycle_id,
                        **arguments,
                    }
                )
            if call.tool_name == "log_reasoning":
                if state.carbon is None:
                    raise AgentProposalError(
                        "log_reasoning requires a correlated carbon signal"
                    )
                return LogReasoningRequest.model_validate(
                    {
                        "request_id": request_id,
                        "cycle_id": snapshot.cycle_id,
                        "snapshot_id": snapshot.snapshot_id,
                        **arguments,
                    }
                )
            if call.tool_name == "set_control_action":
                if state.reasoning is None:
                    raise AgentProposalError(
                        "set_control_action requires a correlated reasoning log"
                    )
                return SetControlActionRequest.model_validate(
                    {
                        "request_id": request_id,
                        "cycle_id": snapshot.cycle_id,
                        "snapshot_id": snapshot.snapshot_id,
                        "reasoning_log_id": (
                            state.reasoning.reasoning_log_id
                        ),
                        "idempotency_key": self._action_idempotency_key(
                            state,
                            snapshot=snapshot,
                            arguments=arguments,
                        ),
                        **arguments,
                    }
                )
        except AgentProposalError:
            raise
        except Exception as exc:
            raise AgentProposalError(
                f"{call.tool_name} arguments failed canonical validation"
            ) from exc

        raise AgentProposalError(f"Unsupported tool {call.tool_name!r}")

    async def _invoke(
        self,
        client: Phase2MCPClient,
        state: _CycleState,
        tool_name: str,
        request: BaseModel,
        response_type: type[ResponseT],
        round_number: int,
    ) -> ResponseT:
        if state.tool_calls_used >= state.limits.max_tool_calls:
            raise AgentToolCallLimitError(
                "maximum MCP tool calls reached"
            )
        self._validate_discovered_tool_schema(state, tool_name)

        state.tool_calls_used += 1
        state.tool_sequence.append(tool_name)
        self._trace(
            state,
            round_number,
            AgentTraceKind.TOOL_CALL,
            tool_name,
            "executed",
            f"MCP call {state.tool_calls_used} of "
            f"{state.limits.max_tool_calls}.",
        )
        try:
            payload = await client.call_request(tool_name, request)
        except MCPToolInvocationError as exc:
            if tool_name in _SIDE_EFFECTING_TOOLS:
                raise MCPIndeterminateOutcomeError(
                    f"{tool_name} returned an execution error after it may "
                    "have written state"
                ) from exc
            raise
        try:
            response = response_type.model_validate(payload)
        except Exception as exc:
            if tool_name in _SIDE_EFFECTING_TOOLS:
                raise MCPIndeterminateOutcomeError(
                    f"{tool_name} returned malformed structured data after "
                    "it may have executed"
                ) from exc
            raise MCPProtocolError(
                f"{tool_name} returned malformed structured data"
            ) from exc
        self._trace(
            state,
            round_number,
            AgentTraceKind.TOOL_RESULT,
            tool_name,
            "ok",
            "MCP returned a validated phase2.v1 response.",
        )
        return response

    async def _apply_response(
        self,
        client: Phase2MCPClient,
        state: _CycleState,
        tool_name: str,
        response: BaseModel,
        round_number: int,
    ) -> str:
        if tool_name == "read_sensor_data":
            sensor = ReadSensorDataResponse.model_validate(response)
            previous_id = (
                state.snapshot.snapshot_id
                if state.snapshot is not None
                else None
            )
            self._validate_snapshot_zones(sensor.snapshot)
            state.snapshot = sensor.snapshot
            if previous_id != sensor.snapshot.snapshot_id:
                state.carbon = None
                state.reasoning = None
            return "continue"

        if tool_name == "get_grid_carbon_intensity":
            carbon = GridCarbonIntensityResponse.model_validate(response)
            if (
                state.snapshot is None
                or carbon.snapshot_id != state.snapshot.snapshot_id
            ):
                raise MCPProtocolError(
                    "carbon response lost snapshot correlation"
                )
            state.carbon = carbon
            return "continue"

        if tool_name == "parse_runtime_errors":
            runtime = ParseRuntimeErrorsResponse.model_validate(response)
            state.runtime_errors = runtime
            for error in runtime.errors:
                if error.error_id not in state.runtime_error_ids:
                    state.runtime_error_ids.append(error.error_id)
                    state.runtime_error_records.append(
                        error.model_copy(deep=True)
                    )
                if error.code not in state.observed_error_codes:
                    state.observed_error_codes.append(error.code)
                if error.severity in (
                    RuntimeErrorSeverity.SEVERE,
                    RuntimeErrorSeverity.FATAL,
                ):
                    state.runtime_blocking_seen = True
            return "continue"

        if tool_name == "log_reasoning":
            reasoning = LogReasoningResponse.model_validate(response)
            state.reasoning = reasoning
            if reasoning.reasoning_log_id not in state.reasoning_log_ids:
                state.reasoning_log_ids.append(
                    reasoning.reasoning_log_id
                )
            return "continue"

        action = SetControlActionResponse.model_validate(response)
        state.last_action = action
        state.last_feedback = (
            "action_rejected"
            if action.status is ControlActionStatus.REJECTED
            else None
        )
        state.last_error_codes = tuple(
            error.code.value for error in action.errors
        )
        for error_code in state.last_error_codes:
            if error_code not in state.observed_error_codes:
                state.observed_error_codes.append(error_code)
        self._trace(
            state,
            round_number,
            AgentTraceKind.TOOL_RESULT,
            "set_control_action",
            action.status.value,
            (
                "Action response contained no safety errors."
                if not state.last_error_codes
                else "Action safety errors: "
                + ",".join(state.last_error_codes)
            ),
        )
        if action.status in (
            ControlActionStatus.ACCEPTED,
            ControlActionStatus.DUPLICATE,
        ):
            self._set_terminal(
                state,
                AgentTerminalStatus.ACCEPTED,
                "The validated five-zone control action was accepted.",
            )
            return "terminal"

        error_codes = {error.code for error in action.errors}
        if any(not error.retryable for error in action.errors):
            self._set_terminal(
                state,
                AgentTerminalStatus.FAILED,
                "The server returned a non-retryable structured action "
                "rejection; no correction or fallback was attempted.",
            )
            return "accepted"
        if SafetyErrorCode.RUNTIME_ERROR_PENDING in error_codes:
            state.runtime_blocking_seen = True
            await self._parse_runtime_errors(
                client,
                state,
                round_number,
            )
            if state.corrected_action_proposals >= (
                state.limits.max_corrected_action_proposals
            ):
                return "runtime_blocked"
        if SafetyErrorCode.STALE_SNAPSHOT in error_codes:
            await self._refresh_after_stale(
                client,
                state,
                round_number,
            )

        if state.corrected_action_proposals >= (
            state.limits.max_corrected_action_proposals
        ):
            if state.runtime_blocking_seen:
                return "runtime_blocked"
            return "fallback"
        return "continue"

    async def _refresh_after_stale(
        self,
        client: Phase2MCPClient,
        state: _CycleState,
        round_number: int,
    ) -> None:
        sensor_request = ReadSensorDataRequest(
            request_id=self._next_request_id(
                state,
                "read_sensor_data",
            ),
            history_steps=2,
        )
        sensor = await self._invoke(
            client,
            state,
            "read_sensor_data",
            sensor_request,
            ReadSensorDataResponse,
            round_number,
        )
        await self._apply_response(
            client,
            state,
            "read_sensor_data",
            sensor,
            round_number,
        )
        if state.snapshot is None:
            raise MCPProtocolError("stale refresh returned no snapshot")
        carbon_request = GridCarbonIntensityRequest(
            request_id=self._next_request_id(
                state,
                "get_grid_carbon_intensity",
            ),
            snapshot_id=state.snapshot.snapshot_id,
            forecast_steps=4,
        )
        carbon = await self._invoke(
            client,
            state,
            "get_grid_carbon_intensity",
            carbon_request,
            GridCarbonIntensityResponse,
            round_number,
        )
        await self._apply_response(
            client,
            state,
            "get_grid_carbon_intensity",
            carbon,
            round_number,
        )

    async def _parse_runtime_errors(
        self,
        client: Phase2MCPClient,
        state: _CycleState,
        round_number: int,
    ) -> None:
        if state.snapshot is None:
            raise MCPProtocolError(
                "runtime error parsing requires a snapshot"
            )
        after_error_id: str | None = None
        while True:
            request = ParseRuntimeErrorsRequest(
                request_id=self._next_request_id(
                    state,
                    "parse_runtime_errors",
                ),
                cycle_id=state.snapshot.cycle_id,
                after_error_id=after_error_id,
                limit=20,
            )
            response = await self._invoke(
                client,
                state,
                "parse_runtime_errors",
                request,
                ParseRuntimeErrorsResponse,
                round_number,
            )
            await self._apply_response(
                client,
                state,
                "parse_runtime_errors",
                response,
                round_number,
            )
            if not response.has_more:
                return
            if response.next_error_id is None:
                raise MCPProtocolError(
                    "runtime error pagination omitted its next cursor"
                )
            after_error_id = response.next_error_id

    async def _apply_fallback(
        self,
        client: Phase2MCPClient,
        state: _CycleState,
        *,
        round_number: int,
        reason: str,
    ) -> None:
        if state.terminal:
            return
        if state.runtime_blocking_seen:
            self._set_terminal(
                state,
                AgentTerminalStatus.RUNTIME_BLOCKED,
                "Blocking runtime errors prevent a safe control write; "
                "baseline control remains active.",
            )
            return
        if state.snapshot is None:
            self._set_terminal(
                state,
                AgentTerminalStatus.FAILED,
                "No validated sensor snapshot was available for fallback.",
            )
            return
        self._validate_snapshot_zones(state.snapshot)
        if state.tool_calls_used + 2 > state.limits.max_tool_calls:
            self._set_terminal(
                state,
                AgentTerminalStatus.LIMIT_EXHAUSTED,
                "The tool-call limit left insufficient budget for the "
                "two-call safe fallback.",
            )
            return

        state.fallback_used = True
        self._trace(
            state,
            round_number,
            AgentTraceKind.FALLBACK,
            "set_control_action",
            "started",
            reason,
        )
        reasoning_request = LogReasoningRequest(
            request_id=self._next_request_id(
                state,
                "log_reasoning",
            ),
            cycle_id=state.snapshot.cycle_id,
            snapshot_id=state.snapshot.snapshot_id,
            decision_summary=(
                "Release all five zones to deterministic baseline control "
                "because bounded action proposals were exhausted."
            ),
            objective_tags=("safety", "thermal_comfort"),
            tradeoff_summary=(
                "Prefer proven baseline schedules over an unvalidated "
                "thermostat override."
            ),
            confidence=1.0,
        )
        reasoning = await self._invoke(
            client,
            state,
            "log_reasoning",
            reasoning_request,
            LogReasoningResponse,
            round_number,
        )
        await self._apply_response(
            client,
            state,
            "log_reasoning",
            reasoning,
            round_number,
        )
        if state.reasoning is None:
            raise MCPProtocolError(
                "fallback reasoning response was not retained"
            )

        action_request = SetControlActionRequest(
            request_id=self._next_request_id(
                state,
                "set_control_action",
            ),
            cycle_id=state.snapshot.cycle_id,
            snapshot_id=state.snapshot.snapshot_id,
            reasoning_log_id=state.reasoning.reasoning_log_id,
            idempotency_key=self._fallback_idempotency_key(state),
            commands=tuple(
                ReleaseZoneCommand(zone_id=zone_id)
                for zone_id in PHASE1_ZONE_IDS
            ),
            hold_steps=1,
        )
        action = await self._invoke(
            client,
            state,
            "set_control_action",
            action_request,
            SetControlActionResponse,
            round_number,
        )
        state.last_action = action
        state.last_error_codes = tuple(
            error.code.value for error in action.errors
        )
        for error_code in state.last_error_codes:
            if error_code not in state.observed_error_codes:
                state.observed_error_codes.append(error_code)
        self._trace(
            state,
            round_number,
            AgentTraceKind.TOOL_RESULT,
            "set_control_action",
            action.status.value,
            "Deterministic release fallback response received.",
        )
        if action.status in (
            ControlActionStatus.ACCEPTED,
            ControlActionStatus.DUPLICATE,
        ):
            self._set_terminal(
                state,
                AgentTerminalStatus.FALLBACK_ACCEPTED,
                "The deterministic five-zone release fallback was accepted.",
            )
            return
        self._set_terminal(
            state,
            AgentTerminalStatus.FAILED,
            "The deterministic release fallback was rejected; no further "
            "write was attempted.",
        )

    def _build_observation(
        self,
        state: _CycleState,
        round_number: int,
    ) -> ProviderObservation:
        return ProviderObservation(
            run_id=state.run_id,
            round_number=round_number,
            discovered_tool_names=state.discovered_tool_names,
            discovered_tool_schemas={
                name: copy.deepcopy(tool.inputSchema)
                for name, tool in state.discovered_tools.items()
            },
            cycle_id=(
                state.snapshot.cycle_id
                if state.snapshot is not None
                else None
            ),
            snapshot_id=(
                state.snapshot.snapshot_id
                if state.snapshot is not None
                else None
            ),
            sensor_snapshot=(
                state.snapshot.model_copy(deep=True)
                if state.snapshot is not None
                else None
            ),
            carbon_signal=(
                state.carbon.model_copy(deep=True)
                if state.carbon is not None
                else None
            ),
            reasoning_log_id=(
                state.reasoning.reasoning_log_id
                if state.reasoning is not None
                else None
            ),
            last_action_status=(
                state.last_action.status.value
                if state.last_action is not None
                else None
            ),
            last_error_codes=state.last_error_codes,
            last_action_errors=(
                state.last_action.errors
                if state.last_action is not None
                else ()
            ),
            last_feedback=state.last_feedback,
            runtime_errors=tuple(
                RuntimeErrorObservation(
                    error_id=error.error_id,
                    severity=error.severity.value,
                    code=error.code,
                    summary=error.summary,
                    retryable=error.retryable,
                    correction_fields=error.correction_fields,
                    correction_hint=error.correction_hint,
                )
                for error in state.runtime_error_records[-20:]
            ),
            runtime_errors_has_more=(
                state.runtime_errors.has_more
                if state.runtime_errors is not None
                else False
            ),
            next_runtime_error_id=(
                state.runtime_errors.next_error_id
                if state.runtime_errors is not None
                else None
            ),
            tool_calls_remaining=(
                state.limits.max_tool_calls - state.tool_calls_used
            ),
            corrected_action_proposals_remaining=max(
                0,
                state.limits.max_corrected_action_proposals
                - state.corrected_action_proposals,
            ),
        )

    def _next_request_id(
        self,
        state: _CycleState,
        tool_name: str,
    ) -> str:
        state.request_sequence += 1
        return (
            f"{state.run_id}:{tool_name}:"
            f"{state.request_sequence:02d}"
        )

    def _side_effect_fingerprint(
        self,
        tool_name: str,
        request: BaseModel,
    ) -> str:
        payload = request.model_dump(mode="json")
        payload.pop("request_id", None)
        if tool_name == "set_control_action":
            payload.pop("idempotency_key", None)
        return _canonical_json(
            {"tool_name": tool_name, "request": payload}
        )

    def _is_cacheable_side_effect(
        self,
        tool_name: str,
        response: BaseModel,
    ) -> bool:
        if tool_name == "log_reasoning":
            return isinstance(response, LogReasoningResponse)
        if tool_name == "set_control_action":
            action = SetControlActionResponse.model_validate(response)
            return action.status in (
                ControlActionStatus.ACCEPTED,
                ControlActionStatus.DUPLICATE,
            )
        return False

    def _action_idempotency_key(
        self,
        state: _CycleState,
        *,
        snapshot: SensorSnapshot,
        arguments: Mapping[str, Any],
    ) -> str:
        logical_action = {
            "cycle_id": snapshot.cycle_id,
            "snapshot_id": snapshot.snapshot_id,
            "reasoning_log_id": (
                state.reasoning.reasoning_log_id
                if state.reasoning is not None
                else None
            ),
            **copy.deepcopy(dict(arguments)),
        }
        digest = hashlib.sha256(
            _canonical_json(logical_action).encode("utf-8")
        ).hexdigest()[:24]
        return f"{state.run_id}:action:{digest}"

    def _fallback_idempotency_key(
        self,
        state: _CycleState,
    ) -> str:
        assert state.snapshot is not None
        digest = hashlib.sha256(
            (
                f"{state.snapshot.cycle_id}:"
                f"{state.snapshot.snapshot_id}:release"
            ).encode("utf-8")
        ).hexdigest()[:24]
        return f"{state.run_id}:fallback:{digest}"

    def _validate_discovered_tools(self, state: _CycleState) -> None:
        if (
            len(state.discovered_tool_names) != len(PHASE2_TOOL_NAMES)
            or set(state.discovered_tool_names) != set(PHASE2_TOOL_NAMES)
        ):
            raise MCPProtocolError(
                "Agent requires the exact five Phase 2 MCP tools"
            )
        for tool_name in PHASE2_TOOL_NAMES:
            self._validate_discovered_tool_schema(state, tool_name)

    def _validate_discovered_tool_schema(
        self,
        state: _CycleState,
        tool_name: str,
    ) -> None:
        tool = state.discovered_tools.get(tool_name)
        if tool is None:
            raise MCPProtocolError(
                f"Discovered schema missing {tool_name!r}"
            )
        schema = tool.inputSchema
        if (
            set(schema.get("properties", {})) != {"request"}
            or schema.get("required") != ["request"]
            or schema.get("additionalProperties") is not False
        ):
            raise MCPProtocolError(
                f"{tool_name} advertised an unsafe input schema"
            )

    def _validate_snapshot_zones(
        self,
        snapshot: SensorSnapshot,
    ) -> None:
        zone_ids = tuple(zone.zone_id for zone in snapshot.zones)
        if len(zone_ids) != 5 or set(zone_ids) != set(PHASE1_ZONE_IDS):
            raise MCPProtocolError(
                "sensor snapshot does not contain the five Phase 1 zones"
            )

    def _trace(
        self,
        state: _CycleState,
        round_number: int,
        kind: AgentTraceKind,
        tool_name: str | None,
        status: str,
        detail: str,
    ) -> None:
        state.trace_sequence += 1
        state.trace.append(
            AgentTraceEvent(
                sequence=state.trace_sequence,
                round_number=round_number,
                kind=kind,
                tool_name=tool_name,
                status=status,
                detail=detail,
            )
        )

    def _set_terminal(
        self,
        state: _CycleState,
        status: AgentTerminalStatus,
        summary: str,
        *,
        force: bool = False,
    ) -> None:
        if force or not state.terminal:
            state.terminal_status = status
            state.terminal_summary = summary

    def _build_record(self, state: _CycleState) -> AgentCycleRecord:
        assert state.terminal_status is not None
        return AgentCycleRecord(
            run_id=state.run_id,
            provider_name=state.provider_name,
            terminal_status=state.terminal_status,
            terminal_summary=state.terminal_summary,
            cycle_id=(
                state.snapshot.cycle_id
                if state.snapshot is not None
                else None
            ),
            snapshot_id=(
                state.snapshot.snapshot_id
                if state.snapshot is not None
                else None
            ),
            rounds_used=state.rounds_used,
            tool_calls_used=state.tool_calls_used,
            corrected_action_proposals=(
                state.corrected_action_proposals
            ),
            fallback_used=state.fallback_used,
            action_status=(
                state.last_action.status
                if state.last_action is not None
                else None
            ),
            action_id=(
                state.last_action.action_id
                if state.last_action is not None
                else None
            ),
            reasoning_log_ids=tuple(state.reasoning_log_ids),
            runtime_error_ids=tuple(state.runtime_error_ids),
            error_codes=tuple(state.observed_error_codes),
            discovered_tool_names=state.discovered_tool_names,
            tool_sequence=tuple(state.tool_sequence),
        )


__all__ = [
    "AGENT_RECORD_SCHEMA_VERSION",
    "MAX_AGENT_ROUNDS",
    "MAX_CORRECTED_ACTION_PROPOSALS",
    "MAX_MCP_TOOL_CALLS",
    "AgentCycleRecord",
    "AgentCycleResult",
    "AgentLoopError",
    "AgentLoopLimits",
    "AgentProposalError",
    "AgentTerminalStatus",
    "AgentToolCallLimitError",
    "AgentTraceEvent",
    "AgentTraceKind",
    "Phase2AgentOrchestrator",
]
