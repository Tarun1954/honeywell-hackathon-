"""Real-stdio tests for the deterministic bounded Phase 2 agent loop."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mcp.client.stdio as mcp_stdio
import anyio
from mcp import StdioServerParameters
from pydantic import BaseModel

from scripts.run_phase2_agent_smoke import (
    build_correction_provider,
    build_fallback_provider,
    build_success_provider,
    run_smoke_scenarios,
)
from src.mcp_client import (
    MCPClientBridgeError,
    MCPToolInvocationError,
    Phase2MCPClient,
)
from src.phase2_agent import (
    AgentLoopLimits,
    AgentTerminalStatus,
    AgentTraceKind,
    Phase2AgentOrchestrator,
)
from src.phase2_contracts import (
    ControlActionStatus,
    ReleaseZoneCommand,
    SafetyErrorCode,
    SetControlActionRequest,
    SetControlActionResponse,
    ToolError,
)
from src.phase2_mock_services import PHASE1_ZONE_IDS
from src.scripted_provider import (
    ProviderObservation,
    ScriptedProvider,
    ScriptedProviderExhausted,
    ScriptedToolCall,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def reasoning_arguments(summary: str) -> dict[str, Any]:
    return {
        "decision_summary": summary,
        "objective_tags": [
            "thermal_comfort",
            "energy_reduction",
            "safety",
        ],
        "tradeoff_summary": (
            "Preserve comfort while respecting deterministic safety policy."
        ),
        "confidence": 0.9,
    }


def set_commands(
    heating_c: float,
    cooling_c: float,
) -> list[dict[str, Any]]:
    return [
        {
            "mode": "set",
            "zone_id": zone_id,
            "heating_c": heating_c,
            "cooling_c": cooling_c,
        }
        for zone_id in PHASE1_ZONE_IDS
    ]


def base_calls(prefix: str) -> list[ScriptedToolCall]:
    return [
        ScriptedToolCall(
            call_id=f"{prefix}-read",
            tool_name="read_sensor_data",
            arguments={"history_steps": 1},
        ),
        ScriptedToolCall(
            call_id=f"{prefix}-carbon",
            tool_name="get_grid_carbon_intensity",
            arguments={"forecast_steps": 4},
        ),
        ScriptedToolCall(
            call_id=f"{prefix}-reasoning",
            tool_name="log_reasoning",
            arguments=reasoning_arguments(
                "Apply a deterministic safe five-zone action."
            ),
        ),
    ]


def action_call(
    call_id: str,
    heating_c: float = 20.0,
    cooling_c: float = 26.0,
) -> ScriptedToolCall:
    return ScriptedToolCall(
        call_id=call_id,
        tool_name="set_control_action",
        arguments={
            "commands": set_commands(heating_c, cooling_c),
            "hold_steps": 1,
        },
    )


def fixture_client_factory(scenario: str) -> Callable[[], Phase2MCPClient]:
    def create_client() -> Phase2MCPClient:
        return Phase2MCPClient(
            StdioServerParameters(
                command=sys.executable,
                args=[
                    "-u",
                    "-m",
                    "tests.phase2_agent_fixture_server",
                    scenario,
                ],
                cwd=REPOSITORY_ROOT,
                env=None,
                encoding="utf-8",
                encoding_error_handler="strict",
            ),
            allow_read_reconnect=False,
        )

    return create_client


class ScriptedProviderTests(unittest.IsolatedAsyncioTestCase):
    """Protect provider determinism, reset behavior, and exhaustion."""

    async def test_sequence_is_deterministic_and_resets_per_cycle(
        self,
    ) -> None:
        calls = (
            ScriptedToolCall(
                call_id="provider-read",
                tool_name="read_sensor_data",
                arguments={"history_steps": 0},
            ),
            ScriptedToolCall(
                call_id="provider-carbon",
                tool_name="get_grid_carbon_intensity",
                arguments={"forecast_steps": 1},
            ),
        )
        provider = ScriptedProvider(calls, name="deterministic-provider")
        observation = ProviderObservation(
            run_id="provider-test",
            round_number=1,
            discovered_tool_names=(),
            tool_calls_remaining=12,
            corrected_action_proposals_remaining=2,
        )

        provider.start_cycle()
        first = await provider.next_tool_call(observation)
        second = await provider.next_tool_call(observation)
        with self.assertRaises(ScriptedProviderExhausted):
            await provider.next_tool_call(observation)

        provider.start_cycle()
        repeated = await provider.next_tool_call(observation)

        self.assertEqual((first, second), calls)
        self.assertEqual(repeated, first)
        self.assertEqual(provider.responses_emitted, 1)


class Phase2AgentRealStdioTests(unittest.IsolatedAsyncioTestCase):
    """Run every acceptance path through tracked real MCP subprocesses."""

    def setUp(self) -> None:
        self.processes: list[Any] = []
        self.original_process_factory: Callable[..., Any] = (
            mcp_stdio._create_platform_compatible_process
        )
        self.process_patch = patch(
            "mcp.client.stdio._create_platform_compatible_process",
            new=self._capture_process,
        )
        self.process_patch.start()

    def tearDown(self) -> None:
        self.process_patch.stop()

    async def asyncTearDown(self) -> None:
        orphan_pids: list[int] = []
        for process in self.processes:
            if process.returncode is not None:
                continue
            orphan_pids.append(process.pid)
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=3.0)
        self.assertEqual(
            orphan_pids,
            [],
            f"orphan agent MCP processes required cleanup: {orphan_pids}",
        )

    async def _capture_process(self, *args: Any, **kwargs: Any) -> Any:
        process = await self.original_process_factory(*args, **kwargs)
        self.processes.append(process)
        return process

    def assert_latest_process_clean(self) -> None:
        self.assertTrue(self.processes)
        self.assertEqual(self.processes[-1].returncode, 0)

    async def test_happy_path_calls_exact_sequence_and_is_accepted(
        self,
    ) -> None:
        sink = io.StringIO()
        result = await Phase2AgentOrchestrator().run_cycle(
            build_success_provider(),
            run_id="agent-happy",
            record_stream=sink,
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assertEqual(
            result.record.action_status,
            ControlActionStatus.ACCEPTED,
        )
        self.assertEqual(
            result.record.tool_sequence,
            (
                "read_sensor_data",
                "get_grid_carbon_intensity",
                "log_reasoning",
                "set_control_action",
            ),
        )
        self.assertIsNotNone(result.record.cycle_id)
        self.assertIsNotNone(result.record.snapshot_id)
        self.assertEqual(len(result.record.reasoning_log_ids), 1)
        lines = sink.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            json.loads(lines[0])["terminal_status"],
            "accepted",
        )
        self.assertEqual(result.jsonl.count("\n"), 1)
        self.assert_latest_process_clean()

    async def test_unsafe_action_is_rejected_then_corrected(
        self,
    ) -> None:
        captured_actions: list[SetControlActionRequest] = []

        class CapturingClient(Phase2MCPClient):
            async def call_request(
                self,
                tool_name: str,
                request: BaseModel | Mapping[str, Any],
            ) -> dict[str, Any]:
                if (
                    tool_name == "set_control_action"
                    and isinstance(request, SetControlActionRequest)
                ):
                    captured_actions.append(
                        request.model_copy(deep=True)
                    )
                return await super().call_request(tool_name, request)

        result = await Phase2AgentOrchestrator(
            client_factory=lambda: CapturingClient(
                allow_read_reconnect=False
            )
        ).run_cycle(
            build_correction_provider(),
            run_id="agent-correction",
        )

        outcomes = [
            event.status
            for event in result.trace
            if event.tool_name == "set_control_action"
            and event.status in {"accepted", "rejected", "duplicate"}
        ]
        self.assertEqual(outcomes, ["rejected", "accepted"])
        self.assertEqual(result.record.corrected_action_proposals, 1)
        self.assertIn(
            "OUT_OF_RANGE",
            result.record.error_codes,
        )
        self.assertTrue(
            all(
                getattr(command, "heating_c", None) == 15.0
                and getattr(command, "cooling_c", None) == 31.0
                for command in captured_actions[0].commands
            )
        )
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assertFalse(result.record.fallback_used)
        self.assert_latest_process_clean()

    async def test_malformed_arguments_are_detected_then_corrected(
        self,
    ) -> None:
        calls = base_calls("malformed")
        unsupported_commands = set_commands(20.0, 26.0)
        for command in unsupported_commands:
            command["damper_position"] = 0.5
        calls.extend(
            (
                ScriptedToolCall(
                    call_id="malformed-action",
                    tool_name="set_control_action",
                    arguments={
                        "commands": unsupported_commands,
                        "hold_steps": 1,
                    },
                ),
                action_call("malformed-corrected"),
            )
        )
        result = await Phase2AgentOrchestrator().run_cycle(
            ScriptedProvider(calls, name="malformed-script"),
            run_id="agent-malformed",
        )

        rejected = [
            event
            for event in result.trace
            if event.kind is AgentTraceKind.PROPOSAL_REJECTED
        ]
        self.assertTrue(rejected)
        self.assertEqual(
            result.record.tool_sequence.count("set_control_action"),
            1,
        )
        self.assertEqual(result.record.corrected_action_proposals, 1)
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assert_latest_process_clean()

    async def test_unknown_tool_is_rejected_without_execution(
        self,
    ) -> None:
        calls = [
            ScriptedToolCall(
                call_id="unknown-call",
                tool_name="set_control_actoin",
                arguments={},
            ),
            *base_calls("unknown"),
            action_call("unknown-safe"),
        ]
        result = await Phase2AgentOrchestrator().run_cycle(
            ScriptedProvider(calls, name="unknown-tool-script"),
            run_id="agent-unknown",
        )

        self.assertNotIn(
            "set_control_actoin",
            result.record.tool_sequence,
        )
        self.assertTrue(
            any(
                event.kind is AgentTraceKind.PROPOSAL_REJECTED
                and event.tool_name == "set_control_actoin"
                for event in result.trace
            )
        )
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assert_latest_process_clean()

    async def test_hidden_chain_of_thought_field_is_rejected_locally(
        self,
    ) -> None:
        hidden_sentinel = "SECRET_INTERNAL_STEP_DO_NOT_EXPOSE"
        calls = base_calls("hidden-reasoning")[:2]
        calls.extend(
            (
                ScriptedToolCall(
                    call_id="hidden-reasoning-rejected",
                    tool_name="log_reasoning",
                    arguments={
                        **reasoning_arguments(
                            "Use only an auditable decision summary."
                        ),
                        "chain_of_thought": hidden_sentinel,
                    },
                ),
                ScriptedToolCall(
                    call_id="hidden-reasoning-safe",
                    tool_name="log_reasoning",
                    arguments=reasoning_arguments(
                        "Use only an auditable decision summary."
                    ),
                ),
                action_call("hidden-reasoning-action"),
            )
        )
        result = await Phase2AgentOrchestrator().run_cycle(
            ScriptedProvider(calls, name="hidden-reasoning-script"),
            run_id="agent-hidden-reasoning",
        )

        self.assertEqual(
            result.record.tool_sequence.count("log_reasoning"),
            1,
        )
        self.assertTrue(
            any(
                event.kind is AgentTraceKind.PROPOSAL_REJECTED
                and event.tool_name == "log_reasoning"
                for event in result.trace
            )
        )
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        exposed_output = result.jsonl + "".join(
            event.model_dump_json() for event in result.trace
        )
        self.assertNotIn(hidden_sentinel, exposed_output)
        self.assertNotIn("chain_of_thought", exposed_output)
        self.assert_latest_process_clean()

    async def test_stale_snapshot_refreshes_then_retries_action(
        self,
    ) -> None:
        calls = base_calls("stale")
        calls.extend(
            (
                action_call("stale-first-action"),
                ScriptedToolCall(
                    call_id="stale-refreshed-reasoning",
                    tool_name="log_reasoning",
                    arguments=reasoning_arguments(
                        "Use refreshed snapshot correlation after staleness."
                    ),
                ),
                action_call("stale-refreshed-action"),
            )
        )
        result = await Phase2AgentOrchestrator(
            client_factory=fixture_client_factory("stale_once")
        ).run_cycle(
            ScriptedProvider(calls, name="stale-script"),
            run_id="agent-stale",
        )

        self.assertEqual(
            result.record.tool_sequence.count("read_sensor_data"),
            2,
        )
        self.assertEqual(
            result.record.tool_sequence.count(
                "get_grid_carbon_intensity"
            ),
            2,
        )
        self.assertEqual(
            result.record.tool_sequence.count("set_control_action"),
            2,
        )
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assertEqual(result.record.corrected_action_proposals, 1)
        self.assert_latest_process_clean()

    async def test_runtime_error_is_parsed_before_corrected_action(
        self,
    ) -> None:
        calls = base_calls("runtime")[:2]
        calls.extend(
            (
                ScriptedToolCall(
                    call_id="runtime-parse",
                    tool_name="parse_runtime_errors",
                    arguments={"limit": 20},
                ),
                ScriptedToolCall(
                    call_id="runtime-reasoning",
                    tool_name="log_reasoning",
                    arguments=reasoning_arguments(
                        "Release every zone after reviewing the runtime hint."
                    ),
                ),
                ScriptedToolCall(
                    call_id="runtime-corrected-action",
                    tool_name="set_control_action",
                    arguments={
                        "commands": [
                            {"mode": "release", "zone_id": zone_id}
                            for zone_id in PHASE1_ZONE_IDS
                        ],
                        "hold_steps": 1,
                    },
                ),
            )
        )

        class ObservationCapturingProvider(ScriptedProvider):
            def __init__(self) -> None:
                super().__init__(calls, name="runtime-script")
                self.observations: list[ProviderObservation] = []

            async def next_tool_call(
                self,
                observation: ProviderObservation,
            ) -> ScriptedToolCall:
                self.observations.append(observation)
                return await super().next_tool_call(observation)

        provider = ObservationCapturingProvider()
        result = await Phase2AgentOrchestrator(
            client_factory=fixture_client_factory("runtime_error")
        ).run_cycle(
            provider,
            run_id="agent-runtime",
        )

        tool_sequence = result.record.tool_sequence
        self.assertIn("parse_runtime_errors", tool_sequence)
        parse_index = tool_sequence.index("parse_runtime_errors")
        action_index = tool_sequence.index("set_control_action")
        self.assertLess(parse_index, action_index)
        self.assertEqual(len(result.record.runtime_error_ids), 1)
        self.assertIn(
            "TRANSIENT_SETPOINT_WARNING",
            result.record.error_codes,
        )
        runtime_observations = [
            observation
            for observation in provider.observations
            if observation.runtime_errors
        ]
        self.assertTrue(runtime_observations)
        self.assertIsNotNone(
            runtime_observations[0].sensor_snapshot
        )
        self.assertIsNotNone(runtime_observations[0].carbon_signal)
        self.assertEqual(
            runtime_observations[0].runtime_errors[0].correction_fields,
            ("commands",),
        )
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assert_latest_process_clean()

    async def test_severe_runtime_error_parses_then_blocks_more_writes(
        self,
    ) -> None:
        calls = [*base_calls("runtime-blocking")]
        calls.append(action_call("runtime-blocking-action"))
        result = await Phase2AgentOrchestrator(
            client_factory=fixture_client_factory("runtime_blocking")
        ).run_cycle(
            ScriptedProvider(calls, name="runtime-blocking-script"),
            run_id="agent-runtime-blocking",
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.RUNTIME_BLOCKED,
        )
        self.assertEqual(
            result.record.tool_sequence.count("set_control_action"),
            1,
        )
        self.assertEqual(
            result.record.tool_sequence.count("parse_runtime_errors"),
            1,
        )
        self.assertIn(
            "RUNTIME_ERROR_PENDING",
            result.record.error_codes,
        )
        self.assertIn(
            "ACTUATOR_WRITEBACK_FAILED",
            result.record.error_codes,
        )
        self.assertFalse(result.record.fallback_used)
        self.assert_latest_process_clean()

    async def test_retry_exhaustion_applies_release_fallback(
        self,
    ) -> None:
        captured_actions: list[SetControlActionRequest] = []

        class CapturingClient(Phase2MCPClient):
            async def call_request(
                self,
                tool_name: str,
                request: BaseModel | Mapping[str, Any],
            ) -> dict[str, Any]:
                if (
                    tool_name == "set_control_action"
                    and isinstance(request, SetControlActionRequest)
                ):
                    captured_actions.append(
                        request.model_copy(deep=True)
                    )
                return await super().call_request(tool_name, request)

        result = await Phase2AgentOrchestrator(
            client_factory=lambda: CapturingClient(
                allow_read_reconnect=False
            )
        ).run_cycle(
            build_fallback_provider(),
            run_id="agent-fallback",
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.FALLBACK_ACCEPTED,
        )
        self.assertTrue(result.record.fallback_used)
        self.assertEqual(result.record.corrected_action_proposals, 2)
        self.assertIn("OUT_OF_RANGE", result.record.error_codes)
        self.assertIn(
            "DEADBAND_VIOLATION",
            result.record.error_codes,
        )
        fallback = captured_actions[-1]
        self.assertTrue(
            all(
                isinstance(command, ReleaseZoneCommand)
                for command in fallback.commands
            )
        )
        self.assertEqual(
            {command.zone_id for command in fallback.commands},
            set(PHASE1_ZONE_IDS),
        )
        self.assertEqual(len(fallback.commands), 5)
        self.assert_latest_process_clean()

    async def test_duplicate_side_effects_are_not_executed_twice(
        self,
    ) -> None:
        calls = base_calls("duplicate")[:2]
        duplicate_reasoning = reasoning_arguments(
            "Use one auditable summary despite an identical replay."
        )
        calls.extend(
            (
                ScriptedToolCall(
                    call_id="duplicate-reasoning-1",
                    tool_name="log_reasoning",
                    arguments=duplicate_reasoning,
                ),
                ScriptedToolCall(
                    call_id="duplicate-reasoning-2",
                    tool_name="log_reasoning",
                    arguments=duplicate_reasoning,
                ),
                action_call("duplicate-safe"),
                action_call("duplicate-safe-replay"),
            )
        )
        provider = ScriptedProvider(calls, name="duplicate-script")
        result = await Phase2AgentOrchestrator().run_cycle(
            provider,
            run_id="agent-duplicate",
        )

        self.assertTrue(
            any(
                event.kind is AgentTraceKind.DUPLICATE_PREVENTED
                for event in result.trace
            )
        )
        self.assertEqual(
            result.record.tool_sequence.count("log_reasoning"),
            1,
        )
        self.assertEqual(
            result.record.tool_sequence.count("set_control_action"),
            1,
        )
        self.assertEqual(provider.responses_emitted, 5)
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assert_latest_process_clean()

    async def test_identical_retry_reuses_the_same_idempotency_key(
        self,
    ) -> None:
        action_requests: list[SetControlActionRequest] = []

        class TransientRuntimeClient(Phase2MCPClient):
            async def call_request(
                self,
                tool_name: str,
                request: BaseModel | Mapping[str, Any],
            ) -> dict[str, Any]:
                if (
                    tool_name == "set_control_action"
                    and isinstance(request, SetControlActionRequest)
                ):
                    action_requests.append(
                        request.model_copy(deep=True)
                    )
                    if len(action_requests) == 1:
                        return SetControlActionResponse(
                            request_id=request.request_id,
                            cycle_id=request.cycle_id,
                            snapshot_id=request.snapshot_id,
                            status=ControlActionStatus.REJECTED,
                            errors=(
                                ToolError(
                                    code=(
                                        SafetyErrorCode.RUNTIME_ERROR_PENDING
                                    ),
                                    field="commands",
                                    message=(
                                        "A transient mock runtime condition "
                                        "requires one corrected proposal."
                                    ),
                                    retryable=True,
                                ),
                            ),
                        ).model_dump(mode="json")
                return await super().call_request(tool_name, request)

        calls = [*base_calls("stable-idempotency")]
        calls.extend(
            (
                action_call("stable-idempotency-first"),
                action_call("stable-idempotency-retry"),
            )
        )
        result = await Phase2AgentOrchestrator(
            client_factory=lambda: TransientRuntimeClient(
                allow_read_reconnect=False
            )
        ).run_cycle(
            ScriptedProvider(calls, name="stable-idempotency-script"),
            run_id="agent-stable-idempotency",
        )

        self.assertEqual(len(action_requests), 2)
        self.assertEqual(
            action_requests[0].idempotency_key,
            action_requests[1].idempotency_key,
        )
        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.ACCEPTED,
        )
        self.assertEqual(
            result.record.tool_sequence.count("parse_runtime_errors"),
            1,
        )
        self.assert_latest_process_clean()

    async def test_post_write_tool_error_is_indeterminate_and_not_retried(
        self,
    ) -> None:
        action_calls = 0

        class PostWriteErrorClient(Phase2MCPClient):
            async def call_request(
                self,
                tool_name: str,
                request: BaseModel | Mapping[str, Any],
            ) -> dict[str, Any]:
                nonlocal action_calls
                response = await super().call_request(tool_name, request)
                if tool_name == "set_control_action":
                    action_calls += 1
                    raise MCPToolInvocationError(
                        "simulated error after the accepted write"
                    )
                return response

        calls = [*base_calls("post-write-error")]
        calls.extend(
            (
                action_call("post-write-error-first"),
                action_call("post-write-error-retry"),
            )
        )
        provider = ScriptedProvider(calls, name="post-write-error-script")
        result = await Phase2AgentOrchestrator(
            client_factory=lambda: PostWriteErrorClient(
                allow_read_reconnect=False
            )
        ).run_cycle(
            provider,
            run_id="agent-post-write-error",
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.INDETERMINATE,
        )
        self.assertEqual(action_calls, 1)
        self.assertEqual(
            result.record.tool_sequence.count("set_control_action"),
            1,
        )
        self.assertFalse(result.record.fallback_used)
        self.assertEqual(provider.responses_emitted, 4)
        self.assert_latest_process_clean()

    async def test_malformed_post_write_response_is_indeterminate(
        self,
    ) -> None:
        reasoning_calls = 0

        class MalformedPostWriteClient(Phase2MCPClient):
            async def call_request(
                self,
                tool_name: str,
                request: BaseModel | Mapping[str, Any],
            ) -> dict[str, Any]:
                nonlocal reasoning_calls
                response = await super().call_request(tool_name, request)
                if tool_name == "log_reasoning":
                    reasoning_calls += 1
                    response.pop("reasoning_log_id", None)
                return response

        result = await Phase2AgentOrchestrator(
            client_factory=lambda: MalformedPostWriteClient(
                allow_read_reconnect=False
            )
        ).run_cycle(
            build_success_provider(),
            run_id="agent-malformed-post-write",
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.INDETERMINATE,
        )
        self.assertEqual(reasoning_calls, 1)
        self.assertNotIn(
            "set_control_action",
            result.record.tool_sequence,
        )
        self.assertFalse(result.record.fallback_used)
        self.assert_latest_process_clean()

    async def test_provider_timeout_is_bounded_and_closes_stdio(
        self,
    ) -> None:
        class HangingProvider(ScriptedProvider):
            async def next_tool_call(
                self,
                observation: ProviderObservation,
            ) -> ScriptedToolCall:
                await anyio.sleep_forever()
                raise AssertionError("unreachable")

        provider = HangingProvider([], name="hanging-script")
        result = await Phase2AgentOrchestrator(
            limits=AgentLoopLimits(
                provider_response_timeout_seconds=0.05
            )
        ).run_cycle(
            provider,
            run_id="agent-provider-timeout",
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.FAILED,
        )
        self.assertEqual(result.record.rounds_used, 0)
        self.assertEqual(result.record.tool_calls_used, 0)
        self.assert_latest_process_clean()

    async def test_cleanup_failure_overrides_an_accepted_terminal(
        self,
    ) -> None:
        class CleanupFailingClient(Phase2MCPClient):
            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: Any,
            ) -> None:
                await super().__aexit__(exc_type, exc, traceback)
                raise MCPClientBridgeError(
                    "simulated lifecycle failure after cleanup"
                )

        result = await Phase2AgentOrchestrator(
            client_factory=lambda: CleanupFailingClient(
                allow_read_reconnect=False
            )
        ).run_cycle(
            build_success_provider(),
            run_id="agent-cleanup-failure",
        )

        self.assertEqual(
            result.record.terminal_status,
            AgentTerminalStatus.FAILED,
        )
        self.assertEqual(
            result.record.action_status,
            ControlActionStatus.ACCEPTED,
        )
        self.assert_latest_process_clean()

    async def test_external_cancellation_propagates_and_closes_stdio(
        self,
    ) -> None:
        provider_entered = asyncio.Event()

        class CancellableProvider(ScriptedProvider):
            async def next_tool_call(
                self,
                observation: ProviderObservation,
            ) -> ScriptedToolCall:
                provider_entered.set()
                await anyio.sleep_forever()
                raise AssertionError("unreachable")

        cycle_task = asyncio.create_task(
            Phase2AgentOrchestrator().run_cycle(
                CancellableProvider([], name="cancellable-script"),
                run_id="agent-cancelled",
            )
        )
        await asyncio.wait_for(provider_entered.wait(), timeout=5.0)
        cycle_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cycle_task

        self.assert_latest_process_clean()

    async def test_round_and_tool_call_limits_are_enforced(
        self,
    ) -> None:
        round_provider = ScriptedProvider(
            [
                ScriptedToolCall(
                    call_id=f"round-read-{index}",
                    tool_name="read_sensor_data",
                    arguments={"history_steps": 0},
                )
                for index in range(1, 11)
            ],
            name="round-limit-script",
        )
        round_result = await Phase2AgentOrchestrator().run_cycle(
            round_provider,
            run_id="agent-round-limit",
        )

        self.assertEqual(round_result.record.rounds_used, 6)
        self.assertEqual(round_provider.responses_emitted, 6)
        self.assertLessEqual(round_result.record.tool_calls_used, 12)
        self.assertEqual(
            round_result.record.terminal_status,
            AgentTerminalStatus.FALLBACK_ACCEPTED,
        )
        self.assert_latest_process_clean()

        tool_result = await Phase2AgentOrchestrator(
            limits=AgentLoopLimits(max_tool_calls=3)
        ).run_cycle(
            build_success_provider(),
            run_id="agent-tool-limit",
        )

        self.assertEqual(tool_result.record.tool_calls_used, 3)
        self.assertEqual(
            tool_result.record.terminal_status,
            AgentTerminalStatus.LIMIT_EXHAUSTED,
        )
        self.assertEqual(len(self.processes), 2)
        self.assertTrue(
            all(process.returncode == 0 for process in self.processes)
        )

    async def test_every_smoke_cycle_shuts_down_without_orphan(
        self,
    ) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            results = await run_smoke_scenarios()

        self.assertEqual(len(results), 3)
        self.assertEqual(
            tuple(result.record.terminal_status for result in results),
            (
                AgentTerminalStatus.ACCEPTED,
                AgentTerminalStatus.ACCEPTED,
                AgentTerminalStatus.FALLBACK_ACCEPTED,
            ),
        )
        for result in results:
            jsonl_lines = result.jsonl.splitlines()
            self.assertEqual(len(jsonl_lines), 1)
            self.assertEqual(
                json.loads(jsonl_lines[0])["schema_version"],
                "phase2.agent.v1",
            )
        self.assertEqual(len(self.processes), 3)
        self.assertTrue(
            all(process.returncode == 0 for process in self.processes)
        )
        output = stdout.getvalue()
        self.assertIn("successful-cycle trace:", output)
        self.assertIn("correction trace:", output)
        self.assertIn("fallback trace:", output)

    async def test_agent_cycle_state_is_fresh_between_runs(
        self,
    ) -> None:
        provider = build_success_provider()
        orchestrator = Phase2AgentOrchestrator()

        first = await orchestrator.run_cycle(
            provider,
            run_id="agent-isolation-one",
        )
        second = await orchestrator.run_cycle(
            provider,
            run_id="agent-isolation-two",
        )

        for result in (first, second):
            self.assertEqual(
                result.record.terminal_status,
                AgentTerminalStatus.ACCEPTED,
            )
            self.assertEqual(result.record.rounds_used, 4)
            self.assertEqual(result.record.tool_calls_used, 4)
            self.assertEqual(result.record.action_id, "action-000001")
            self.assertEqual(len(result.record.reasoning_log_ids), 1)
        self.assertEqual(len(self.processes), 2)
        self.assertTrue(
            all(process.returncode == 0 for process in self.processes)
        )


if __name__ == "__main__":
    unittest.main()
