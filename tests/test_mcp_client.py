"""Real-stdio lifecycle tests for the reusable Phase 2 MCP client."""

from __future__ import annotations

import asyncio
import contextlib
import io
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import mcp.client.stdio as mcp_stdio

from scripts.run_phase2_mcp_smoke import run_smoke_workflow
from src.mcp_client import (
    MCPIndeterminateOutcomeError,
    MCPTransportError,
    PHASE2_TOOL_NAMES,
    Phase2MCPClient,
)
from src.phase2_contracts import (
    ControlActionStatus,
    GridCarbonIntensityRequest,
    GridCarbonIntensityResponse,
    LogReasoningRequest,
    LogReasoningResponse,
    ObjectiveTag,
    ParseRuntimeErrorsRequest,
    ParseRuntimeErrorsResponse,
    ReadSensorDataRequest,
    ReadSensorDataResponse,
    SafetyErrorCode,
    SetControlActionRequest,
    SetControlActionResponse,
    SetZoneCommand,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
class ExpectedWorkflowError(RuntimeError):
    """Sentinel raised to exercise exceptional client cleanup."""


def build_reasoning_request(
    snapshot: Any,
    request_id: str,
) -> LogReasoningRequest:
    """Build one concise, snapshot-correlated reasoning record."""

    return LogReasoningRequest(
        request_id=request_id,
        cycle_id=snapshot.cycle_id,
        snapshot_id=snapshot.snapshot_id,
        decision_summary=(
            "Hold safe setpoints while the occupied zones remain comfortable."
        ),
        objective_tags=(
            ObjectiveTag.THERMAL_COMFORT,
            ObjectiveTag.ENERGY_REDUCTION,
            ObjectiveTag.SAFETY,
        ),
        tradeoff_summary=(
            "Preserve comfort without adding avoidable electricity demand."
        ),
        confidence=0.95,
    )


def build_action_request(
    snapshot: Any,
    reasoning_log_id: str,
    *,
    request_id: str,
    idempotency_key: str,
    unsafe: bool = False,
) -> SetControlActionRequest:
    """Build a complete five-zone action from protocol-returned zones."""

    commands = tuple(
        SetZoneCommand(
            zone_id=zone.zone_id,
            heating_c=(
                15.0 if unsafe else zone.heating_setpoint_c
            ),
            cooling_c=(
                31.0 if unsafe else zone.cooling_setpoint_c
            ),
        )
        for zone in snapshot.zones
    )
    return SetControlActionRequest(
        request_id=request_id,
        cycle_id=snapshot.cycle_id,
        snapshot_id=snapshot.snapshot_id,
        reasoning_log_id=reasoning_log_id,
        idempotency_key=idempotency_key,
        commands=commands,
        hold_steps=1,
    )


class MCPClientRealStdioTests(unittest.IsolatedAsyncioTestCase):
    """Exercise actual subprocesses while tracking every exact child process."""

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
            f"orphan MCP server processes required cleanup: {orphan_pids}",
        )

    async def _capture_process(self, *args: Any, **kwargs: Any) -> Any:
        process = await self.original_process_factory(*args, **kwargs)
        self.processes.append(process)
        return process

    async def _read_snapshot(
        self,
        client: Phase2MCPClient,
        request_id: str,
    ) -> ReadSensorDataResponse:
        return ReadSensorDataResponse.model_validate(
            await client.call_request(
                "read_sensor_data",
                ReadSensorDataRequest(
                    request_id=request_id,
                    history_steps=1,
                ),
            )
        )

    async def test_stdio_initializes_and_discovers_exactly_five_tools(
        self,
    ) -> None:
        client = Phase2MCPClient()

        async with client:
            self.assertEqual(
                client.initialization.serverInfo.name,
                "eco-loop-phase2",
            )
            self.assertEqual(len(client.tools), 5)
            self.assertEqual(client.tool_names, PHASE2_TOOL_NAMES)
            for tool in client.tools.values():
                with self.subTest(tool=tool.name):
                    self.assertIn("request", tool.inputSchema["properties"])
                    self.assertIn(
                        "schema_version",
                        tool.outputSchema["properties"],
                    )

        self.assertFalse(client.is_connected)
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.processes[0].returncode, 0)

    async def test_all_five_tools_and_safety_outcomes_use_real_protocol(
        self,
    ) -> None:
        async with Phase2MCPClient() as client:
            sensor = await self._read_snapshot(
                client,
                "client-flow-read",
            )
            snapshot = sensor.snapshot
            self.assertEqual(len(snapshot.zones), 5)

            carbon = GridCarbonIntensityResponse.model_validate(
                await client.call_request(
                    "get_grid_carbon_intensity",
                    GridCarbonIntensityRequest(
                        request_id="client-flow-carbon",
                        snapshot_id=snapshot.snapshot_id,
                        forecast_steps=4,
                    ),
                )
            )
            self.assertEqual(carbon.snapshot_id, snapshot.snapshot_id)

            reasoning = LogReasoningResponse.model_validate(
                await client.call_request(
                    "log_reasoning",
                    build_reasoning_request(
                        snapshot,
                        "client-flow-reasoning",
                    ),
                )
            )
            self.assertTrue(reasoning.logged)

            safe_action = SetControlActionResponse.model_validate(
                await client.call_request(
                    "set_control_action",
                    build_action_request(
                        snapshot,
                        reasoning.reasoning_log_id,
                        request_id="client-flow-safe-action",
                        idempotency_key="client-flow-safe-action-key",
                    ),
                )
            )
            self.assertEqual(
                safe_action.status,
                ControlActionStatus.ACCEPTED,
            )

            unsafe_action = SetControlActionResponse.model_validate(
                await client.call_request(
                    "set_control_action",
                    build_action_request(
                        snapshot,
                        reasoning.reasoning_log_id,
                        request_id="client-flow-unsafe-action",
                        idempotency_key="client-flow-unsafe-action-key",
                        unsafe=True,
                    ),
                )
            )
            self.assertEqual(
                unsafe_action.status,
                ControlActionStatus.REJECTED,
            )
            self.assertIn(
                SafetyErrorCode.OUT_OF_RANGE,
                {error.code for error in unsafe_action.errors},
            )

            runtime_errors = ParseRuntimeErrorsResponse.model_validate(
                await client.call_request(
                    "parse_runtime_errors",
                    ParseRuntimeErrorsRequest(
                        request_id="client-flow-runtime-errors",
                        cycle_id=snapshot.cycle_id,
                        limit=20,
                    ),
                )
            )
            self.assertEqual(runtime_errors.cycle_id, snapshot.cycle_id)

        self.assertEqual(self.processes[0].returncode, 0)

    async def test_client_cleans_up_after_success_without_orphan(
        self,
    ) -> None:
        client = Phase2MCPClient()

        async with client:
            await self._read_snapshot(client, "client-clean-success")

        self.assertFalse(client.is_connected)
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.processes[0].returncode, 0)

    async def test_client_cleans_up_after_exception_without_orphan(
        self,
    ) -> None:
        client = Phase2MCPClient()

        with self.assertRaisesRegex(
            ExpectedWorkflowError,
            "intentional workflow failure",
        ):
            async with client:
                await self._read_snapshot(
                    client,
                    "client-clean-exception",
                )
                raise ExpectedWorkflowError(
                    "intentional workflow failure"
                )

        self.assertFalse(client.is_connected)
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.processes[0].returncode, 0)

    async def test_smoke_workflow_succeeds_and_shuts_down_cleanly(
        self,
    ) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            summary = await run_smoke_workflow()

        output = stdout.getvalue()
        self.assertIn("Discovered MCP tools:", output)
        self.assertIn("Smoke result:", output)
        self.assertEqual(summary["action_status"], "accepted")
        self.assertEqual(summary["zone_count"], 5)
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.processes[0].returncode, 0)

    async def test_read_only_call_reconnects_once_after_transport_loss(
        self,
    ) -> None:
        client = Phase2MCPClient()

        async with client:
            first_process = self.processes[0]
            first_process.terminate()
            await asyncio.wait_for(first_process.wait(), timeout=3.0)

            response = await self._read_snapshot(
                client,
                "client-reconnected-read",
            )

            self.assertEqual(len(response.snapshot.zones), 5)
            self.assertEqual(len(self.processes), 2)
            self.assertIsNotNone(first_process.returncode)
            self.assertIsNone(self.processes[1].returncode)

        self.assertTrue(
            all(process.returncode is not None for process in self.processes)
        )
        self.assertEqual(self.processes[1].returncode, 0)

    async def test_side_effecting_call_is_not_retried_after_transport_loss(
        self,
    ) -> None:
        client = Phase2MCPClient()

        async with client:
            snapshot = (
                await self._read_snapshot(
                    client,
                    "client-no-write-retry-read",
                )
            ).snapshot
            process = self.processes[0]
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=3.0)

            with self.assertRaises(MCPIndeterminateOutcomeError):
                await client.call_request(
                    "log_reasoning",
                    build_reasoning_request(
                        snapshot,
                        "client-no-write-retry-reasoning",
                    ),
                )

            self.assertEqual(len(self.processes), 1)
            self.assertFalse(client.is_connected)

    async def test_read_does_not_reconnect_after_mutation_state_exists(
        self,
    ) -> None:
        client = Phase2MCPClient()

        async with client:
            snapshot = (
                await self._read_snapshot(
                    client,
                    "client-stateful-read",
                )
            ).snapshot
            reasoning = LogReasoningResponse.model_validate(
                await client.call_request(
                    "log_reasoning",
                    build_reasoning_request(
                        snapshot,
                        "client-stateful-reasoning",
                    ),
                )
            )
            self.assertTrue(reasoning.logged)

            process = self.processes[0]
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=3.0)

            with self.assertRaisesRegex(
                MCPTransportError,
                "discard successful mutation state",
            ):
                await self._read_snapshot(
                    client,
                    "client-stateful-reconnect-read",
                )

            self.assertEqual(len(self.processes), 1)
            self.assertFalse(client.is_connected)

    async def test_request_cancellation_is_not_wrapped_and_still_cleans_up(
        self,
    ) -> None:
        client = Phase2MCPClient()

        async with client:
            session = client._session
            self.assertIsNotNone(session)

            async def wait_until_cancelled(*args: Any, **kwargs: Any) -> Any:
                await anyio.sleep(10.0)

            with (
                patch.object(
                    session,
                    "call_tool",
                    new=wait_until_cancelled,
                ),
                anyio.move_on_after(0.05) as cancel_scope,
            ):
                await client.call_request(
                    "read_sensor_data",
                    ReadSensorDataRequest(
                        request_id="client-cancelled-read",
                        history_steps=0,
                    ),
                )

            self.assertTrue(cancel_scope.cancelled_caught)
            self.assertTrue(client.is_connected)

        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.processes[0].returncode, 0)


if __name__ == "__main__":
    unittest.main()
