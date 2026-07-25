"""In-process protocol tests for the validated Phase 2 MCP application."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import Mock, patch

from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult
from pydantic import BaseModel

from src.mcp_server import (
    DEFAULT_TRANSPORT,
    create_mcp_app,
    run_mcp_server,
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
    SafetyErrorCode,
    SetControlActionRequest,
    SetControlActionResponse,
)
from src.phase2_mock_services import (
    Phase2Fixture,
    Phase2Services,
    build_control_action_fixture,
    build_reasoning_fixture,
)


EXPECTED_TOOL_NAMES = {
    "read_sensor_data",
    "get_grid_carbon_intensity",
    "log_reasoning",
    "set_control_action",
    "parse_runtime_errors",
}


async def call_tool(
    app: object,
    tool_name: str,
    request: BaseModel,
) -> CallToolResult:
    """Call one MCP tool through the SDK's in-memory protocol transport."""

    async with create_connected_server_and_client_session(app) as client:
        return await client.call_tool(
            tool_name,
            {"request": request.model_dump(mode="json")},
        )


def parse_success(
    result: CallToolResult,
    response_type: type[BaseModel],
) -> BaseModel:
    """Require successful structured output and parse its canonical model."""

    if result.isError:
        raise AssertionError(f"unexpected MCP tool error: {result.content}")
    if result.structuredContent is None:
        raise AssertionError("successful MCP result has no structured content")
    response = response_type.model_validate(result.structuredContent)
    if response.model_dump()["schema_version"] != "phase2.v1":
        raise AssertionError("response schema_version is not phase2.v1")
    return response


def error_text(result: CallToolResult) -> str:
    """Flatten text blocks from an MCP error result."""

    return " ".join(
        block.text
        for block in result.content
        if hasattr(block, "text")
    )


class MCPToolDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    """Protect the exact public MCP surface and response schemas."""

    async def test_discovery_exposes_exactly_five_required_tools(self) -> None:
        app = create_mcp_app(Phase2Services.deterministic())

        async with create_connected_server_and_client_session(app) as client:
            discovered = (await client.list_tools()).tools

        self.assertEqual(
            {tool.name for tool in discovered},
            EXPECTED_TOOL_NAMES,
        )
        self.assertEqual(len(discovered), 5)
        for tool in discovered:
            with self.subTest(tool=tool.name):
                self.assertEqual(
                    tool.outputSchema["properties"]["schema_version"]["default"],
                    "phase2.v1",
                )
                self.assertEqual(
                    set(tool.inputSchema["properties"]),
                    {"request"},
                )
                self.assertFalse(tool.inputSchema["additionalProperties"])
                self.assertNotIn("path", str(tool.inputSchema).lower())


class MCPToolBehaviorTests(unittest.IsolatedAsyncioTestCase):
    """Exercise all five tools through an in-memory MCP session."""

    def setUp(self) -> None:
        logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
        self.services = Phase2Services.deterministic()
        self.app = create_mcp_app(self.services)

    async def log_fixture_reasoning(
        self,
        fixture: Phase2Fixture,
    ) -> LogReasoningResponse:
        """Log a fixture rationale through MCP and return its identifier."""

        snapshot = self.services.sensor_store.snapshot_for(fixture)
        request = build_reasoning_fixture(fixture, snapshot)
        result = await call_tool(
            self.app,
            "log_reasoning",
            request,
        )
        return LogReasoningResponse.model_validate(
            parse_success(result, LogReasoningResponse)
        )

    async def test_read_sensor_data_returns_current_snapshot(self) -> None:
        result = await call_tool(
            self.app,
            "read_sensor_data",
            ReadSensorDataRequest(
                request_id="mcp-read-1",
                history_steps=2,
            ),
        )
        response = ReadSensorDataResponse.model_validate(
            parse_success(result, ReadSensorDataResponse)
        )

        self.assertEqual(response.request_id, "mcp-read-1")
        self.assertEqual(response.snapshot, self.services.sensor_store.current)
        self.assertEqual(len(response.snapshot.zones), 5)
        self.assertEqual(len(response.history), 2)

    async def test_grid_carbon_tool_returns_mock_signal(self) -> None:
        snapshot = self.services.sensor_store.current
        result = await call_tool(
            self.app,
            "get_grid_carbon_intensity",
            GridCarbonIntensityRequest(
                request_id="mcp-grid-1",
                snapshot_id=snapshot.snapshot_id,
                forecast_steps=4,
            ),
        )
        response = GridCarbonIntensityResponse.model_validate(
            parse_success(result, GridCarbonIntensityResponse)
        )

        self.assertEqual(response.request_id, "mcp-grid-1")
        self.assertEqual(response.snapshot_id, snapshot.snapshot_id)
        self.assertEqual(response.source, "mock_fixture")
        self.assertEqual(len(response.forecast), 4)

    async def test_log_reasoning_creates_auditable_summary_record(self) -> None:
        snapshot = self.services.sensor_store.current
        request = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            snapshot,
        )

        result = await call_tool(self.app, "log_reasoning", request)
        response = LogReasoningResponse.model_validate(
            parse_success(result, LogReasoningResponse)
        )

        self.assertEqual(len(self.services.reasoning_ledger.records), 1)
        record = self.services.reasoning_ledger.records[0]
        self.assertEqual(response.request_id, request.request_id)
        self.assertEqual(response.cycle_id, request.cycle_id)
        self.assertEqual(response.snapshot_id, request.snapshot_id)
        self.assertEqual(record.request, request)
        self.assertEqual(
            record.response.reasoning_log_id,
            response.reasoning_log_id,
        )
        self.assertNotIn(
            "chain_of_thought",
            record.request.model_dump(mode="json"),
        )
        self.assertLessEqual(len(record.request.decision_summary), 1_000)

    async def test_log_reasoning_rejects_chain_of_thought_field(self) -> None:
        snapshot = self.services.sensor_store.current
        request = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            snapshot,
        )
        payload = request.model_dump(mode="json")
        payload["chain_of_thought"] = "private hidden reasoning"

        async with create_connected_server_and_client_session(
            self.app
        ) as client:
            result = await client.call_tool(
                "log_reasoning",
                {"request": payload},
            )

        self.assertTrue(result.isError)
        self.assertEqual(self.services.reasoning_ledger.records, ())

    async def test_all_tools_reject_unknown_top_level_arguments(self) -> None:
        snapshot = self.services.sensor_store.current
        requests = {
            "read_sensor_data": ReadSensorDataRequest(
                request_id="mcp-strict-read",
                history_steps=0,
            ),
            "get_grid_carbon_intensity": GridCarbonIntensityRequest(
                request_id="mcp-strict-grid",
                snapshot_id=snapshot.snapshot_id,
                forecast_steps=1,
            ),
            "log_reasoning": build_reasoning_fixture(
                Phase2Fixture.COMFORTABLE_OCCUPIED,
                snapshot,
            ),
            "set_control_action": build_control_action_fixture(
                Phase2Fixture.COMFORTABLE_OCCUPIED,
                snapshot,
                "reasoning-log-strict-envelope",
            ),
            "parse_runtime_errors": ParseRuntimeErrorsRequest(
                request_id="mcp-strict-errors",
                cycle_id=snapshot.cycle_id,
                limit=20,
            ),
        }

        async with create_connected_server_and_client_session(
            self.app
        ) as client:
            for tool_name, request in requests.items():
                with self.subTest(tool=tool_name):
                    result = await client.call_tool(
                        tool_name,
                        {
                            "request": request.model_dump(mode="json"),
                            "path": "C:/secret",
                        },
                    )
                    self.assertTrue(result.isError)

        self.assertEqual(self.services.reasoning_ledger.records, ())
        self.assertEqual(self.services.action_ledger.records, ())

    async def test_log_reasoning_rejects_bad_snapshot_correlation(self) -> None:
        snapshot = self.services.sensor_store.current
        valid = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            snapshot,
        )
        mismatched = LogReasoningRequest.model_validate(
            {
                **valid.model_dump(mode="json"),
                "cycle_id": "cycle-mismatch",
            }
        )

        result = await call_tool(self.app, "log_reasoning", mismatched)

        self.assertTrue(result.isError)
        self.assertIn("cycle_id does not match", error_text(result))
        self.assertEqual(self.services.reasoning_ledger.records, ())

    async def test_valid_five_zone_action_is_accepted(self) -> None:
        reasoning = await self.log_fixture_reasoning(
            Phase2Fixture.COMFORTABLE_OCCUPIED
        )
        snapshot = self.services.sensor_store.current
        request = build_control_action_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            snapshot,
            reasoning.reasoning_log_id,
        )

        result = await call_tool(self.app, "set_control_action", request)
        response = SetControlActionResponse.model_validate(
            parse_success(result, SetControlActionResponse)
        )

        self.assertEqual(response.status, ControlActionStatus.ACCEPTED)
        self.assertEqual(response.request_id, request.request_id)
        self.assertEqual(response.cycle_id, request.cycle_id)
        self.assertEqual(response.snapshot_id, request.snapshot_id)
        self.assertIsNotNone(response.action_id)
        self.assertEqual(len(self.services.action_ledger.records), 1)
        self.assertEqual(len(request.commands), 5)

    async def test_unsafe_setpoints_return_structured_rejection(self) -> None:
        reasoning = await self.log_fixture_reasoning(
            Phase2Fixture.COMFORTABLE_OCCUPIED
        )
        request = build_control_action_fixture(
            Phase2Fixture.INVALID_SETPOINT_ERROR,
            self.services.sensor_store.current,
            reasoning.reasoning_log_id,
        )

        result = await call_tool(self.app, "set_control_action", request)
        response = SetControlActionResponse.model_validate(
            parse_success(result, SetControlActionResponse)
        )

        self.assertEqual(response.status, ControlActionStatus.REJECTED)
        self.assertIn(
            SafetyErrorCode.OUT_OF_RANGE,
            {error.code for error in response.errors},
        )
        self.assertEqual(self.services.action_ledger.records, ())

    async def test_deadband_violation_is_rejected_without_clamping(self) -> None:
        reasoning = await self.log_fixture_reasoning(
            Phase2Fixture.COMFORTABLE_OCCUPIED
        )
        request = build_control_action_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            self.services.sensor_store.current,
            reasoning.reasoning_log_id,
        )
        payload = request.model_dump(mode="json")
        payload["commands"][0]["heating_c"] = 22.0
        payload["commands"][0]["cooling_c"] = 22.5
        unsafe = SetControlActionRequest.model_validate(payload)

        result = await call_tool(self.app, "set_control_action", unsafe)
        response = SetControlActionResponse.model_validate(
            parse_success(result, SetControlActionResponse)
        )

        self.assertEqual(response.status, ControlActionStatus.REJECTED)
        self.assertIn(
            SafetyErrorCode.DEADBAND_VIOLATION,
            {error.code for error in response.errors},
        )
        self.assertEqual(unsafe.commands[0].heating_c, 22.0)
        self.assertEqual(unsafe.commands[0].cooling_c, 22.5)

    async def test_stale_snapshot_action_is_rejected(self) -> None:
        current = self.services.sensor_store.current
        reasoning = await self.log_fixture_reasoning(
            Phase2Fixture.COMFORTABLE_OCCUPIED
        )
        request = build_control_action_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            current,
            reasoning.reasoning_log_id,
        )
        self.services.sensor_store.advance()

        result = await call_tool(self.app, "set_control_action", request)
        response = SetControlActionResponse.model_validate(
            parse_success(result, SetControlActionResponse)
        )

        self.assertEqual(response.status, ControlActionStatus.REJECTED)
        self.assertIn(
            SafetyErrorCode.STALE_SNAPSHOT,
            {error.code for error in response.errors},
        )
        self.assertEqual(self.services.action_ledger.records, ())

    async def test_missing_duplicate_and_unknown_zones_are_rejected(self) -> None:
        cases = (
            ("missing", SafetyErrorCode.MISSING_ZONE),
            ("duplicate", SafetyErrorCode.DUPLICATE_ZONE),
            ("unknown", SafetyErrorCode.UNKNOWN_ZONE),
        )
        for case_name, expected_code in cases:
            with self.subTest(case=case_name):
                services = Phase2Services.deterministic()
                app = create_mcp_app(services)
                snapshot = services.sensor_store.current
                reasoning_request = build_reasoning_fixture(
                    Phase2Fixture.COMFORTABLE_OCCUPIED,
                    snapshot,
                )
                reasoning_result = await call_tool(
                    app,
                    "log_reasoning",
                    reasoning_request,
                )
                reasoning = LogReasoningResponse.model_validate(
                    parse_success(
                        reasoning_result,
                        LogReasoningResponse,
                    )
                )
                request = build_control_action_fixture(
                    Phase2Fixture.COMFORTABLE_OCCUPIED,
                    snapshot,
                    reasoning.reasoning_log_id,
                )
                payload = request.model_dump(mode="json")
                if case_name == "missing":
                    payload["commands"] = payload["commands"][:-1]
                elif case_name == "duplicate":
                    payload["commands"][-1]["zone_id"] = payload["commands"][0][
                        "zone_id"
                    ]
                else:
                    payload["commands"][-1]["zone_id"] = "UNKNOWN-ZONE"
                malformed = SetControlActionRequest.model_validate(payload)

                result = await call_tool(
                    app,
                    "set_control_action",
                    malformed,
                )
                response = SetControlActionResponse.model_validate(
                    parse_success(result, SetControlActionResponse)
                )

                self.assertEqual(
                    response.status,
                    ControlActionStatus.REJECTED,
                )
                self.assertIn(
                    expected_code,
                    {error.code for error in response.errors},
                )
                self.assertEqual(services.action_ledger.records, ())

    async def test_duplicate_idempotency_key_does_not_append_twice(self) -> None:
        reasoning = await self.log_fixture_reasoning(
            Phase2Fixture.COMFORTABLE_OCCUPIED
        )
        request = build_control_action_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            self.services.sensor_store.current,
            reasoning.reasoning_log_id,
        )
        first_result = await call_tool(
            self.app,
            "set_control_action",
            request,
        )
        retry_payload = request.model_dump(mode="json")
        retry_payload["request_id"] = "mcp-action-retry"
        retry = SetControlActionRequest.model_validate(retry_payload)
        duplicate_result = await call_tool(
            self.app,
            "set_control_action",
            retry,
        )
        first = SetControlActionResponse.model_validate(
            parse_success(first_result, SetControlActionResponse)
        )
        duplicate = SetControlActionResponse.model_validate(
            parse_success(duplicate_result, SetControlActionResponse)
        )

        self.assertEqual(first.status, ControlActionStatus.ACCEPTED)
        self.assertEqual(duplicate.status, ControlActionStatus.DUPLICATE)
        self.assertEqual(duplicate.action_id, first.action_id)
        self.assertEqual(len(self.services.action_ledger.records), 1)

    async def test_parse_runtime_errors_returns_injected_error(self) -> None:
        snapshot = self.services.sensor_store.current
        injected = self.services.runtime_error_store.inject_fixture(
            snapshot.cycle_id,
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
            action_id="action-fixture",
        )

        result = await call_tool(
            self.app,
            "parse_runtime_errors",
            ParseRuntimeErrorsRequest(
                request_id="mcp-errors-1",
                cycle_id=snapshot.cycle_id,
                limit=20,
            ),
        )
        response = ParseRuntimeErrorsResponse.model_validate(
            parse_success(result, ParseRuntimeErrorsResponse)
        )

        self.assertEqual(response.cycle_id, snapshot.cycle_id)
        self.assertEqual(response.request_id, "mcp-errors-1")
        self.assertEqual(response.errors, (injected,))

    async def test_unexpected_failure_returns_sanitized_mcp_error(self) -> None:
        self.services.sensor_store.read = Mock(  # type: ignore[method-assign]
            side_effect=RuntimeError("private failure detail")
        )

        with self.assertLogs("src.mcp_server", level="ERROR"):
            result = await call_tool(
                self.app,
                "read_sensor_data",
                ReadSensorDataRequest(
                    request_id="mcp-internal-error",
                    history_steps=0,
                ),
            )

        self.assertTrue(result.isError)
        self.assertIn(
            "internal Phase 2 service failure",
            error_text(result),
        )
        self.assertNotIn("private failure detail", error_text(result))

    async def test_malformed_service_output_is_sanitized(self) -> None:
        private_value = "SHOULD_NOT_LEAK"
        self.services.sensor_store.read = Mock(  # type: ignore[method-assign]
            return_value={"private_token": private_value}
        )

        with self.assertLogs("src.mcp_server", level="ERROR") as captured:
            result = await call_tool(
                self.app,
                "read_sensor_data",
                ReadSensorDataRequest(
                    request_id="mcp-invalid-service-output",
                    history_steps=0,
                ),
            )

        self.assertTrue(result.isError)
        self.assertIn(
            "internal Phase 2 service failure",
            error_text(result),
        )
        self.assertNotIn(private_value, error_text(result))
        self.assertNotIn(private_value, " ".join(captured.output))


class MCPStateIsolationTests(unittest.IsolatedAsyncioTestCase):
    """Ensure factories do not share mutable ledgers."""

    async def test_factory_instances_start_with_isolated_state(self) -> None:
        reference = Phase2Services.deterministic()
        snapshot = reference.sensor_store.current
        request = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            snapshot,
        )
        first_app = create_mcp_app(Phase2Services.deterministic())
        second_app = create_mcp_app(Phase2Services.deterministic())

        first_result = await call_tool(first_app, "log_reasoning", request)
        second_result = await call_tool(second_app, "log_reasoning", request)
        first = LogReasoningResponse.model_validate(
            parse_success(first_result, LogReasoningResponse)
        )
        second = LogReasoningResponse.model_validate(
            parse_success(second_result, LogReasoningResponse)
        )

        self.assertEqual(first.reasoning_log_id, "reasoning-log-000001")
        self.assertEqual(second.reasoning_log_id, "reasoning-log-000001")

    async def test_injected_service_bundles_remain_isolated(self) -> None:
        first_services = Phase2Services.deterministic()
        second_services = Phase2Services.deterministic()
        first_app = create_mcp_app(first_services)
        second_app = create_mcp_app(second_services)
        request = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            first_services.sensor_store.current,
        )

        await call_tool(first_app, "log_reasoning", request)

        self.assertEqual(len(first_services.reasoning_ledger.records), 1)
        self.assertEqual(second_services.reasoning_ledger.records, ())
        second_result = await call_tool(second_app, "log_reasoning", request)
        self.assertFalse(second_result.isError)
        self.assertEqual(len(second_services.reasoning_ledger.records), 1)


class MCPTransportTests(unittest.TestCase):
    """Protect the required default stdio server transport."""

    def test_runner_uses_stdio_transport(self) -> None:
        fake_app = Mock()
        services = Phase2Services.deterministic()

        with patch("src.mcp_server.create_mcp_app", return_value=fake_app):
            run_mcp_server(services)

        self.assertEqual(DEFAULT_TRANSPORT, "stdio")
        fake_app.run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    unittest.main()
