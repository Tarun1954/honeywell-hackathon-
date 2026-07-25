"""Run the deterministic Phase 2 workflow through a real MCP stdio session.

Usage:
    python -m scripts.run_phase2_mcp_smoke
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import anyio

from src.mcp_client import PHASE2_TOOL_NAMES, Phase2MCPClient
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
    SetControlActionRequest,
    SetControlActionResponse,
    SetZoneCommand,
)


async def run_smoke_workflow() -> dict[str, Any]:
    """Invoke all five tools over one real stdio subprocess."""

    async with Phase2MCPClient() as client:
        tool_names = client.tool_names
        if tool_names != PHASE2_TOOL_NAMES:
            raise RuntimeError(
                f"Unexpected MCP tool discovery result: {tool_names!r}"
            )
        print(
            "Discovered MCP tools: " + ", ".join(tool_names),
            flush=True,
        )

        sensor_response = ReadSensorDataResponse.model_validate(
            await client.call_request(
                "read_sensor_data",
                ReadSensorDataRequest(
                    request_id="smoke-read-001",
                    history_steps=2,
                ),
            )
        )
        snapshot = sensor_response.snapshot

        carbon_response = GridCarbonIntensityResponse.model_validate(
            await client.call_request(
                "get_grid_carbon_intensity",
                GridCarbonIntensityRequest(
                    request_id="smoke-carbon-001",
                    snapshot_id=snapshot.snapshot_id,
                    forecast_steps=4,
                ),
            )
        )

        reasoning_response = LogReasoningResponse.model_validate(
            await client.call_request(
                "log_reasoning",
                LogReasoningRequest(
                    request_id="smoke-reasoning-001",
                    cycle_id=snapshot.cycle_id,
                    snapshot_id=snapshot.snapshot_id,
                    decision_summary=(
                        "Hold the current safe setpoints because occupied "
                        "zones are comfortable."
                    ),
                    objective_tags=(
                        ObjectiveTag.THERMAL_COMFORT,
                        ObjectiveTag.ENERGY_REDUCTION,
                        ObjectiveTag.SAFETY,
                    ),
                    tradeoff_summary=(
                        "Preserve comfort while avoiding unnecessary demand."
                    ),
                    confidence=0.95,
                ),
            )
        )

        commands = tuple(
            SetZoneCommand(
                zone_id=zone.zone_id,
                heating_c=zone.heating_setpoint_c,
                cooling_c=zone.cooling_setpoint_c,
            )
            for zone in snapshot.zones
        )
        if len(commands) != 5:
            raise RuntimeError(
                f"Expected five zones in the snapshot, found {len(commands)}"
            )
        action_response = SetControlActionResponse.model_validate(
            await client.call_request(
                "set_control_action",
                SetControlActionRequest(
                    request_id="smoke-action-001",
                    cycle_id=snapshot.cycle_id,
                    snapshot_id=snapshot.snapshot_id,
                    reasoning_log_id=(
                        reasoning_response.reasoning_log_id
                    ),
                    idempotency_key=(
                        f"smoke-action-key-{snapshot.snapshot_id}"
                    ),
                    commands=commands,
                    hold_steps=1,
                ),
            )
        )
        if action_response.status is not ControlActionStatus.ACCEPTED:
            error_codes = [
                error.code.value for error in action_response.errors
            ]
            raise RuntimeError(
                "Smoke control action was not accepted: "
                f"status={action_response.status.value}, "
                f"errors={error_codes}"
            )
        if action_response.action_id is None:
            raise RuntimeError(
                "Smoke control action was accepted without an action_id"
            )

        runtime_response = ParseRuntimeErrorsResponse.model_validate(
            await client.call_request(
                "parse_runtime_errors",
                ParseRuntimeErrorsRequest(
                    request_id="smoke-runtime-errors-001",
                    cycle_id=snapshot.cycle_id,
                    limit=20,
                ),
            )
        )

        summary = {
            "schema_version": sensor_response.schema_version,
            "snapshot_id": snapshot.snapshot_id,
            "zone_count": len(snapshot.zones),
            "carbon_g_co2_per_kwh": (
                carbon_response.current.g_co2_per_kwh
            ),
            "reasoning_log_id": reasoning_response.reasoning_log_id,
            "action_status": action_response.status.value,
            "action_id": action_response.action_id,
            "runtime_error_count": len(runtime_response.errors),
        }
        print(
            "Smoke result: "
            + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        return summary


def main() -> None:
    """Configure stderr logging and run the async smoke workflow."""

    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )
    anyio.run(run_smoke_workflow)


if __name__ == "__main__":
    main()
