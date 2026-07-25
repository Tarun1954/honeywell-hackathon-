"""Validated Phase 2 MCP tools backed by injected deterministic services.

This module owns only the MCP application boundary.  It does not start a client,
invoke an LLM, or connect to EnergyPlus.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal, TypeVar

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel

from src.phase2_contracts import (
    GridCarbonIntensityRequest,
    GridCarbonIntensityResponse,
    LogReasoningRequest,
    LogReasoningResponse,
    ParseRuntimeErrorsRequest,
    ParseRuntimeErrorsResponse,
    ReadSensorDataRequest,
    ReadSensorDataResponse,
    SetControlActionRequest,
    SetControlActionResponse,
)
from src.phase2_mock_services import MockServiceError, Phase2Services


MCP_SERVER_NAME = "eco-loop-phase2"
DEFAULT_TRANSPORT: Literal["stdio"] = "stdio"
PHASE2_TOOL_NAMES = (
    "read_sensor_data",
    "get_grid_carbon_intensity",
    "log_reasoning",
    "set_control_action",
    "parse_runtime_errors",
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
logger = logging.getLogger(__name__)


def _enforce_strict_tool_envelopes(app: FastMCP) -> None:
    """Reject unknown top-level arguments instead of silently ignoring them.

    MCP 1.28.1 builds each FastMCP argument envelope from a Pydantic model whose
    default is ``extra="ignore"``.  FastMCP also bypasses low-level JSON Schema
    validation, so both the runtime model and its published schema must be made
    strict after registration.  Keeping this pinned-SDK workaround here avoids
    changing global Pydantic or MCP behavior.
    """

    for tool_name in PHASE2_TOOL_NAMES:
        tool = app._tool_manager.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(
                f"Phase 2 MCP tool registration is missing {tool_name!r}"
            )
        argument_model = tool.fn_metadata.arg_model
        argument_model.model_config["extra"] = "forbid"
        argument_model.model_rebuild(force=True)
        tool.parameters = argument_model.model_json_schema(by_alias=True)


def _execute_tool(
    tool_name: str,
    response_type: type[ResponseT],
    operation: Callable[[], object],
) -> ResponseT:
    """Validate service output and convert failures into clear MCP errors."""

    try:
        return response_type.model_validate(operation())
    except ToolError:
        raise
    except (KeyError, MockServiceError) as exc:
        raise ToolError(
            f"{tool_name} rejected the request: {exc}"
        ) from exc
    except Exception as exc:
        logger.error(
            "phase2_mcp_tool_failed",
            extra={
                "tool_name": tool_name,
                "error_type": type(exc).__name__,
            },
        )
        raise ToolError(
            f"{tool_name} encountered an internal Phase 2 service failure"
        ) from exc


def create_mcp_app(
    services: Phase2Services,
) -> FastMCP:
    """Create one isolated five-tool MCP application.

    Requiring a dependency bundle makes state ownership explicit, so separate
    application factories cannot accidentally share ledgers or fixture cursors.
    """

    service_bundle = services
    app = FastMCP(
        name=MCP_SERVER_NAME,
        instructions=(
            "Operate the deterministic Phase 2 building mock through validated "
            "phase2.v1 contracts. Log only concise decision summaries."
        ),
        log_level="ERROR",
    )

    @app.tool(
        name="read_sensor_data",
        description=(
            "Read the current deterministic building snapshot and bounded "
            "15-minute history."
        ),
        structured_output=True,
    )
    def read_sensor_data(
        request: ReadSensorDataRequest,
    ) -> ReadSensorDataResponse:
        """Return the current mock snapshot without advancing simulation time."""

        return _execute_tool(
            "read_sensor_data",
            ReadSensorDataResponse,
            lambda: service_bundle.sensor_store.read(request),
        )

    @app.tool(
        name="get_grid_carbon_intensity",
        description=(
            "Read deterministic current and forecast grid-carbon intensity for "
            "a known snapshot."
        ),
        structured_output=True,
    )
    def get_grid_carbon_intensity(
        request: GridCarbonIntensityRequest,
    ) -> GridCarbonIntensityResponse:
        """Return the mock carbon signal correlated to snapshot_id."""

        return _execute_tool(
            "get_grid_carbon_intensity",
            GridCarbonIntensityResponse,
            lambda: service_bundle.grid_carbon_store.read(request),
        )

    @app.tool(
        name="log_reasoning",
        description=(
            "Append a concise auditable decision summary, objective tags, "
            "tradeoff summary, and confidence. Never submit hidden "
            "chain-of-thought."
        ),
        structured_output=True,
    )
    def log_reasoning(
        request: LogReasoningRequest,
    ) -> LogReasoningResponse:
        """Store only the bounded summary fields in the canonical contract."""

        def append_correlated_reasoning() -> LogReasoningResponse:
            snapshot = service_bundle.sensor_store.get(request.snapshot_id)
            if snapshot.cycle_id != request.cycle_id:
                raise ToolError(
                    "log_reasoning rejected the request: cycle_id does not "
                    "match snapshot_id"
                )
            return service_bundle.log_reasoning(request)

        return _execute_tool(
            "log_reasoning",
            LogReasoningResponse,
            append_correlated_reasoning,
        )

    @app.tool(
        name="set_control_action",
        description=(
            "Validate and submit one idempotent five-zone thermostat action. "
            "Unsafe, incomplete, duplicate-zone, or stale actions are rejected "
            "without clamping."
        ),
        structured_output=True,
    )
    def set_control_action(
        request: SetControlActionRequest,
    ) -> SetControlActionResponse:
        """Return a structured accepted, rejected, or duplicate outcome."""

        return _execute_tool(
            "set_control_action",
            SetControlActionResponse,
            lambda: service_bundle.submit_action(request),
        )

    @app.tool(
        name="parse_runtime_errors",
        description=(
            "Retrieve a bounded, cycle-scoped page of deterministic runtime "
            "errors for corrective action."
        ),
        structured_output=True,
    )
    def parse_runtime_errors(
        request: ParseRuntimeErrorsRequest,
    ) -> ParseRuntimeErrorsResponse:
        """Return injected runtime errors through the canonical response."""

        return _execute_tool(
            "parse_runtime_errors",
            ParseRuntimeErrorsResponse,
            lambda: service_bundle.runtime_error_store.retrieve(request),
        )

    _enforce_strict_tool_envelopes(app)
    return app


def run_mcp_server(
    services: Phase2Services | None = None,
) -> None:
    """Run the validated MCP application over the default stdio transport."""

    service_bundle = services or Phase2Services.deterministic()
    create_mcp_app(service_bundle).run(transport=DEFAULT_TRANSPORT)


def main() -> None:
    """CLI entry point using stdio and an isolated deterministic bundle."""

    run_mcp_server()


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_TRANSPORT",
    "MCP_SERVER_NAME",
    "PHASE2_TOOL_NAMES",
    "create_mcp_app",
    "run_mcp_server",
]
