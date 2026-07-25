"""Reusable Phase 2 MCP client bridge over a real stdio subprocess.

The bridge owns the server subprocess, MCP session, dynamic tool discovery, and
cleanup. It intentionally contains no LLM or control-loop orchestration.
"""

from __future__ import annotations

import copy
import logging
import math
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, TextIO

import anyio
from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.session import CONNECTION_CLOSED
from mcp.types import InitializeResult, Tool
from pydantic import BaseModel

from src.phase2_contracts import SCHEMA_VERSION


DEFAULT_CONNECTION_TIMEOUT_SECONDS = 10.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0
PHASE2_TOOL_NAMES = (
    "read_sensor_data",
    "get_grid_carbon_intensity",
    "log_reasoning",
    "set_control_action",
    "parse_runtime_errors",
)
READ_ONLY_TOOL_NAMES = frozenset(
    {
        "read_sensor_data",
        "get_grid_carbon_intensity",
        "parse_runtime_errors",
    }
)
RECONNECTABLE_READ_TOOL_NAMES = frozenset(
    {
        "read_sensor_data",
        "get_grid_carbon_intensity",
    }
)
_MAX_DISCOVERY_PAGES = 8
_MAX_DISCOVERED_TOOLS = 64
_REQUEST_TIMEOUT_ERROR_CODE = 408

logger = logging.getLogger(__name__)


class MCPClientBridgeError(RuntimeError):
    """Base error for the reusable stdio bridge."""


class MCPClientStateError(MCPClientBridgeError):
    """The caller used the client outside its active context."""


class MCPTransportError(MCPClientBridgeError):
    """The stdio process or MCP session could not complete a request."""


class MCPIndeterminateOutcomeError(MCPTransportError):
    """A mutation may have executed before its transport response was lost."""


class MCPProtocolError(MCPClientBridgeError):
    """The server violated its dynamically advertised MCP contract."""


class MCPToolInvocationError(MCPClientBridgeError):
    """The server returned an MCP-level tool execution error."""


def default_phase2_server_parameters(
    repository_root: Path | None = None,
) -> StdioServerParameters:
    """Build trusted parameters for the checked-in Phase 2 stdio server."""

    root = (
        repository_root
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    ).resolve(strict=True)
    server_module = root / "src" / "mcp_server.py"
    if not server_module.is_file():
        raise FileNotFoundError(
            f"Phase 2 MCP server was not found below {root}"
        )
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "src.mcp_server"],
        cwd=root,
        env=None,
        encoding="utf-8",
        encoding_error_handler="strict",
    )


def _positive_timeout(value: float, field_name: str) -> float:
    """Reject disabled, nonfinite, or nonsensical lifecycle timeouts."""

    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(value)


def _exception_group_matches(
    exc: BaseException,
    predicate: Callable[[BaseException], bool],
) -> bool:
    """Return whether every leaf in an exception group matches a predicate."""

    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            _exception_group_matches(child, predicate)
            for child in exc.exceptions
        )
    return bool(predicate(exc))


def _is_transport_failure(exc: BaseException) -> bool:
    """Classify only failures that mean the MCP transport is unavailable."""

    def is_leaf_transport_failure(leaf: BaseException) -> bool:
        if isinstance(leaf, McpError):
            return leaf.error.code in {
                CONNECTION_CLOSED,
                _REQUEST_TIMEOUT_ERROR_CODE,
            }
        return isinstance(
            leaf,
            (
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
                anyio.EndOfStream,
                ConnectionError,
                OSError,
                TimeoutError,
            ),
        )

    return _exception_group_matches(exc, is_leaf_transport_failure)


def _is_cancellation(exc: BaseException) -> bool:
    """Recognize cancellation without translating it into a protocol error."""

    cancellation_type = anyio.get_cancelled_exc_class()
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_cancellation(child) for child in exc.exceptions)
    return isinstance(exc, cancellation_type)


def _is_expected_teardown_failure(exc: BaseException) -> bool:
    """Recognize stream-close races emitted by MCP 1.28.1 during teardown."""

    return _exception_group_matches(
        exc,
        lambda leaf: isinstance(
            leaf,
            (
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
                anyio.EndOfStream,
                ProcessLookupError,
            ),
        ),
    )


def _tool_error_text(content: list[Any]) -> str:
    """Extract a concise client-facing message from MCP content blocks."""

    text = " ".join(
        block.text.strip()
        for block in content
        if hasattr(block, "text") and block.text.strip()
    )
    return text[:1_000] if text else "server returned an unspecified tool error"


class _ProtocolFaultMonitor:
    """Record non-JSON data received on the server's protocol stdout."""

    def __init__(self) -> None:
        self._fault_types: list[str] = []

    async def handle(self, message: Any) -> None:
        if isinstance(message, Exception):
            self._fault_types.append(type(message).__name__)
        await anyio.lowlevel.checkpoint()

    def raise_if_faulted(self) -> None:
        if self._fault_types:
            raise MCPProtocolError(
                "MCP server stdout contained non-protocol data"
            )


def _validate_phase2_tool_surface(tools: Mapping[str, Tool]) -> None:
    """Fail closed unless discovery matches the strict Phase 2 surface."""

    if len(tools) != len(PHASE2_TOOL_NAMES) or set(tools) != set(
        PHASE2_TOOL_NAMES
    ):
        raise MCPProtocolError(
            f"Unexpected Phase 2 MCP tools: {tuple(tools)!r}"
        )

    for tool_name, tool in tools.items():
        input_schema = tool.inputSchema
        properties = input_schema.get("properties")
        required = input_schema.get("required")
        if (
            not isinstance(properties, dict)
            or set(properties) != {"request"}
            or required != ["request"]
            or input_schema.get("additionalProperties") is not False
        ):
            raise MCPProtocolError(
                f"{tool_name} advertised an unsafe request envelope"
            )

        output_schema = tool.outputSchema
        output_properties = (
            output_schema.get("properties")
            if isinstance(output_schema, dict)
            else None
        )
        version_schema = (
            output_properties.get("schema_version")
            if isinstance(output_properties, dict)
            else None
        )
        if (
            not isinstance(version_schema, dict)
            or version_schema.get("default") != SCHEMA_VERSION
        ):
            raise MCPProtocolError(
                f"{tool_name} did not advertise {SCHEMA_VERSION} output"
            )


@asynccontextmanager
async def _open_stdio_session(
    parameters: StdioServerParameters,
    connection_timeout_seconds: float,
    errlog: TextIO,
) -> AsyncIterator[
    tuple[ClientSession, InitializeResult, _ProtocolFaultMonitor]
]:
    """Own the SDK contexts lexically so their cancel scopes unwind in order."""

    protocol_monitor = _ProtocolFaultMonitor()
    async with stdio_client(parameters, errlog=errlog) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(
                seconds=connection_timeout_seconds
            ),
            message_handler=protocol_monitor.handle,
        ) as session:
            initialization = await session.initialize()
            protocol_monitor.raise_if_faulted()
            yield session, initialization, protocol_monitor


class Phase2MCPClient:
    """One reusable, sequential MCP session backed by a stdio subprocess."""

    def __init__(
        self,
        server_parameters: StdioServerParameters | None = None,
        *,
        connection_timeout_seconds: float = (
            DEFAULT_CONNECTION_TIMEOUT_SECONDS
        ),
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        allow_read_reconnect: bool = True,
        errlog: TextIO | None = None,
    ) -> None:
        self._server_parameters = (
            server_parameters or default_phase2_server_parameters()
        )
        self._connection_timeout_seconds = _positive_timeout(
            connection_timeout_seconds,
            "connection_timeout_seconds",
        )
        self._request_timeout_seconds = _positive_timeout(
            request_timeout_seconds,
            "request_timeout_seconds",
        )
        if not isinstance(allow_read_reconnect, bool):
            raise TypeError("allow_read_reconnect must be a bool")
        self._allow_read_reconnect = allow_read_reconnect
        self._errlog = errlog if errlog is not None else sys.stderr
        self._connection_context: (
            AbstractAsyncContextManager[
                tuple[
                    ClientSession,
                    InitializeResult,
                    _ProtocolFaultMonitor,
                ]
            ]
            | None
        ) = None
        self._session: ClientSession | None = None
        self._initialization: InitializeResult | None = None
        self._protocol_monitor: _ProtocolFaultMonitor | None = None
        self._tools: dict[str, Tool] = {}
        self._owner_task_id: int | None = None
        self._has_mutation_state = False
        self._lifecycle_lock = anyio.Lock()

    async def __aenter__(self) -> Phase2MCPClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        try:
            await self.close()
        except BaseException as cleanup_exc:
            if _is_cancellation(cleanup_exc):
                raise
            if exc is None:
                raise
            logger.exception(
                "phase2_mcp_cleanup_failed",
                extra={"original_error_type": type(exc).__name__},
            )

    @property
    def is_connected(self) -> bool:
        """Return whether an initialized stdio session is currently active."""

        return self._session is not None

    @property
    def initialization(self) -> InitializeResult:
        """Return the negotiated MCP initialization result."""

        if self._initialization is None:
            raise MCPClientStateError("MCP client is not connected")
        return self._initialization

    @property
    def tools(self) -> Mapping[str, Tool]:
        """Expose the dynamically discovered tool metadata read-only."""

        self._require_session()
        return MappingProxyType(
            {
                name: tool.model_copy(deep=True)
                for name, tool in self._tools.items()
            }
        )

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return discovered tool names in server discovery order."""

        self._require_session()
        return tuple(self._tools)

    async def connect(self) -> None:
        """Start the stdio server, initialize MCP, and discover its tools."""

        async with self._lifecycle_lock:
            await self._connect_unlocked()

    async def _connect_unlocked(self) -> None:
        """Connect while the public lifecycle lock is held or unnecessary."""

        if self.is_connected:
            raise MCPClientStateError("MCP client is already connected")
        self._owner_task_id = anyio.get_current_task().id

        context = _open_stdio_session(
            self._server_parameters,
            self._connection_timeout_seconds,
            self._errlog,
        )
        entered = False
        try:
            (
                session,
                initialization,
                protocol_monitor,
            ) = await context.__aenter__()
            entered = True
            self._connection_context = context
            self._session = session
            self._initialization = initialization
            self._protocol_monitor = protocol_monitor
            self._has_mutation_state = False
            with anyio.fail_after(self._connection_timeout_seconds):
                self._tools = await self._discover_tools_once()
            _validate_phase2_tool_surface(self._tools)
        except BaseException as exc:
            if entered:
                try:
                    await self._close_active_connection()
                except BaseException as cleanup_exc:
                    if _is_cancellation(cleanup_exc):
                        raise
                    logger.error(
                        "phase2_mcp_failed_connect_cleanup",
                        extra={
                            "error_type": type(cleanup_exc).__name__,
                            "original_error_type": type(exc).__name__,
                        },
                    )
            else:
                self._owner_task_id = None
            if _is_cancellation(exc):
                raise
            if isinstance(exc, MCPClientBridgeError):
                raise
            if _is_transport_failure(exc):
                raise MCPTransportError(
                    "Could not initialize the Phase 2 MCP stdio session"
                ) from exc
            raise MCPProtocolError(
                "Could not validate the Phase 2 MCP stdio session"
            ) from exc

    async def close(self) -> None:
        """Close the session and subprocess, tolerating only known close races."""

        self._require_owner_task()
        async with self._lifecycle_lock:
            await self._close_active_connection()

    async def discover_tools(self) -> tuple[Tool, ...]:
        """Refresh and return all server tools through paginated discovery."""

        self._require_owner_task()
        self._tools = await self._discover_tools_once()
        _validate_phase2_tool_surface(self._tools)
        return tuple(
            tool.model_copy(deep=True)
            for tool in self._tools.values()
        )

    async def call_request(
        self,
        tool_name: str,
        request: BaseModel | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Call a tool using the request parameter advertised by its schema."""

        self._require_owner_task()
        tool = self._require_tool(tool_name)
        properties = tool.inputSchema.get("properties")
        required = tool.inputSchema.get("required")
        if (
            not isinstance(properties, dict)
            or not isinstance(required, list)
            or len(required) != 1
            or required[0] not in properties
        ):
            raise MCPProtocolError(
                f"{tool_name} did not advertise one required request envelope"
            )
        parameter_name = required[0]
        if isinstance(request, BaseModel):
            request_payload = request.model_dump(mode="json")
        else:
            request_payload = copy.deepcopy(dict(request))
        response = await self.call_tool(
            tool_name,
            {parameter_name: request_payload},
        )
        for correlation_field in (
            "request_id",
            "cycle_id",
            "snapshot_id",
        ):
            if (
                correlation_field in request_payload
                and response.get(correlation_field)
                != request_payload[correlation_field]
            ):
                await self._retire_connection_after_failure()
                if tool_name not in READ_ONLY_TOOL_NAMES:
                    raise MCPIndeterminateOutcomeError(
                        f"{tool_name} returned a mismatched "
                        f"{correlation_field} after it may have executed"
                    )
                raise MCPProtocolError(
                    f"{tool_name} returned a mismatched "
                    f"{correlation_field}"
                )
        return response

    async def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Invoke one discovered tool, retrying only safe reads once."""

        self._require_owner_task()
        self._require_tool(tool_name)
        stable_arguments = copy.deepcopy(dict(arguments))
        try:
            response = await self._call_tool_once(
                tool_name,
                stable_arguments,
            )
        except MCPProtocolError as exc:
            await self._retire_connection_after_failure()
            if tool_name not in READ_ONLY_TOOL_NAMES:
                raise MCPIndeterminateOutcomeError(
                    f"{tool_name} returned an invalid response after it "
                    "may have executed"
                ) from exc
            raise
        except MCPTransportError as exc:
            if (
                not self._allow_read_reconnect
                or tool_name not in RECONNECTABLE_READ_TOOL_NAMES
                or self._has_mutation_state
            ):
                state_continuity_was_required = self._has_mutation_state
                await self._retire_connection_after_failure()
                if tool_name not in READ_ONLY_TOOL_NAMES:
                    raise MCPIndeterminateOutcomeError(
                        f"{tool_name} may have executed; it was not retried"
                    ) from exc
                if state_continuity_was_required:
                    raise MCPTransportError(
                        f"{tool_name} was not replayed because reconnecting "
                        "would discard successful mutation state"
                    ) from exc
                raise
            logger.warning(
                "phase2_mcp_read_reconnect",
                extra={"tool_name": tool_name, "maximum_retries": 1},
            )
            await self._restart_connection()
            self._require_tool(tool_name)
            try:
                return await self._call_tool_once(
                    tool_name,
                    stable_arguments,
                )
            except (MCPProtocolError, MCPTransportError):
                await self._retire_connection_after_failure()
                raise
        if tool_name not in READ_ONLY_TOOL_NAMES:
            self._has_mutation_state = True
        return response

    async def _discover_tools_once(self) -> dict[str, Tool]:
        session = self._require_session()
        discovered: dict[str, Tool] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page_count = 0
        try:
            while True:
                page_count += 1
                if page_count > _MAX_DISCOVERY_PAGES:
                    raise MCPProtocolError(
                        "Server exceeded the tool-discovery page limit"
                    )
                page = await session.list_tools(cursor=cursor)
                self._raise_if_protocol_faulted()
                for tool in page.tools:
                    if tool.name in discovered:
                        raise MCPProtocolError(
                            f"Server advertised duplicate tool {tool.name!r}"
                        )
                    discovered[tool.name] = tool
                    if len(discovered) > _MAX_DISCOVERED_TOOLS:
                        raise MCPProtocolError(
                            "Server exceeded the discovered-tool limit"
                        )
                cursor = page.nextCursor
                if cursor is None:
                    break
                if cursor in seen_cursors:
                    raise MCPProtocolError(
                        "Server repeated a tool-discovery cursor"
                    )
                seen_cursors.add(cursor)
        except BaseException as exc:
            if _is_cancellation(exc):
                raise
            if isinstance(exc, MCPClientBridgeError):
                raise
            if _is_transport_failure(exc):
                raise MCPTransportError(
                    "MCP tool discovery failed because the transport closed"
                ) from exc
            raise MCPProtocolError("MCP tool discovery failed") from exc
        return discovered

    async def _call_tool_once(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        session = self._require_session()
        self._raise_if_protocol_faulted()
        try:
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=timedelta(
                    seconds=self._request_timeout_seconds
                ),
            )
        except BaseException as exc:
            if _is_cancellation(exc):
                raise
            if _is_transport_failure(exc):
                raise MCPTransportError(
                    f"{tool_name} failed because the MCP transport closed"
                ) from exc
            raise MCPProtocolError(
                f"{tool_name} returned an invalid MCP response"
            ) from exc

        self._raise_if_protocol_faulted()

        if result.isError:
            raise MCPToolInvocationError(
                f"{tool_name} failed: {_tool_error_text(result.content)}"
            )
        if result.structuredContent is None:
            raise MCPProtocolError(
                f"{tool_name} returned no structured content"
            )
        structured = copy.deepcopy(dict(result.structuredContent))
        if structured.get("schema_version") != SCHEMA_VERSION:
            raise MCPProtocolError(
                f"{tool_name} returned an unsupported schema_version"
            )
        return structured

    async def _restart_connection(self) -> None:
        await self._close_active_connection()
        await self._connect_unlocked()

    async def _retire_connection_after_failure(self) -> None:
        try:
            await self._close_active_connection()
        except BaseException as exc:
            if _is_cancellation(exc):
                raise
            logger.error(
                "phase2_mcp_failed_transport_cleanup",
                extra={"error_type": type(exc).__name__},
            )

    async def _close_active_connection(self) -> None:
        context = self._connection_context
        self._connection_context = None
        self._session = None
        self._initialization = None
        self._protocol_monitor = None
        self._tools = {}
        self._owner_task_id = None
        self._has_mutation_state = False
        if context is None:
            return
        try:
            await context.__aexit__(None, None, None)
        except BaseException as exc:
            if _is_cancellation(exc):
                raise
            if not _is_expected_teardown_failure(exc):
                raise
            logger.warning(
                "phase2_mcp_teardown_stream_closed",
                extra={"error_type": type(exc).__name__},
            )

    def _raise_if_protocol_faulted(self) -> None:
        if self._protocol_monitor is not None:
            self._protocol_monitor.raise_if_faulted()

    def _require_owner_task(self) -> None:
        if (
            self._owner_task_id is not None
            and self._owner_task_id != anyio.get_current_task().id
        ):
            raise MCPClientStateError(
                "MCP client lifecycle and calls must use one async task"
            )

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPClientStateError(
                "MCP client must be used inside an active async context"
            )
        return self._session

    def _require_tool(self, tool_name: str) -> Tool:
        self._require_session()
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise MCPProtocolError(
                f"MCP server did not advertise tool {tool_name!r}"
            ) from exc


__all__ = [
    "DEFAULT_CONNECTION_TIMEOUT_SECONDS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "MCPClientBridgeError",
    "MCPIndeterminateOutcomeError",
    "MCPClientStateError",
    "MCPProtocolError",
    "MCPToolInvocationError",
    "MCPTransportError",
    "PHASE2_TOOL_NAMES",
    "Phase2MCPClient",
    "READ_ONLY_TOOL_NAMES",
    "RECONNECTABLE_READ_TOOL_NAMES",
    "default_phase2_server_parameters",
]
