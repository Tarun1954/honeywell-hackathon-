"""Fixed-scenario stdio MCP server used by Phase 2 agent integration tests.

Usage:
    python -m tests.phase2_agent_fixture_server stale_once
    python -m tests.phase2_agent_fixture_server --scenario runtime_error

The helper exposes only the production server's five tools.  Scenario
selection happens at process startup and is never exposed as an MCP tool.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from enum import StrEnum
from threading import RLock

from src.mcp_server import DEFAULT_TRANSPORT, create_mcp_app
from src.phase2_contracts import (
    ReadSensorDataRequest,
    ReadSensorDataResponse,
    RuntimeErrorRecord,
    RuntimeErrorSeverity,
    RuntimeErrorSource,
)
from src.phase2_mock_services import (
    MockGridCarbonStore,
    MockSensorStore,
    Phase2Fixture,
    Phase2Services,
)


class FixtureScenario(StrEnum):
    """The only deterministic fixture-server modes accepted by the CLI."""

    STALE_ONCE = "stale_once"
    RUNTIME_ERROR = "runtime_error"
    RUNTIME_BLOCKING = "runtime_blocking"


class _StaleOnceSensorStore(MockSensorStore):
    """Advance after the first returned observation and remain stable after it."""

    def __init__(self, *, lock: RLock) -> None:
        super().__init__(
            initial_fixture=Phase2Fixture.COMFORTABLE_OCCUPIED,
            lock=lock,
        )
        self._advance_after_next_read = True

    def read(
        self,
        request: ReadSensorDataRequest,
    ) -> ReadSensorDataResponse:
        """Return the initial snapshot, then make that snapshot stale once."""

        with self._lock:
            response = super().read(request)
            if self._advance_after_next_read:
                self._advance_after_next_read = False
                super().advance()
            return response


def _replace_sensor_store(
    services: Phase2Services,
    sensor_store: MockSensorStore,
) -> Phase2Services:
    """Build a coherent bundle around a replacement shared-lock sensor store."""

    return Phase2Services(
        sensor_store=sensor_store,
        grid_carbon_store=MockGridCarbonStore(sensor_store),
        action_ledger=services.action_ledger,
        reasoning_ledger=services.reasoning_ledger,
        runtime_error_store=services.runtime_error_store,
        safety_policy=services.safety_policy,
        _transaction_lock=services._transaction_lock,
    )


def build_fixture_services(
    scenario: FixtureScenario | str,
) -> Phase2Services:
    """Create one isolated shared-lock service bundle for a fixed scenario."""

    selected = FixtureScenario(scenario)
    if selected is FixtureScenario.RUNTIME_BLOCKING:
        return Phase2Services.deterministic(
            initial_fixture=Phase2Fixture.ACTUATOR_RUNTIME_ERROR
        )

    services = Phase2Services.deterministic(
        initial_fixture=Phase2Fixture.COMFORTABLE_OCCUPIED
    )
    shared_lock = services._transaction_lock

    if selected is FixtureScenario.STALE_ONCE:
        return _replace_sensor_store(
            services,
            _StaleOnceSensorStore(lock=shared_lock),
        )

    services.runtime_error_store.inject(
        services.sensor_store.current.cycle_id,
        RuntimeErrorRecord(
            error_id="runtime-warning-000001",
            source=RuntimeErrorSource.CONTROL_LOOP,
            severity=RuntimeErrorSeverity.WARNING,
            code="TRANSIENT_SETPOINT_WARNING",
            summary=(
                "A prior mock setpoint request required a safe correction."
            ),
            retryable=True,
            correction_fields=("commands",),
            correction_hint=(
                "Release all zones to baseline control for this cycle."
            ),
        ),
    )
    return services


def _parse_scenario(
    argv: Sequence[str] | None,
) -> FixtureScenario:
    """Parse exactly one fixed scenario without accepting paths or admin input."""

    choices = tuple(item.value for item in FixtureScenario)
    parser = argparse.ArgumentParser(
        description="Run a fixed Phase 2 agent-test MCP fixture over stdio."
    )
    parser.add_argument("scenario", nargs="?", choices=choices)
    parser.add_argument(
        "--scenario",
        dest="scenario_option",
        choices=choices,
    )
    arguments = parser.parse_args(argv)
    supplied = tuple(
        value
        for value in (arguments.scenario, arguments.scenario_option)
        if value is not None
    )
    if len(supplied) != 1:
        parser.error("provide exactly one fixture scenario")
    return FixtureScenario(supplied[0])


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the existing five-tool MCP application over stdio."""

    scenario = _parse_scenario(argv)
    services = build_fixture_services(scenario)
    create_mcp_app(services).run(transport=DEFAULT_TRANSPORT)


if __name__ == "__main__":
    main()


__all__ = [
    "FixtureScenario",
    "build_fixture_services",
    "main",
]
