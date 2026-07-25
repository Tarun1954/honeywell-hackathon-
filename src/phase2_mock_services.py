"""Deterministic in-memory services for the Phase 2 mocked control loop.

The services in this module are deliberately transport-agnostic.  They do not
register MCP tools, invoke an LLM, or connect to EnergyPlus.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

import yaml
from pydantic import TypeAdapter

from src.phase2_contracts import (
    CarbonIntensityCategory,
    CarbonIntensitySample,
    ControlActionStatus,
    GridCarbonIntensityRequest,
    GridCarbonIntensityResponse,
    Identifier,
    LogReasoningRequest,
    LogReasoningResponse,
    ObjectiveTag,
    ParseRuntimeErrorsRequest,
    ParseRuntimeErrorsResponse,
    ReadSensorDataRequest,
    ReadSensorDataResponse,
    RuntimeErrorRecord,
    RuntimeErrorSeverity,
    RuntimeErrorSource,
    SafetyErrorCode,
    SensorSnapshot,
    SensorSource,
    SetControlActionRequest,
    SetControlActionResponse,
    SetZoneCommand,
    ToolError,
    ZoneSensorData,
)
from src.phase2_validation import SafetyPolicy, validate_control_action


PHASE1_ZONE_IDS = (
    "SPACE1-1",
    "SPACE2-1",
    "SPACE3-1",
    "SPACE4-1",
    "SPACE5-1",
)
FIXTURE_START = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
TIMESTEP = timedelta(minutes=15)
DEFAULT_PHASE2_CONFIG = Path(__file__).resolve().parents[1] / "config" / "phase2.yaml"
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class Phase2Fixture(StrEnum):
    """Named deterministic scenarios used by the mocked services."""

    UNOCCUPIED_MILD = "unoccupied_mild_conditions"
    OCCUPIED_TOO_COLD = "occupied_too_cold"
    OCCUPIED_TOO_WARM = "occupied_too_warm"
    STALE_SNAPSHOT = "stale_snapshot"
    COMFORTABLE_OCCUPIED = "comfortable_occupied_conditions"
    HIGH_DEMAND = "high_demand"
    HIGH_MOCK_CARBON_INTENSITY = "high_mock_carbon_intensity"
    ACTUATOR_RUNTIME_ERROR = "actuator_runtime_error"
    INVALID_SETPOINT_ERROR = "invalid_setpoint_error"


class MockServiceError(RuntimeError):
    """Base class for deterministic service-layer failures."""


class FixtureTimelineExhaustedError(MockServiceError):
    """Raised when no later deterministic sensor snapshot exists."""


class DuplicateLedgerEntryError(MockServiceError):
    """Raised when an append would overwrite an existing immutable record."""


class UnknownRuntimeErrorCursorError(MockServiceError):
    """Raised when runtime-error pagination receives an unknown cycle cursor."""


def _canonical_identifier(value: str) -> str:
    """Validate and normalize an externally supplied identifier."""

    return _IDENTIFIER_ADAPTER.validate_python(value)


@dataclass(frozen=True, slots=True)
class _SensorProfile:
    """Fixed values used to construct one whole-building snapshot."""

    outdoor_drybulb_c: float
    facility_electricity_demand_w: float
    air_temperature_c: float
    relative_humidity_pct: float
    co2_ppm: float
    occupant_count: float
    fanger_pmv: float
    heating_setpoint_c: float
    cooling_setpoint_c: float


_SENSOR_PROFILES: Mapping[Phase2Fixture, _SensorProfile] = MappingProxyType(
    {
        Phase2Fixture.UNOCCUPIED_MILD: _SensorProfile(
            outdoor_drybulb_c=18.0,
            facility_electricity_demand_w=3_200.0,
            air_temperature_c=21.0,
            relative_humidity_pct=42.0,
            co2_ppm=425.0,
            occupant_count=0.0,
            fanger_pmv=0.0,
            heating_setpoint_c=18.0,
            cooling_setpoint_c=28.0,
        ),
        Phase2Fixture.OCCUPIED_TOO_COLD: _SensorProfile(
            outdoor_drybulb_c=5.0,
            facility_electricity_demand_w=9_500.0,
            air_temperature_c=18.0,
            relative_humidity_pct=38.0,
            co2_ppm=780.0,
            occupant_count=3.0,
            fanger_pmv=-1.25,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
        Phase2Fixture.OCCUPIED_TOO_WARM: _SensorProfile(
            outdoor_drybulb_c=35.0,
            facility_electricity_demand_w=12_000.0,
            air_temperature_c=28.0,
            relative_humidity_pct=52.0,
            co2_ppm=820.0,
            occupant_count=3.0,
            fanger_pmv=1.35,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
        Phase2Fixture.STALE_SNAPSHOT: _SensorProfile(
            outdoor_drybulb_c=24.0,
            facility_electricity_demand_w=7_200.0,
            air_temperature_c=22.3,
            relative_humidity_pct=45.0,
            co2_ppm=700.0,
            occupant_count=2.0,
            fanger_pmv=0.05,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
        Phase2Fixture.COMFORTABLE_OCCUPIED: _SensorProfile(
            outdoor_drybulb_c=24.5,
            facility_electricity_demand_w=7_400.0,
            air_temperature_c=22.5,
            relative_humidity_pct=45.0,
            co2_ppm=720.0,
            occupant_count=2.0,
            fanger_pmv=0.1,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
        Phase2Fixture.HIGH_DEMAND: _SensorProfile(
            outdoor_drybulb_c=33.0,
            facility_electricity_demand_w=32_000.0,
            air_temperature_c=23.5,
            relative_humidity_pct=48.0,
            co2_ppm=760.0,
            occupant_count=3.0,
            fanger_pmv=0.35,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
        Phase2Fixture.HIGH_MOCK_CARBON_INTENSITY: _SensorProfile(
            outdoor_drybulb_c=30.0,
            facility_electricity_demand_w=10_500.0,
            air_temperature_c=22.8,
            relative_humidity_pct=46.0,
            co2_ppm=735.0,
            occupant_count=2.0,
            fanger_pmv=0.15,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
        Phase2Fixture.ACTUATOR_RUNTIME_ERROR: _SensorProfile(
            outdoor_drybulb_c=29.0,
            facility_electricity_demand_w=9_800.0,
            air_temperature_c=22.7,
            relative_humidity_pct=46.0,
            co2_ppm=730.0,
            occupant_count=2.0,
            fanger_pmv=0.12,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
        Phase2Fixture.INVALID_SETPOINT_ERROR: _SensorProfile(
            outdoor_drybulb_c=28.0,
            facility_electricity_demand_w=9_000.0,
            air_temperature_c=22.4,
            relative_humidity_pct=44.0,
            co2_ppm=710.0,
            occupant_count=2.0,
            fanger_pmv=0.08,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        ),
    }
)

_BASE_CARBON_INTENSITY: Mapping[Phase2Fixture, float] = MappingProxyType(
    {
        Phase2Fixture.UNOCCUPIED_MILD: 260.0,
        Phase2Fixture.OCCUPIED_TOO_COLD: 315.0,
        Phase2Fixture.OCCUPIED_TOO_WARM: 420.0,
        Phase2Fixture.STALE_SNAPSHOT: 340.0,
        Phase2Fixture.COMFORTABLE_OCCUPIED: 350.0,
        Phase2Fixture.HIGH_DEMAND: 460.0,
        Phase2Fixture.HIGH_MOCK_CARBON_INTENSITY: 780.0,
        Phase2Fixture.ACTUATOR_RUNTIME_ERROR: 390.0,
        Phase2Fixture.INVALID_SETPOINT_ERROR: 370.0,
    }
)

_CARBON_FORECAST_DELTAS = (
    12.0,
    24.0,
    18.0,
    6.0,
    -8.0,
    -16.0,
    -10.0,
    4.0,
    20.0,
    30.0,
    22.0,
    8.0,
    -6.0,
    -18.0,
    -12.0,
    2.0,
)


def _build_sensor_timeline() -> tuple[SensorSnapshot, ...]:
    """Build the immutable nine-scenario fixture timeline."""

    snapshots: list[SensorSnapshot] = []
    cumulative_kwh = 0.0
    for sequence, (fixture, profile) in enumerate(
        _SENSOR_PROFILES.items(),
        start=1,
    ):
        cumulative_kwh += (
            profile.facility_electricity_demand_w
            * TIMESTEP.total_seconds()
            / 3_600_000.0
        )
        zones = tuple(
            ZoneSensorData(
                zone_id=zone_id,
                air_temperature_c=profile.air_temperature_c,
                relative_humidity_pct=profile.relative_humidity_pct,
                co2_ppm=profile.co2_ppm,
                occupant_count=profile.occupant_count,
                fanger_pmv=profile.fanger_pmv,
                heating_setpoint_c=profile.heating_setpoint_c,
                cooling_setpoint_c=profile.cooling_setpoint_c,
            )
            for zone_id in PHASE1_ZONE_IDS
        )
        snapshots.append(
            SensorSnapshot(
                cycle_id=f"cycle-{sequence:04d}",
                snapshot_id=f"snapshot-{sequence:04d}-{fixture.value}",
                sequence=sequence,
                timestamp=FIXTURE_START + (sequence - 1) * TIMESTEP,
                source=SensorSource.MOCK,
                outdoor_drybulb_c=profile.outdoor_drybulb_c,
                facility_electricity_demand_w=(
                    profile.facility_electricity_demand_w
                ),
                facility_electricity_kwh_since_start=cumulative_kwh,
                zones=zones,
            )
        )
    return tuple(snapshots)


class MockSensorStore:
    """Read and advance through a fixed 15-minute sensor timeline."""

    def __init__(
        self,
        *,
        initial_fixture: Phase2Fixture | str = (
            Phase2Fixture.COMFORTABLE_OCCUPIED
        ),
        lock: RLock | None = None,
    ) -> None:
        selected_fixture = Phase2Fixture(initial_fixture)
        self._timeline = _build_sensor_timeline()
        self._fixture_to_index = {
            fixture: index for index, fixture in enumerate(_SENSOR_PROFILES)
        }
        self._snapshot_by_id = {
            snapshot.snapshot_id: snapshot for snapshot in self._timeline
        }
        self._current_index = self._fixture_to_index[selected_fixture]
        if selected_fixture == Phase2Fixture.STALE_SNAPSHOT:
            self._current_index += 1
        self._lock = lock or RLock()

    @property
    def timeline(self) -> tuple[SensorSnapshot, ...]:
        """Return all immutable fixtures in deterministic order."""

        return self._timeline

    @property
    def current(self) -> SensorSnapshot:
        """Return the current snapshot without advancing time."""

        with self._lock:
            return self._timeline[self._current_index]

    def snapshot_for(
        self,
        fixture: Phase2Fixture | str,
    ) -> SensorSnapshot:
        """Return the immutable sensor snapshot for a named fixture."""

        return self._timeline[
            self._fixture_to_index[Phase2Fixture(fixture)]
        ]

    def get(self, snapshot_id: str) -> SensorSnapshot:
        """Return a known snapshot by its immutable identifier."""

        canonical_id = _canonical_identifier(snapshot_id)
        try:
            return self._snapshot_by_id[canonical_id]
        except KeyError as exc:
            raise KeyError(f"unknown snapshot_id: {canonical_id}") from exc

    def is_stale(self, snapshot_id: str) -> bool:
        """Fail closed when an ID is unknown or is not the current snapshot."""

        canonical_id = _canonical_identifier(snapshot_id)
        with self._lock:
            return (
                canonical_id not in self._snapshot_by_id
                or canonical_id
                != self._timeline[self._current_index].snapshot_id
            )

    def advance(self) -> SensorSnapshot:
        """Advance exactly one 15-minute fixture step."""

        with self._lock:
            if self._current_index + 1 >= len(self._timeline):
                raise FixtureTimelineExhaustedError(
                    "the deterministic sensor timeline is exhausted"
                )
            self._current_index += 1
            return self._timeline[self._current_index]

    def read(self, request: ReadSensorDataRequest) -> ReadSensorDataResponse:
        """Return current state and the requested bounded prior history."""

        with self._lock:
            current = self._timeline[self._current_index]
            history_start = max(0, self._current_index - request.history_steps)
            history = self._timeline[history_start : self._current_index]
        return ReadSensorDataResponse(
            request_id=request.request_id,
            snapshot=current,
            history=history,
        )


def _carbon_category(value: float) -> CarbonIntensityCategory:
    """Map a fixed numerical intensity to a stable coarse category."""

    if value < 300.0:
        return CarbonIntensityCategory.LOW
    if value < 500.0:
        return CarbonIntensityCategory.MODERATE
    if value < 700.0:
        return CarbonIntensityCategory.HIGH
    return CarbonIntensityCategory.VERY_HIGH


class MockGridCarbonStore:
    """Provide fixed carbon values keyed by deterministic snapshot IDs."""

    def __init__(self, sensor_store: MockSensorStore) -> None:
        self._snapshot_by_id = {
            snapshot.snapshot_id: snapshot for snapshot in sensor_store.timeline
        }
        self._base_by_snapshot_id = {
            sensor_store.snapshot_for(fixture).snapshot_id: value
            for fixture, value in _BASE_CARBON_INTENSITY.items()
        }

    def read(
        self,
        request: GridCarbonIntensityRequest,
    ) -> GridCarbonIntensityResponse:
        """Return current and requested forecast samples for one snapshot."""

        try:
            snapshot = self._snapshot_by_id[request.snapshot_id]
            base_value = self._base_by_snapshot_id[request.snapshot_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown snapshot_id for grid carbon: {request.snapshot_id}"
            ) from exc

        current = CarbonIntensitySample(
            offset_steps=0,
            timestamp=snapshot.timestamp,
            g_co2_per_kwh=base_value,
            category=_carbon_category(base_value),
        )
        forecast = tuple(
            CarbonIntensitySample(
                offset_steps=offset,
                timestamp=snapshot.timestamp + offset * TIMESTEP,
                g_co2_per_kwh=max(
                    0.0,
                    base_value + _CARBON_FORECAST_DELTAS[offset - 1],
                ),
                category=_carbon_category(
                    max(
                        0.0,
                        base_value + _CARBON_FORECAST_DELTAS[offset - 1],
                    )
                ),
            )
            for offset in range(1, request.forecast_steps + 1)
        )
        return GridCarbonIntensityResponse(
            request_id=request.request_id,
            snapshot_id=request.snapshot_id,
            current=current,
            forecast=forecast,
        )


@dataclass(frozen=True, slots=True)
class ReasoningLedgerEntry:
    """One immutable reasoning request and acknowledgement."""

    request: LogReasoningRequest
    response: LogReasoningResponse


class ReasoningLedger:
    """Append-only deterministic log of concise reasoning summaries."""

    def __init__(self, *, lock: RLock | None = None) -> None:
        self._entries: list[ReasoningLedgerEntry] = []
        self._by_log_id: dict[str, ReasoningLedgerEntry] = {}
        self._by_request_id: dict[str, ReasoningLedgerEntry] = {}
        self._next_sequence = 1
        self._lock = lock or RLock()

    @property
    def records(self) -> tuple[ReasoningLedgerEntry, ...]:
        """Return an immutable snapshot of entries in append order."""

        with self._lock:
            return tuple(self._entries)

    @property
    def correlations(self) -> Mapping[str, tuple[str, str]]:
        """Return read-only log-to-cycle/snapshot correlations."""

        with self._lock:
            correlations = {
                log_id: (
                    entry.request.cycle_id,
                    entry.request.snapshot_id,
                )
                for log_id, entry in self._by_log_id.items()
            }
        return MappingProxyType(correlations)

    def append(
        self,
        request: LogReasoningRequest,
        *,
        reasoning_log_id: str | None = None,
    ) -> LogReasoningResponse:
        """Append once, treating an identical request retry idempotently."""

        with self._lock:
            canonical_log_id = (
                None
                if reasoning_log_id is None
                else _canonical_identifier(reasoning_log_id)
            )
            existing_request = self._by_request_id.get(request.request_id)
            if existing_request is not None:
                if existing_request.request != request:
                    raise DuplicateLedgerEntryError(
                        f"reasoning request_id {request.request_id!r} "
                        "already belongs to different content"
                    )
                if (
                    canonical_log_id is not None
                    and canonical_log_id
                    != existing_request.response.reasoning_log_id
                ):
                    raise DuplicateLedgerEntryError(
                        "an idempotent reasoning retry cannot change "
                        "reasoning_log_id"
                    )
                return existing_request.response

            selected_id = (
                self._next_available_id()
                if canonical_log_id is None
                else canonical_log_id
            )
            if selected_id in self._by_log_id:
                raise DuplicateLedgerEntryError(
                    f"reasoning_log_id {selected_id!r} already exists"
                )

            response = LogReasoningResponse(
                request_id=request.request_id,
                cycle_id=request.cycle_id,
                snapshot_id=request.snapshot_id,
                reasoning_log_id=selected_id,
            )
            entry = ReasoningLedgerEntry(request=request, response=response)
            self._entries.append(entry)
            self._by_log_id[selected_id] = entry
            self._by_request_id[request.request_id] = entry
            return response

    def _next_available_id(self) -> str:
        """Reserve the next deterministic reasoning identifier."""

        while True:
            candidate = f"reasoning-log-{self._next_sequence:06d}"
            self._next_sequence += 1
            if candidate not in self._by_log_id:
                return candidate


@dataclass(frozen=True, slots=True)
class ActionLedgerEntry:
    """One accepted immutable control action."""

    request: SetControlActionRequest
    response: SetControlActionResponse

    @property
    def action_id(self) -> str:
        """Return the accepted response's required action identifier."""

        if self.response.action_id is None:
            raise AssertionError("accepted action entry has no action_id")
        return self.response.action_id


class MockActionLedger:
    """Validate and append accepted actions with safe duplicate handling."""

    def __init__(self, *, lock: RLock | None = None) -> None:
        self._entries: list[ActionLedgerEntry] = []
        self._by_action_id: dict[str, ActionLedgerEntry] = {}
        self._by_idempotency_key: dict[str, ActionLedgerEntry] = {}
        self._next_sequence = 1
        self._lock = lock or RLock()

    @property
    def records(self) -> tuple[ActionLedgerEntry, ...]:
        """Return accepted actions in immutable append order."""

        with self._lock:
            return tuple(self._entries)

    def submit(
        self,
        request: SetControlActionRequest,
        *,
        snapshot: SensorSnapshot,
        policy: SafetyPolicy,
        known_reasoning_logs: Mapping[str, tuple[str, str]],
        runtime_error_pending: bool = False,
        action_id: str | None = None,
    ) -> SetControlActionResponse:
        """Validate and append one action or return a structured safe outcome."""

        with self._lock:
            canonical_action_id = (
                None if action_id is None else _canonical_identifier(action_id)
            )
            existing_key = self._by_idempotency_key.get(request.idempotency_key)
            if existing_key is not None:
                if self._logical_payload(existing_key.request) != self._logical_payload(
                    request
                ):
                    return self._rejected(
                        request,
                        ToolError(
                            code=SafetyErrorCode.INTERNAL_ERROR,
                            field="idempotency_key",
                            message=(
                                "idempotency_key already belongs to a different "
                                "control action"
                            ),
                            retryable=False,
                        ),
                    )
                return SetControlActionResponse(
                    request_id=request.request_id,
                    cycle_id=request.cycle_id,
                    snapshot_id=request.snapshot_id,
                    status=ControlActionStatus.DUPLICATE,
                    action_id=existing_key.action_id,
                )

            errors = validate_control_action(
                request,
                snapshot=snapshot,
                policy=policy,
                known_reasoning_logs=known_reasoning_logs,
                runtime_error_pending=runtime_error_pending,
            )
            if errors:
                return SetControlActionResponse(
                    request_id=request.request_id,
                    cycle_id=request.cycle_id,
                    snapshot_id=request.snapshot_id,
                    status=ControlActionStatus.REJECTED,
                    errors=errors,
                )

            selected_id = (
                self._next_available_id()
                if canonical_action_id is None
                else canonical_action_id
            )
            if selected_id in self._by_action_id:
                return self._rejected(
                    request,
                    ToolError(
                        code=SafetyErrorCode.INTERNAL_ERROR,
                        field="action_id",
                        message=f"action_id {selected_id!r} already exists",
                        retryable=False,
                    ),
                )

            response = SetControlActionResponse(
                request_id=request.request_id,
                cycle_id=request.cycle_id,
                snapshot_id=request.snapshot_id,
                status=ControlActionStatus.ACCEPTED,
                action_id=selected_id,
            )
            entry = ActionLedgerEntry(request=request, response=response)
            self._entries.append(entry)
            self._by_action_id[selected_id] = entry
            self._by_idempotency_key[request.idempotency_key] = entry
            return response

    @staticmethod
    def _logical_payload(request: SetControlActionRequest) -> dict[str, Any]:
        """Exclude retry-envelope identity from idempotency comparison."""

        return request.model_dump(
            mode="json",
            exclude={"request_id"},
        )

    @staticmethod
    def _rejected(
        request: SetControlActionRequest,
        error: ToolError,
    ) -> SetControlActionResponse:
        """Build a one-error rejection response."""

        return SetControlActionResponse(
            request_id=request.request_id,
            cycle_id=request.cycle_id,
            snapshot_id=request.snapshot_id,
            status=ControlActionStatus.REJECTED,
            errors=(error,),
        )

    def _next_available_id(self) -> str:
        """Reserve the next deterministic action identifier."""

        while True:
            candidate = f"action-{self._next_sequence:06d}"
            self._next_sequence += 1
            if candidate not in self._by_action_id:
                return candidate


@dataclass(frozen=True, slots=True)
class RuntimeErrorLedgerEntry:
    """One immutable runtime error correlated to a control cycle."""

    cycle_id: str
    error: RuntimeErrorRecord


class MockRuntimeErrorStore:
    """Append-only cycle-scoped runtime-error injection and retrieval."""

    def __init__(self, *, lock: RLock | None = None) -> None:
        self._entries: list[RuntimeErrorLedgerEntry] = []
        self._by_error_id: dict[str, RuntimeErrorLedgerEntry] = {}
        self._next_sequence = 1
        self._lock = lock or RLock()

    @property
    def records(self) -> tuple[RuntimeErrorLedgerEntry, ...]:
        """Return errors in immutable append order."""

        with self._lock:
            return tuple(self._entries)

    def inject(
        self,
        cycle_id: str,
        error: RuntimeErrorRecord,
    ) -> RuntimeErrorRecord:
        """Append an externally constructed error without overwriting IDs."""

        canonical_cycle_id = _canonical_identifier(cycle_id)
        with self._lock:
            if error.error_id in self._by_error_id:
                raise DuplicateLedgerEntryError(
                    f"runtime error_id {error.error_id!r} already exists"
                )
            entry = RuntimeErrorLedgerEntry(
                cycle_id=canonical_cycle_id,
                error=error,
            )
            self._entries.append(entry)
            self._by_error_id[error.error_id] = entry
            return error

    def inject_fixture(
        self,
        cycle_id: str,
        fixture: Phase2Fixture | str,
        *,
        action_id: str | None = None,
        error_id: str | None = None,
    ) -> RuntimeErrorRecord:
        """Append one of the two deterministic runtime-error fixtures."""

        selected_fixture = Phase2Fixture(fixture)
        if selected_fixture not in (
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
            Phase2Fixture.INVALID_SETPOINT_ERROR,
        ):
            raise ValueError(
                f"{selected_fixture.value!r} is not a runtime-error fixture"
            )

        canonical_cycle_id = _canonical_identifier(cycle_id)
        canonical_action_id = (
            None if action_id is None else _canonical_identifier(action_id)
        )
        canonical_error_id = (
            None if error_id is None else _canonical_identifier(error_id)
        )
        with self._lock:
            selected_id = (
                self._next_available_id()
                if canonical_error_id is None
                else canonical_error_id
            )
            if selected_fixture == Phase2Fixture.ACTUATOR_RUNTIME_ERROR:
                error = RuntimeErrorRecord(
                    error_id=selected_id,
                    source=RuntimeErrorSource.ENERGYPLUS_ERROR_FILE,
                    severity=RuntimeErrorSeverity.SEVERE,
                    code="ACTUATOR_WRITEBACK_FAILED",
                    summary=(
                        "Mock actuator handle rejected the thermostat writeback."
                    ),
                    action_id=canonical_action_id,
                    retryable=True,
                    correction_fields=("commands",),
                    correction_hint=(
                        "Release the affected zone or submit supported thermostat "
                        "setpoints."
                    ),
                )
            else:
                error = RuntimeErrorRecord(
                    error_id=selected_id,
                    source=RuntimeErrorSource.CONTROL_LOOP,
                    severity=RuntimeErrorSeverity.SEVERE,
                    code="INVALID_SETPOINT",
                    summary="Mock runtime rejected an unsafe thermostat setpoint.",
                    action_id=canonical_action_id,
                    retryable=True,
                    correction_fields=("heating_c", "cooling_c"),
                    correction_hint=(
                        "Use configured bounds and preserve the minimum deadband."
                    ),
                )
            return self.inject(canonical_cycle_id, error)

    def retrieve(
        self,
        request: ParseRuntimeErrorsRequest,
    ) -> ParseRuntimeErrorsResponse:
        """Return a stable, exclusive-cursor page for one control cycle."""

        with self._lock:
            cycle_errors = [
                entry.error
                for entry in self._entries
                if entry.cycle_id == request.cycle_id
            ]
            start = 0
            if request.after_error_id is not None:
                matching_indices = [
                    index
                    for index, error in enumerate(cycle_errors)
                    if error.error_id == request.after_error_id
                ]
                if not matching_indices:
                    raise UnknownRuntimeErrorCursorError(
                        f"error cursor {request.after_error_id!r} does not "
                        f"belong to cycle {request.cycle_id!r}"
                    )
                start = matching_indices[0] + 1

            page = tuple(cycle_errors[start : start + request.limit])
            has_more = start + len(page) < len(cycle_errors)
            next_error_id = page[-1].error_id if has_more else None

        return ParseRuntimeErrorsResponse(
            request_id=request.request_id,
            cycle_id=request.cycle_id,
            errors=page,
            next_error_id=next_error_id,
            has_more=has_more,
        )

    def has_blocking_error(self, cycle_id: str) -> bool:
        """Return whether the cycle contains any severe or fatal error."""

        canonical_cycle_id = _canonical_identifier(cycle_id)
        with self._lock:
            return any(
                entry.cycle_id == canonical_cycle_id
                and entry.error.severity
                in (RuntimeErrorSeverity.SEVERE, RuntimeErrorSeverity.FATAL)
                for entry in self._entries
            )

    def _next_available_id(self) -> str:
        """Reserve the next deterministic runtime-error identifier."""

        while True:
            candidate = f"runtime-error-{self._next_sequence:06d}"
            self._next_sequence += 1
            if candidate not in self._by_error_id:
                return candidate


def build_reasoning_fixture(
    fixture: Phase2Fixture | str,
    snapshot: SensorSnapshot,
) -> LogReasoningRequest:
    """Build a deterministic concise rationale for a fixture snapshot."""

    selected_fixture = Phase2Fixture(fixture)
    return LogReasoningRequest(
        request_id=f"fixture-reasoning-request-{selected_fixture.value}",
        cycle_id=snapshot.cycle_id,
        snapshot_id=snapshot.snapshot_id,
        decision_summary=(
            f"Evaluate the deterministic {selected_fixture.value} scenario."
        ),
        objective_tags=(
            ObjectiveTag.THERMAL_COMFORT,
            ObjectiveTag.ENERGY_REDUCTION,
            ObjectiveTag.SAFETY,
        ),
        tradeoff_summary=(
            "Preserve the configured comfort envelope while limiting demand."
        ),
        confidence=0.9,
    )


def build_control_action_fixture(
    fixture: Phase2Fixture | str,
    snapshot: SensorSnapshot,
    reasoning_log_id: str,
) -> SetControlActionRequest:
    """Build a safe action, except for the explicit invalid-setpoint fixture."""

    selected_fixture = Phase2Fixture(fixture)
    if selected_fixture == Phase2Fixture.UNOCCUPIED_MILD:
        heating_c, cooling_c = 18.0, 28.0
    elif selected_fixture == Phase2Fixture.OCCUPIED_TOO_COLD:
        heating_c, cooling_c = 21.0, 25.0
    elif selected_fixture == Phase2Fixture.OCCUPIED_TOO_WARM:
        heating_c, cooling_c = 20.0, 25.0
    elif selected_fixture == Phase2Fixture.INVALID_SETPOINT_ERROR:
        heating_c, cooling_c = 15.0, 31.0
    else:
        heating_c, cooling_c = 20.0, 26.0

    return SetControlActionRequest(
        request_id=f"fixture-action-request-{selected_fixture.value}",
        cycle_id=snapshot.cycle_id,
        snapshot_id=snapshot.snapshot_id,
        reasoning_log_id=reasoning_log_id,
        idempotency_key=f"fixture-action-key-{selected_fixture.value}",
        commands=tuple(
            SetZoneCommand(
                zone_id=zone_id,
                heating_c=heating_c,
                cooling_c=cooling_c,
            )
            for zone_id in PHASE1_ZONE_IDS
        ),
        hold_steps=1,
    )


def _load_safety_policy(config_path: Path) -> SafetyPolicy:
    """Load only the validated safety section from Phase 2 YAML."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("safety"), dict):
        raise ValueError(f"Phase 2 config has no safety mapping: {config_path}")
    return SafetyPolicy.model_validate(payload["safety"])


@dataclass(frozen=True, slots=True)
class Phase2Services:
    """Coherent dependency bundle for the future mocked Phase 2 loop."""

    sensor_store: MockSensorStore
    grid_carbon_store: MockGridCarbonStore
    action_ledger: MockActionLedger
    reasoning_ledger: ReasoningLedger
    runtime_error_store: MockRuntimeErrorStore
    safety_policy: SafetyPolicy
    _transaction_lock: RLock = field(repr=False, compare=False)

    @classmethod
    def deterministic(
        cls,
        *,
        initial_fixture: Phase2Fixture | str = (
            Phase2Fixture.COMFORTABLE_OCCUPIED
        ),
        safety_policy: SafetyPolicy | None = None,
        config_path: str | Path = DEFAULT_PHASE2_CONFIG,
    ) -> Phase2Services:
        """Create independent stores backed by the same fixed fixture data."""

        selected_fixture = Phase2Fixture(initial_fixture)
        policy = safety_policy or _load_safety_policy(Path(config_path))
        transaction_lock = RLock()
        sensor_store = MockSensorStore(
            initial_fixture=selected_fixture,
            lock=transaction_lock,
        )
        services = cls(
            sensor_store=sensor_store,
            grid_carbon_store=MockGridCarbonStore(sensor_store),
            action_ledger=MockActionLedger(lock=transaction_lock),
            reasoning_ledger=ReasoningLedger(lock=transaction_lock),
            runtime_error_store=MockRuntimeErrorStore(lock=transaction_lock),
            safety_policy=policy,
            _transaction_lock=transaction_lock,
        )
        if selected_fixture in (
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
            Phase2Fixture.INVALID_SETPOINT_ERROR,
        ):
            services.runtime_error_store.inject_fixture(
                services.sensor_store.current.cycle_id,
                selected_fixture,
            )
        return services

    def log_reasoning(
        self,
        request: LogReasoningRequest,
    ) -> LogReasoningResponse:
        """Append a reasoning summary to this bundle's ledger."""

        with self._transaction_lock:
            return self.reasoning_ledger.append(request)

    def submit_action(
        self,
        request: SetControlActionRequest,
        *,
        action_id: str | None = None,
    ) -> SetControlActionResponse:
        """Submit against current state and any cycle-scoped blocking error."""

        with self._transaction_lock:
            return self.action_ledger.submit(
                request,
                snapshot=self.sensor_store.current,
                policy=self.safety_policy,
                known_reasoning_logs=self.reasoning_ledger.correlations,
                runtime_error_pending=self.runtime_error_store.has_blocking_error(
                    request.cycle_id
                ),
                action_id=action_id,
            )


__all__ = [
    "DEFAULT_PHASE2_CONFIG",
    "FIXTURE_START",
    "PHASE1_ZONE_IDS",
    "TIMESTEP",
    "ActionLedgerEntry",
    "DuplicateLedgerEntryError",
    "FixtureTimelineExhaustedError",
    "MockActionLedger",
    "MockGridCarbonStore",
    "MockRuntimeErrorStore",
    "MockSensorStore",
    "MockServiceError",
    "Phase2Fixture",
    "Phase2Services",
    "ReasoningLedger",
    "ReasoningLedgerEntry",
    "RuntimeErrorLedgerEntry",
    "UnknownRuntimeErrorCursorError",
    "build_control_action_fixture",
    "build_reasoning_fixture",
]
