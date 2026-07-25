"""Canonical, transport-independent contracts for the Phase 2 tool boundary.

This module intentionally contains data contracts only.  It does not register MCP
tools, call an LLM, or connect to EnergyPlus.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)


SCHEMA_VERSION = "phase2.v1"

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
SummaryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[
    StrictFloat,
    Field(ge=0.0, allow_inf_nan=False),
]
Percentage = Annotated[
    StrictFloat,
    Field(ge=0.0, le=100.0, allow_inf_nan=False),
]
Confidence = Annotated[
    StrictFloat,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]


class ContractModel(BaseModel):
    """Strict base model shared by all externally visible contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class VersionedContract(ContractModel):
    """Contract carrying the one supported Phase 2 schema version."""

    schema_version: Literal["phase2.v1"] = SCHEMA_VERSION


class ToolRequest(VersionedContract):
    """Common request envelope used by every future MCP tool."""

    request_id: Identifier


class ToolResponse(VersionedContract):
    """Common response envelope correlated to its initiating request."""

    request_id: Identifier


class SensorSource(StrEnum):
    """Origin of a sensor snapshot."""

    MOCK = "mock"
    ENERGYPLUS = "energyplus"


class AvailableControl(StrEnum):
    """Actuator surface proven safe during Phase 1."""

    ZONE_THERMOSTAT_SETPOINTS = "zone_thermostat_setpoints"


class CarbonIntensityCategory(StrEnum):
    """Coarse carbon signal used by the controller."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ObjectiveTag(StrEnum):
    """Allowed concise objectives for a logged decision rationale."""

    ENERGY_REDUCTION = "energy_reduction"
    THERMAL_COMFORT = "thermal_comfort"
    CARBON_REDUCTION = "carbon_reduction"
    PEAK_DEMAND = "peak_demand"
    SAFETY = "safety"


class ControlActionStatus(StrEnum):
    """Outcome of a set-control request."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class RuntimeErrorSeverity(StrEnum):
    """EnergyPlus-compatible runtime error severity."""

    WARNING = "warning"
    SEVERE = "severe"
    FATAL = "fatal"


class RuntimeErrorSource(StrEnum):
    """Source stream from which a runtime error was parsed."""

    ENERGYPLUS_ERROR_FILE = "energyplus_error_file"
    ENERGYPLUS_CONSOLE = "energyplus_console"
    CONTROL_LOOP = "control_loop"


class SafetyErrorCode(StrEnum):
    """Stable machine-readable safety and tool rejection codes."""

    STALE_SNAPSHOT = "STALE_SNAPSHOT"
    UNKNOWN_ZONE = "UNKNOWN_ZONE"
    DUPLICATE_ZONE = "DUPLICATE_ZONE"
    MISSING_ZONE = "MISSING_ZONE"
    NONFINITE_VALUE = "NONFINITE_VALUE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    DEADBAND_VIOLATION = "DEADBAND_VIOLATION"
    OCCUPIED_COMFORT_VIOLATION = "OCCUPIED_COMFORT_VIOLATION"
    MISSING_REASONING_LOG = "MISSING_REASONING_LOG"
    UNSUPPORTED_CONTROL = "UNSUPPORTED_CONTROL"
    RUNTIME_ERROR_PENDING = "RUNTIME_ERROR_PENDING"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolError(ContractModel):
    """Structured rejection detail suitable for corrective retry prompts."""

    code: SafetyErrorCode
    field: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
    ] | None = None
    message: SummaryText
    retryable: StrictBool


class ZoneSensorData(ContractModel):
    """One controlled zone's observable state."""

    zone_id: Identifier
    air_temperature_c: FiniteFloat
    relative_humidity_pct: Percentage
    co2_ppm: NonNegativeFiniteFloat
    occupant_count: NonNegativeFiniteFloat
    fanger_pmv: FiniteFloat
    heating_setpoint_c: FiniteFloat
    cooling_setpoint_c: FiniteFloat

    @model_validator(mode="after")
    def validate_setpoint_order(self) -> ZoneSensorData:
        """Reject physically inverted reported thermostat setpoints."""

        if self.cooling_setpoint_c < self.heating_setpoint_c:
            raise ValueError(
                "cooling_setpoint_c must be greater than or equal to "
                "heating_setpoint_c"
            )
        return self


class SensorSnapshot(VersionedContract):
    """Canonical whole-building observation for one control cycle."""

    cycle_id: Identifier
    snapshot_id: Identifier
    sequence: Annotated[StrictInt, Field(ge=1)]
    timestamp: AwareDatetime
    timestep_minutes: Literal[15] = 15
    source: SensorSource
    available_controls: Annotated[
        tuple[AvailableControl, ...],
        Field(min_length=1, max_length=1),
    ] = (AvailableControl.ZONE_THERMOSTAT_SETPOINTS,)
    outdoor_drybulb_c: FiniteFloat
    facility_electricity_demand_w: NonNegativeFiniteFloat
    facility_electricity_kwh_since_start: NonNegativeFiniteFloat
    zones: Annotated[tuple[ZoneSensorData, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_zones(self) -> SensorSnapshot:
        """Ensure each sensor snapshot identifies a zone at most once."""

        zone_ids = [zone.zone_id for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("zones must contain unique zone_id values")
        return self


class ReadSensorDataRequest(ToolRequest):
    """Request the latest snapshot and a small bounded history window."""

    history_steps: Annotated[StrictInt, Field(ge=0, le=4)] = 0


class ReadSensorDataResponse(ToolResponse):
    """Current snapshot plus earlier snapshots in ascending sequence order."""

    snapshot: SensorSnapshot
    history: Annotated[
        tuple[SensorSnapshot, ...],
        Field(max_length=4),
    ] = ()

    @model_validator(mode="after")
    def validate_history(self) -> ReadSensorDataResponse:
        """Reject ambiguous, duplicate, or out-of-order history."""

        snapshots = (*self.history, self.snapshot)
        snapshot_ids = [item.snapshot_id for item in snapshots]
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("snapshot and history must have unique snapshot_id values")

        if any(
            earlier.sequence >= later.sequence
            for earlier, later in zip(snapshots, snapshots[1:], strict=False)
        ):
            raise ValueError(
                "history must be in ascending sequence order before snapshot"
            )
        if any(
            earlier.timestamp >= later.timestamp
            for earlier, later in zip(snapshots, snapshots[1:], strict=False)
        ):
            raise ValueError(
                "history must be in ascending timestamp order before snapshot"
            )
        return self


class GridCarbonIntensityRequest(ToolRequest):
    """Request the carbon signal correlated to a sensor snapshot."""

    snapshot_id: Identifier
    forecast_steps: Annotated[StrictInt, Field(ge=1, le=16)] = 4


class CarbonIntensitySample(ContractModel):
    """One current or forecast grid-carbon value."""

    offset_steps: Annotated[StrictInt, Field(ge=0, le=16)]
    timestamp: AwareDatetime
    g_co2_per_kwh: NonNegativeFiniteFloat
    category: CarbonIntensityCategory


class GridCarbonIntensityResponse(ToolResponse):
    """Current and forecast carbon signal from the deterministic fixture."""

    snapshot_id: Identifier
    source: Literal["mock_fixture"] = "mock_fixture"
    current: CarbonIntensitySample
    forecast: Annotated[
        tuple[CarbonIntensitySample, ...],
        Field(min_length=1, max_length=16),
    ]

    @model_validator(mode="after")
    def validate_forecast(self) -> GridCarbonIntensityResponse:
        """Require an offset-zero current sample and ordered future samples."""

        if self.current.offset_steps != 0:
            raise ValueError("current carbon sample must have offset_steps=0")
        offsets = [sample.offset_steps for sample in self.forecast]
        if any(offset <= 0 for offset in offsets):
            raise ValueError("forecast samples must have positive offset_steps")
        if offsets != sorted(set(offsets)):
            raise ValueError(
                "forecast offset_steps must be unique and strictly increasing"
            )
        samples = (self.current, *self.forecast)
        if any(
            earlier.timestamp >= later.timestamp
            for earlier, later in zip(samples, samples[1:], strict=False)
        ):
            raise ValueError("forecast timestamps must be strictly increasing")
        return self


class LogReasoningRequest(ToolRequest):
    """Record a concise decision rationale, never hidden chain-of-thought."""

    cycle_id: Identifier
    snapshot_id: Identifier
    decision_summary: SummaryText
    objective_tags: Annotated[
        tuple[ObjectiveTag, ...],
        Field(min_length=1, max_length=5),
    ]
    tradeoff_summary: SummaryText
    confidence: Confidence

    @model_validator(mode="after")
    def validate_unique_objectives(self) -> LogReasoningRequest:
        """Reject repeated objective labels."""

        if len(self.objective_tags) != len(set(self.objective_tags)):
            raise ValueError("objective_tags must be unique")
        return self


class LogReasoningResponse(ToolResponse):
    """Acknowledgement and stable identifier for a logged rationale."""

    cycle_id: Identifier
    snapshot_id: Identifier
    reasoning_log_id: Identifier
    logged: Literal[True] = True


class SetZoneCommand(ContractModel):
    """Set the two Phase 1-proven thermostat schedule actuators."""

    mode: Literal["set"] = "set"
    zone_id: Identifier
    heating_c: FiniteFloat
    cooling_c: FiniteFloat

    @model_validator(mode="after")
    def validate_setpoint_order(self) -> SetZoneCommand:
        """Reject inverted setpoints before contextual safety checks."""

        if self.cooling_c < self.heating_c:
            raise ValueError("cooling_c must be greater than or equal to heating_c")
        return self


class ReleaseZoneCommand(ContractModel):
    """Release one zone back to its baseline schedule."""

    mode: Literal["release"] = "release"
    zone_id: Identifier


ZoneCommand = Annotated[
    SetZoneCommand | ReleaseZoneCommand,
    Field(discriminator="mode"),
]


class SetControlActionRequest(ToolRequest):
    """Request one complete, correlated, idempotent building action."""

    cycle_id: Identifier
    snapshot_id: Identifier
    reasoning_log_id: Identifier
    idempotency_key: Identifier
    commands: Annotated[tuple[ZoneCommand, ...], Field(min_length=1, max_length=64)]
    hold_steps: Annotated[StrictInt, Field(ge=1, le=4)] = 1


class SetControlActionResponse(ToolResponse):
    """Structured acceptance or rejection of a control action."""

    cycle_id: Identifier
    snapshot_id: Identifier
    status: ControlActionStatus
    action_id: Identifier | None = None
    errors: Annotated[tuple[ToolError, ...], Field(max_length=64)] = ()

    @model_validator(mode="after")
    def validate_outcome(self) -> SetControlActionResponse:
        """Keep status, action identifier, and validation errors consistent."""

        if self.status is ControlActionStatus.REJECTED:
            if not self.errors:
                raise ValueError("rejected responses must include at least one error")
            if self.action_id is not None:
                raise ValueError("rejected responses cannot include action_id")
            return self

        if self.action_id is None:
            raise ValueError("accepted and duplicate responses require action_id")
        if self.errors:
            raise ValueError(
                "accepted and duplicate responses cannot include validation errors"
            )
        return self


class ParseRuntimeErrorsRequest(ToolRequest):
    """Request a bounded page of runtime errors for one cycle."""

    cycle_id: Identifier
    after_error_id: Identifier | None = None
    limit: Annotated[StrictInt, Field(ge=1, le=20)] = 20


class RuntimeErrorRecord(ContractModel):
    """Categorized simulator/runtime failure with a correction hint."""

    error_id: Identifier
    source: RuntimeErrorSource
    severity: RuntimeErrorSeverity
    code: Identifier
    summary: SummaryText
    action_id: Identifier | None = None
    retryable: StrictBool
    correction_fields: Annotated[
        tuple[Identifier, ...],
        Field(max_length=10),
    ] = ()
    correction_hint: SummaryText | None = None

    @model_validator(mode="after")
    def validate_unique_correction_fields(self) -> RuntimeErrorRecord:
        """Keep corrective field lists concise and unambiguous."""

        if len(self.correction_fields) != len(set(self.correction_fields)):
            raise ValueError("correction_fields must be unique")
        return self


class ParseRuntimeErrorsResponse(ToolResponse):
    """Categorized errors and pagination state for corrective retries."""

    cycle_id: Identifier
    errors: Annotated[tuple[RuntimeErrorRecord, ...], Field(max_length=20)] = ()
    next_error_id: Identifier | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_unique_errors(self) -> ParseRuntimeErrorsResponse:
        """Do not return the same runtime error more than once."""

        error_ids = [error.error_id for error in self.errors]
        if len(error_ids) != len(set(error_ids)):
            raise ValueError("errors must contain unique error_id values")
        if self.has_more and self.next_error_id is None:
            raise ValueError("has_more responses require next_error_id")
        return self


__all__ = [
    "SCHEMA_VERSION",
    "AvailableControl",
    "CarbonIntensityCategory",
    "CarbonIntensitySample",
    "ControlActionStatus",
    "GridCarbonIntensityRequest",
    "GridCarbonIntensityResponse",
    "LogReasoningRequest",
    "LogReasoningResponse",
    "ObjectiveTag",
    "ParseRuntimeErrorsRequest",
    "ParseRuntimeErrorsResponse",
    "ReadSensorDataRequest",
    "ReadSensorDataResponse",
    "ReleaseZoneCommand",
    "RuntimeErrorRecord",
    "RuntimeErrorSeverity",
    "RuntimeErrorSource",
    "SafetyErrorCode",
    "SensorSnapshot",
    "SensorSource",
    "SetControlActionRequest",
    "SetControlActionResponse",
    "SetZoneCommand",
    "ToolError",
    "ZoneCommand",
    "ZoneSensorData",
]
