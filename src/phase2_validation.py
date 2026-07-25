"""Pure, config-driven safety validation for Phase 2 control contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated

from pydantic import Field, StrictFloat, StrictInt, model_validator

from src.phase2_contracts import (
    ContractModel,
    Identifier,
    ReleaseZoneCommand,
    SafetyErrorCode,
    SensorSnapshot,
    SetControlActionRequest,
    SetZoneCommand,
    ToolError,
)


PolicyFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
PositivePolicyFloat = Annotated[
    StrictFloat,
    Field(gt=0.0, allow_inf_nan=False),
]


class SafetyPolicy(ContractModel):
    """Configurable hard and occupied-zone limits for one deployment."""

    controlled_zones: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=64),
    ]
    heating_minimum_c: PolicyFloat
    heating_maximum_c: PolicyFloat
    cooling_minimum_c: PolicyFloat
    cooling_maximum_c: PolicyFloat
    minimum_deadband_c: PositivePolicyFloat
    occupied_heating_minimum_c: PolicyFloat
    occupied_cooling_maximum_c: PolicyFloat
    maximum_absolute_pmv: PositivePolicyFloat
    maximum_hold_steps: Annotated[StrictInt, Field(ge=1, le=4)]

    @model_validator(mode="after")
    def validate_policy_consistency(self) -> SafetyPolicy:
        """Reject contradictory or incomplete safety configurations."""

        if len(self.controlled_zones) != len(set(self.controlled_zones)):
            raise ValueError("controlled_zones must be unique")
        if self.heating_minimum_c > self.heating_maximum_c:
            raise ValueError(
                "heating_minimum_c must not exceed heating_maximum_c"
            )
        if self.cooling_minimum_c > self.cooling_maximum_c:
            raise ValueError(
                "cooling_minimum_c must not exceed cooling_maximum_c"
            )
        if not (
            self.heating_minimum_c
            <= self.occupied_heating_minimum_c
            <= self.heating_maximum_c
        ):
            raise ValueError(
                "occupied_heating_minimum_c must be inside the heating range"
            )
        if not (
            self.cooling_minimum_c
            <= self.occupied_cooling_maximum_c
            <= self.cooling_maximum_c
        ):
            raise ValueError(
                "occupied_cooling_maximum_c must be inside the cooling range"
            )
        if (
            self.occupied_cooling_maximum_c
            - self.occupied_heating_minimum_c
            < self.minimum_deadband_c
        ):
            raise ValueError(
                "occupied comfort bounds cannot satisfy minimum_deadband_c"
            )
        return self


def _issue(
    code: SafetyErrorCode,
    field: str,
    message: str,
    *,
    retryable: bool = True,
) -> ToolError:
    """Create one consistently shaped validation issue."""

    return ToolError(
        code=code,
        field=field,
        message=message,
        retryable=retryable,
    )


def validate_control_action(
    request: SetControlActionRequest,
    *,
    snapshot: SensorSnapshot,
    policy: SafetyPolicy,
    known_reasoning_logs: Mapping[str, tuple[str, str]],
    runtime_error_pending: bool = False,
) -> tuple[ToolError, ...]:
    """Return every safety issue without mutating or clamping the request.

    Structural failures are handled by Pydantic when the request is parsed.
    This function handles stateful and configuration-dependent rules needed at
    the eventual tool boundary.
    """

    issues: list[ToolError] = []

    if request.cycle_id != snapshot.cycle_id:
        issues.append(
            _issue(
                SafetyErrorCode.STALE_SNAPSHOT,
                "cycle_id",
                "control cycle does not match the current sensor cycle",
            )
        )
    if request.snapshot_id != snapshot.snapshot_id:
        issues.append(
            _issue(
                SafetyErrorCode.STALE_SNAPSHOT,
                "snapshot_id",
                "control action does not target the current sensor snapshot",
            )
        )
    reasoning_correlation = known_reasoning_logs.get(request.reasoning_log_id)
    if reasoning_correlation is None:
        issues.append(
            _issue(
                SafetyErrorCode.MISSING_REASONING_LOG,
                "reasoning_log_id",
                "reasoning_log_id is not registered for this control cycle",
            )
        )
    elif reasoning_correlation != (request.cycle_id, request.snapshot_id):
        issues.append(
            _issue(
                SafetyErrorCode.MISSING_REASONING_LOG,
                "reasoning_log_id",
                "reasoning_log_id is not correlated to this cycle and snapshot",
            )
        )
    if request.hold_steps > policy.maximum_hold_steps:
        issues.append(
            _issue(
                SafetyErrorCode.OUT_OF_RANGE,
                "hold_steps",
                f"hold_steps exceeds configured maximum "
                f"{policy.maximum_hold_steps}",
            )
        )
    if runtime_error_pending:
        issues.append(
            _issue(
                SafetyErrorCode.RUNTIME_ERROR_PENDING,
                "runtime_error_pending",
                "a severe or fatal runtime error must be resolved first",
            )
        )

    configured_zones = set(policy.controlled_zones)
    snapshot_zone_ids = {zone.zone_id for zone in snapshot.zones}
    for index, zone in enumerate(snapshot.zones):
        if zone.zone_id not in configured_zones:
            issues.append(
                _issue(
                    SafetyErrorCode.UNKNOWN_ZONE,
                    f"snapshot.zones[{index}].zone_id",
                    f"sensor zone {zone.zone_id!r} is not configured for control",
                    retryable=False,
                )
            )
    for zone_id in policy.controlled_zones:
        if zone_id not in snapshot_zone_ids:
            issues.append(
                _issue(
                    SafetyErrorCode.MISSING_ZONE,
                    "snapshot.zones",
                    f"sensor snapshot is missing configured zone {zone_id!r}",
                    retryable=False,
                )
            )

    seen_zones: set[str] = set()
    for index, command in enumerate(request.commands):
        field_prefix = f"commands[{index}]"
        zone_id = command.zone_id
        if zone_id not in configured_zones:
            issues.append(
                _issue(
                    SafetyErrorCode.UNKNOWN_ZONE,
                    f"{field_prefix}.zone_id",
                    f"zone {zone_id!r} is not configured for control",
                )
            )
        if zone_id in seen_zones:
            issues.append(
                _issue(
                    SafetyErrorCode.DUPLICATE_ZONE,
                    f"{field_prefix}.zone_id",
                    f"zone {zone_id!r} appears more than once",
                )
            )
        seen_zones.add(zone_id)

    for zone_id in policy.controlled_zones:
        if zone_id not in seen_zones:
            issues.append(
                _issue(
                    SafetyErrorCode.MISSING_ZONE,
                    "commands",
                    f"configured zone {zone_id!r} is missing",
                )
            )

    snapshot_zones = {zone.zone_id: zone for zone in snapshot.zones}
    for index, command in enumerate(request.commands):
        field_prefix = f"commands[{index}]"
        if isinstance(command, ReleaseZoneCommand):
            continue
        if not isinstance(command, SetZoneCommand):
            issues.append(
                _issue(
                    SafetyErrorCode.UNSUPPORTED_CONTROL,
                    f"{field_prefix}.mode",
                    "only set and release zone commands are supported",
                )
            )
            continue

        heating_c = command.heating_c
        cooling_c = command.cooling_c
        if not math.isfinite(heating_c):
            issues.append(
                _issue(
                    SafetyErrorCode.NONFINITE_VALUE,
                    f"{field_prefix}.heating_c",
                    "heating_c must be finite",
                )
            )
        if not math.isfinite(cooling_c):
            issues.append(
                _issue(
                    SafetyErrorCode.NONFINITE_VALUE,
                    f"{field_prefix}.cooling_c",
                    "cooling_c must be finite",
                )
            )
        if not math.isfinite(heating_c) or not math.isfinite(cooling_c):
            continue

        if not policy.heating_minimum_c <= heating_c <= policy.heating_maximum_c:
            issues.append(
                _issue(
                    SafetyErrorCode.OUT_OF_RANGE,
                    f"{field_prefix}.heating_c",
                    f"heating_c must be within "
                    f"[{policy.heating_minimum_c}, {policy.heating_maximum_c}]",
                )
            )
        if not policy.cooling_minimum_c <= cooling_c <= policy.cooling_maximum_c:
            issues.append(
                _issue(
                    SafetyErrorCode.OUT_OF_RANGE,
                    f"{field_prefix}.cooling_c",
                    f"cooling_c must be within "
                    f"[{policy.cooling_minimum_c}, {policy.cooling_maximum_c}]",
                )
            )
        if cooling_c - heating_c < policy.minimum_deadband_c:
            issues.append(
                _issue(
                    SafetyErrorCode.DEADBAND_VIOLATION,
                    field_prefix,
                    f"cooling_c - heating_c must be at least "
                    f"{policy.minimum_deadband_c}",
                )
            )

        zone_snapshot = snapshot_zones.get(command.zone_id)
        if (
            zone_snapshot is not None
            and zone_snapshot.occupant_count > 0.0
            and (
                heating_c < policy.occupied_heating_minimum_c
                or cooling_c > policy.occupied_cooling_maximum_c
            )
        ):
            issues.append(
                _issue(
                    SafetyErrorCode.OCCUPIED_COMFORT_VIOLATION,
                    field_prefix,
                    "occupied-zone setpoints must stay within configured "
                    "comfort limits",
                )
            )

    return tuple(issues)


__all__ = ["SafetyPolicy", "validate_control_action"]
