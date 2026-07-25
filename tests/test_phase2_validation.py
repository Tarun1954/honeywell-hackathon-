"""Unit tests for config-driven Phase 2 control-action safety."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.phase2_contracts import (
    ReleaseZoneCommand,
    SafetyErrorCode,
    SensorSnapshot,
    SensorSource,
    SetControlActionRequest,
    SetZoneCommand,
    ToolError,
    ZoneSensorData,
)
from src.phase2_validation import SafetyPolicy, validate_control_action


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config" / "phase2.yaml"
ZONE_IDS = (
    "SPACE1-1",
    "SPACE2-1",
    "SPACE3-1",
    "SPACE4-1",
    "SPACE5-1",
)


def load_policy() -> SafetyPolicy:
    """Load the checked-in YAML safety section through its Pydantic model."""

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return SafetyPolicy.model_validate(payload["safety"])


def make_snapshot(*, occupied: bool = True) -> SensorSnapshot:
    """Build a five-zone mock snapshot aligned with the Phase 1 topology."""

    zones = tuple(
        ZoneSensorData(
            zone_id=zone_id,
            air_temperature_c=22.0,
            relative_humidity_pct=45.0,
            co2_ppm=650.0,
            occupant_count=1.0 if occupied and index == 0 else 0.0,
            fanger_pmv=0.1,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
        )
        for index, zone_id in enumerate(ZONE_IDS)
    )
    return SensorSnapshot(
        cycle_id="cycle-1",
        snapshot_id="snapshot-1",
        sequence=1,
        timestamp=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
        source=SensorSource.MOCK,
        outdoor_drybulb_c=31.0,
        facility_electricity_demand_w=12_500.0,
        facility_electricity_kwh_since_start=10.0,
        zones=zones,
    )


def make_request(
    *,
    commands: tuple[SetZoneCommand | ReleaseZoneCommand, ...] | None = None,
    cycle_id: str = "cycle-1",
    snapshot_id: str = "snapshot-1",
    reasoning_log_id: str = "reasoning-1",
    hold_steps: int = 1,
) -> SetControlActionRequest:
    """Build a complete five-zone action."""

    if commands is None:
        commands = tuple(
            SetZoneCommand(
                zone_id=zone_id,
                heating_c=20.0,
                cooling_c=26.0,
            )
            for zone_id in ZONE_IDS
        )
    return SetControlActionRequest(
        request_id="request-1",
        cycle_id=cycle_id,
        snapshot_id=snapshot_id,
        reasoning_log_id=reasoning_log_id,
        idempotency_key="cycle-1-action-1",
        commands=commands,
        hold_steps=hold_steps,
    )


def error_codes(errors: tuple[ToolError, ...]) -> list[SafetyErrorCode]:
    """Extract ordered codes while keeping assertions compact."""

    return [error.code for error in errors]


class Phase2ConfigurationTests(unittest.TestCase):
    """Prevent configuration and safety-policy schema drift."""

    def test_checked_in_policy_is_valid_and_matches_phase1_zones(self) -> None:
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        policy = load_policy()

        self.assertEqual(payload["schema_version"], "phase2.v1")
        self.assertEqual(payload["simulation"]["source"], "mock")
        self.assertEqual(
            payload["simulation"]["available_controls"],
            ["zone_thermostat_setpoints"],
        )
        self.assertEqual(policy.controlled_zones, ZONE_IDS)
        self.assertEqual(policy.maximum_hold_steps, 4)

    def test_contradictory_policy_is_rejected(self) -> None:
        payload = load_policy().model_dump()
        payload["occupied_cooling_maximum_c"] = 19.0

        with self.assertRaises(ValidationError):
            SafetyPolicy.model_validate(payload)


class ControlActionSafetyTests(unittest.TestCase):
    """Exercise contextual checks without any tool or simulator wiring."""

    def setUp(self) -> None:
        self.policy = load_policy()
        self.snapshot = make_snapshot()
        self.reasoning_logs = {
            "reasoning-1": ("cycle-1", "snapshot-1"),
        }

    def validate(
        self,
        request: SetControlActionRequest,
        *,
        snapshot: SensorSnapshot | None = None,
        policy: SafetyPolicy | None = None,
        runtime_error_pending: bool = False,
    ) -> tuple[ToolError, ...]:
        """Run the validator with standard test context."""

        return validate_control_action(
            request,
            snapshot=snapshot or self.snapshot,
            policy=policy or self.policy,
            known_reasoning_logs=self.reasoning_logs,
            runtime_error_pending=runtime_error_pending,
        )

    def test_complete_five_zone_action_is_accepted(self) -> None:
        self.assertEqual(self.validate(make_request()), ())

    def test_release_is_an_allowed_safe_fallback(self) -> None:
        commands = tuple(
            ReleaseZoneCommand(zone_id=zone_id) for zone_id in ZONE_IDS
        )

        self.assertEqual(self.validate(make_request(commands=commands)), ())

    def test_exact_hard_boundaries_and_deadband_are_accepted(self) -> None:
        commands = tuple(
            SetZoneCommand(
                zone_id=zone_id,
                heating_c=16.0,
                cooling_c=20.0,
            )
            for zone_id in ZONE_IDS
        )

        self.assertEqual(
            self.validate(
                make_request(commands=commands),
                snapshot=make_snapshot(occupied=False),
            ),
            (),
        )

    def test_stale_cycle_and_snapshot_are_rejected(self) -> None:
        self.reasoning_logs["reasoning-1"] = ("cycle-old", "snapshot-old")
        errors = self.validate(
            make_request(cycle_id="cycle-old", snapshot_id="snapshot-old")
        )

        self.assertEqual(
            error_codes(errors),
            [
                SafetyErrorCode.STALE_SNAPSHOT,
                SafetyErrorCode.STALE_SNAPSHOT,
            ],
        )

    def test_missing_reasoning_log_is_rejected(self) -> None:
        errors = self.validate(make_request(reasoning_log_id="reasoning-missing"))

        self.assertIn(SafetyErrorCode.MISSING_REASONING_LOG, error_codes(errors))

    def test_mismatched_reasoning_log_correlation_is_rejected(self) -> None:
        self.reasoning_logs["reasoning-1"] = ("cycle-other", "snapshot-other")

        errors = self.validate(make_request())

        self.assertIn(SafetyErrorCode.MISSING_REASONING_LOG, error_codes(errors))

    def test_incomplete_sensor_topology_fails_closed(self) -> None:
        incomplete_payload = self.snapshot.model_dump()
        incomplete_payload["zones"] = self.snapshot.zones[1:]
        incomplete_snapshot = SensorSnapshot.model_validate(incomplete_payload)

        errors = self.validate(
            make_request(),
            snapshot=incomplete_snapshot,
        )

        self.assertIn(SafetyErrorCode.MISSING_ZONE, error_codes(errors))

    def test_zone_coverage_errors_are_accumulated_deterministically(self) -> None:
        commands = (
            SetZoneCommand(
                zone_id="SPACE1-1",
                heating_c=20.0,
                cooling_c=26.0,
            ),
            SetZoneCommand(
                zone_id="SPACE1-1",
                heating_c=20.0,
                cooling_c=26.0,
            ),
            SetZoneCommand(
                zone_id="SPACE2-1",
                heating_c=20.0,
                cooling_c=26.0,
            ),
            SetZoneCommand(
                zone_id="SPACE3-1",
                heating_c=20.0,
                cooling_c=26.0,
            ),
            SetZoneCommand(
                zone_id="NOT-A-ZONE",
                heating_c=20.0,
                cooling_c=26.0,
            ),
        )

        errors = self.validate(make_request(commands=commands))

        self.assertEqual(
            error_codes(errors)[:4],
            [
                SafetyErrorCode.DUPLICATE_ZONE,
                SafetyErrorCode.UNKNOWN_ZONE,
                SafetyErrorCode.MISSING_ZONE,
                SafetyErrorCode.MISSING_ZONE,
            ],
        )

    def test_out_of_range_values_are_rejected_not_clamped(self) -> None:
        commands = tuple(
            SetZoneCommand(
                zone_id=zone_id,
                heating_c=15.0 if index == 0 else 20.0,
                cooling_c=31.0 if index == 1 else 26.0,
            )
            for index, zone_id in enumerate(ZONE_IDS)
        )
        request = make_request(commands=commands)
        original = request.model_dump()

        errors = self.validate(request)

        self.assertGreaterEqual(
            error_codes(errors).count(SafetyErrorCode.OUT_OF_RANGE),
            2,
        )
        self.assertEqual(request.model_dump(), original)

    def test_deadband_violation_is_rejected(self) -> None:
        commands = list(make_request().commands)
        commands[0] = SetZoneCommand(
            zone_id="SPACE1-1",
            heating_c=22.0,
            cooling_c=22.5,
        )

        errors = self.validate(make_request(commands=tuple(commands)))

        self.assertIn(SafetyErrorCode.DEADBAND_VIOLATION, error_codes(errors))

    def test_occupied_comfort_violation_is_rejected(self) -> None:
        commands = list(make_request().commands)
        commands[0] = SetZoneCommand(
            zone_id="SPACE1-1",
            heating_c=19.0,
            cooling_c=27.0,
        )

        errors = self.validate(make_request(commands=tuple(commands)))

        self.assertIn(
            SafetyErrorCode.OCCUPIED_COMFORT_VIOLATION,
            error_codes(errors),
        )

    def test_policy_specific_hold_limit_is_enforced(self) -> None:
        payload = self.policy.model_dump()
        payload["maximum_hold_steps"] = 2
        stricter_policy = SafetyPolicy.model_validate(payload)

        errors = self.validate(
            make_request(hold_steps=3),
            policy=stricter_policy,
        )

        self.assertIn(SafetyErrorCode.OUT_OF_RANGE, error_codes(errors))

    def test_pending_runtime_failure_blocks_action(self) -> None:
        errors = self.validate(
            make_request(),
            runtime_error_pending=True,
        )

        self.assertEqual(
            error_codes(errors),
            [SafetyErrorCode.RUNTIME_ERROR_PENDING],
        )

    def test_nonfinite_setpoints_never_reach_contextual_validation(self) -> None:
        with self.assertRaises(ValidationError):
            SetZoneCommand(
                zone_id="SPACE1-1",
                heating_c=float("nan"),
                cooling_c=26.0,
            )


if __name__ == "__main__":
    unittest.main()
