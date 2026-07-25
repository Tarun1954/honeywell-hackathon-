"""Unit tests for the transport-independent Phase 2 Pydantic contracts."""

from __future__ import annotations

import math
import unittest
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from src.phase2_contracts import (
    SCHEMA_VERSION,
    AvailableControl,
    CarbonIntensityCategory,
    CarbonIntensitySample,
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
    ReleaseZoneCommand,
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


NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


def make_zone(
    zone_id: str = "SPACE1-1",
    *,
    occupant_count: float = 1.0,
) -> ZoneSensorData:
    """Build one valid zone observation."""

    return ZoneSensorData(
        zone_id=zone_id,
        air_temperature_c=22.5,
        relative_humidity_pct=45.0,
        co2_ppm=650.0,
        occupant_count=occupant_count,
        fanger_pmv=0.1,
        heating_setpoint_c=20.0,
        cooling_setpoint_c=26.0,
    )


def make_snapshot(
    *,
    cycle_id: str = "cycle-1",
    snapshot_id: str = "snapshot-2",
    sequence: int = 2,
    timestamp: datetime = NOW,
) -> SensorSnapshot:
    """Build one valid canonical sensor snapshot."""

    return SensorSnapshot(
        cycle_id=cycle_id,
        snapshot_id=snapshot_id,
        sequence=sequence,
        timestamp=timestamp,
        source=SensorSource.MOCK,
        outdoor_drybulb_c=31.0,
        facility_electricity_demand_w=12_500.0,
        facility_electricity_kwh_since_start=42.5,
        zones=(make_zone(),),
    )


class ContractSchemaTests(unittest.TestCase):
    """Protect the common versioned tool envelope."""

    def test_all_tool_contracts_publish_version_and_request_id(self) -> None:
        contract_types = (
            ReadSensorDataRequest,
            ReadSensorDataResponse,
            GridCarbonIntensityRequest,
            GridCarbonIntensityResponse,
            LogReasoningRequest,
            LogReasoningResponse,
            SetControlActionRequest,
            SetControlActionResponse,
            ParseRuntimeErrorsRequest,
            ParseRuntimeErrorsResponse,
        )

        for contract_type in contract_types:
            with self.subTest(contract_type=contract_type.__name__):
                properties = contract_type.model_json_schema()["properties"]
                self.assertIn("schema_version", properties)
                self.assertIn("request_id", properties)
                self.assertEqual(
                    properties["schema_version"]["default"],
                    SCHEMA_VERSION,
                )

    def test_snapshot_round_trips_through_json(self) -> None:
        snapshot = make_snapshot()

        restored = SensorSnapshot.model_validate_json(snapshot.model_dump_json())
        decoded_restored = SensorSnapshot.model_validate(
            snapshot.model_dump(mode="json")
        )

        self.assertEqual(restored, snapshot)
        self.assertEqual(decoded_restored, snapshot)
        self.assertEqual(
            restored.available_controls,
            (AvailableControl.ZONE_THERMOSTAT_SETPOINTS,),
        )

    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ReadSensorDataRequest(
                request_id="request-1",
                history_steps=1,
                unexpected=True,  # type: ignore[call-arg]
            )

    def test_blank_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ReadSensorDataRequest(request_id="   ")

    def test_numeric_strings_are_not_coerced(self) -> None:
        with self.assertRaises(ValidationError):
            ReadSensorDataRequest(
                request_id="request-1",
                history_steps="1",  # type: ignore[arg-type]
            )


class SensorContractTests(unittest.TestCase):
    """Exercise sensor shape, finiteness, and history ordering."""

    def test_nonfinite_zone_values_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ZoneSensorData(
                    zone_id="SPACE1-1",
                    air_temperature_c=value,
                    relative_humidity_pct=45.0,
                    co2_ppm=650.0,
                    occupant_count=1.0,
                    fanger_pmv=0.0,
                    heating_setpoint_c=20.0,
                    cooling_setpoint_c=26.0,
                )

    def test_sensor_ranges_are_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            make_zone(occupant_count=-1.0)
        with self.assertRaises(ValidationError):
            ZoneSensorData(
                zone_id="SPACE1-1",
                air_temperature_c=22.0,
                relative_humidity_pct=101.0,
                co2_ppm=650.0,
                occupant_count=1.0,
                fanger_pmv=0.0,
                heating_setpoint_c=20.0,
                cooling_setpoint_c=26.0,
            )

    def test_duplicate_snapshot_zone_is_rejected(self) -> None:
        payload = make_snapshot().model_dump()
        payload["zones"] = (make_zone(), make_zone())

        with self.assertRaisesRegex(ValidationError, "unique zone_id"):
            SensorSnapshot.model_validate(payload)

    def test_history_must_precede_current_snapshot(self) -> None:
        current = make_snapshot(sequence=2)
        later = make_snapshot(
            snapshot_id="snapshot-3",
            sequence=3,
            timestamp=NOW + timedelta(minutes=15),
        )

        with self.assertRaisesRegex(ValidationError, "ascending sequence"):
            ReadSensorDataResponse(
                request_id="request-1",
                snapshot=current,
                history=(later,),
            )

    def test_history_timestamp_must_precede_current_snapshot(self) -> None:
        current = make_snapshot(sequence=2)
        future_timestamp = make_snapshot(
            snapshot_id="snapshot-1",
            sequence=1,
            timestamp=NOW + timedelta(minutes=15),
        )

        with self.assertRaisesRegex(ValidationError, "timestamp order"):
            ReadSensorDataResponse(
                request_id="request-1",
                snapshot=current,
                history=(future_timestamp,),
            )

    def test_history_can_span_control_cycles(self) -> None:
        current = make_snapshot(cycle_id="cycle-2", sequence=2)
        historical = make_snapshot(
            cycle_id="cycle-1",
            snapshot_id="snapshot-1",
            sequence=1,
            timestamp=NOW - timedelta(minutes=15),
        )

        response = ReadSensorDataResponse(
            request_id="request-1",
            snapshot=current,
            history=(historical,),
        )

        self.assertEqual(response.history, (historical,))

    def test_naive_timestamp_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            make_snapshot(timestamp=datetime(2026, 7, 25, 12, 0))


class ToolSpecificContractTests(unittest.TestCase):
    """Exercise the five planned tool schemas without implementing tools."""

    def test_request_bounds_are_enforced(self) -> None:
        invalid_factories = (
            lambda: ReadSensorDataRequest(
                request_id="request-read",
                history_steps=5,
            ),
            lambda: GridCarbonIntensityRequest(
                request_id="request-carbon",
                snapshot_id="snapshot-2",
                forecast_steps=17,
            ),
            lambda: ParseRuntimeErrorsRequest(
                request_id="request-errors",
                cycle_id="cycle-1",
                limit=21,
            ),
            lambda: LogReasoningRequest(
                request_id="request-reasoning",
                cycle_id="cycle-1",
                snapshot_id="snapshot-2",
                decision_summary="Maintain comfort.",
                objective_tags=(ObjectiveTag.THERMAL_COMFORT,),
                tradeoff_summary="Small energy premium.",
                confidence=1.1,
            ),
        )

        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory()

    def test_release_command_cannot_carry_setpoints(self) -> None:
        with self.assertRaises(ValidationError):
            ReleaseZoneCommand.model_validate(
                {
                    "mode": "release",
                    "zone_id": "SPACE1-1",
                    "heating_c": 20.0,
                }
            )

    def test_discriminated_zone_command_rejects_unknown_mode(self) -> None:
        payload = {
            "request_id": "request-action",
            "cycle_id": "cycle-1",
            "snapshot_id": "snapshot-2",
            "reasoning_log_id": "reasoning-1",
            "idempotency_key": "action-cycle-1",
            "commands": [{"mode": "boost", "zone_id": "SPACE1-1"}],
            "hold_steps": 1,
        }

        with self.assertRaises(ValidationError):
            SetControlActionRequest.model_validate(payload)

    def test_valid_json_decoded_lists_are_accepted(self) -> None:
        payload = {
            "request_id": "request-action",
            "cycle_id": "cycle-1",
            "snapshot_id": "snapshot-2",
            "reasoning_log_id": "reasoning-1",
            "idempotency_key": "action-cycle-1",
            "commands": [
                {
                    "mode": "set",
                    "zone_id": "SPACE1-1",
                    "heating_c": 20.0,
                    "cooling_c": 26.0,
                }
            ],
            "hold_steps": 1,
        }

        request = SetControlActionRequest.model_validate(payload)

        self.assertIsInstance(request.commands, tuple)
        self.assertIsInstance(request.commands[0], SetZoneCommand)

    def test_nonfinite_setpoint_is_rejected_structurally(self) -> None:
        with self.assertRaises(ValidationError):
            SetZoneCommand(
                zone_id="SPACE1-1",
                heating_c=math.nan,
                cooling_c=26.0,
            )

    def test_reasoning_objective_tags_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            LogReasoningRequest(
                request_id="request-reasoning",
                cycle_id="cycle-1",
                snapshot_id="snapshot-2",
                decision_summary="Hold current setpoints.",
                objective_tags=(
                    ObjectiveTag.ENERGY_REDUCTION,
                    ObjectiveTag.ENERGY_REDUCTION,
                ),
                tradeoff_summary="No material comfort impact.",
                confidence=0.8,
            )

    def test_reasoning_accepts_standard_decoded_json_arrays(self) -> None:
        request = LogReasoningRequest.model_validate(
            {
                "request_id": "request-reasoning",
                "cycle_id": "cycle-1",
                "snapshot_id": "snapshot-2",
                "decision_summary": "Maintain the comfort envelope.",
                "objective_tags": ["thermal_comfort", "energy_reduction"],
                "tradeoff_summary": "Hold demand steady for this step.",
                "confidence": 0.8,
            }
        )

        self.assertEqual(
            request.objective_tags,
            (
                ObjectiveTag.THERMAL_COMFORT,
                ObjectiveTag.ENERGY_REDUCTION,
            ),
        )

    def test_carbon_forecast_must_be_ordered(self) -> None:
        current = CarbonIntensitySample(
            offset_steps=0,
            timestamp=NOW,
            g_co2_per_kwh=350.0,
            category=CarbonIntensityCategory.MODERATE,
        )
        forecast = (
            CarbonIntensitySample(
                offset_steps=2,
                timestamp=NOW + timedelta(minutes=30),
                g_co2_per_kwh=400.0,
                category=CarbonIntensityCategory.HIGH,
            ),
            CarbonIntensitySample(
                offset_steps=1,
                timestamp=NOW + timedelta(minutes=15),
                g_co2_per_kwh=375.0,
                category=CarbonIntensityCategory.MODERATE,
            ),
        )

        with self.assertRaisesRegex(ValidationError, "strictly increasing"):
            GridCarbonIntensityResponse(
                request_id="request-carbon",
                snapshot_id="snapshot-2",
                current=current,
                forecast=forecast,
            )

    def test_control_response_status_is_consistent(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one error"):
            SetControlActionResponse(
                request_id="request-action",
                cycle_id="cycle-1",
                snapshot_id="snapshot-2",
                status=ControlActionStatus.REJECTED,
            )

        error = ToolError(
            code=SafetyErrorCode.OUT_OF_RANGE,
            field="commands[0].heating_c",
            message="outside configured range",
            retryable=True,
        )
        rejected = SetControlActionResponse(
            request_id="request-action",
            cycle_id="cycle-1",
            snapshot_id="snapshot-2",
            status=ControlActionStatus.REJECTED,
            errors=(error,),
        )
        self.assertEqual(rejected.errors, (error,))

        with self.assertRaisesRegex(ValidationError, "cannot include action_id"):
            SetControlActionResponse(
                request_id="request-action",
                cycle_id="cycle-1",
                snapshot_id="snapshot-2",
                status=ControlActionStatus.REJECTED,
                action_id="action-1",
                errors=(error,),
            )

    def test_runtime_error_response_requires_unique_records(self) -> None:
        record = RuntimeErrorRecord(
            error_id="error-1",
            source=RuntimeErrorSource.ENERGYPLUS_ERROR_FILE,
            severity=RuntimeErrorSeverity.SEVERE,
            code="ACTUATOR_HANDLE",
            summary="The actuator handle could not be resolved.",
            retryable=True,
            correction_fields=("commands",),
            correction_hint="Release the unsupported command.",
        )

        with self.assertRaisesRegex(ValidationError, "unique error_id"):
            ParseRuntimeErrorsResponse(
                request_id="request-errors",
                cycle_id="cycle-1",
                errors=(record, record),
            )


if __name__ == "__main__":
    unittest.main()
