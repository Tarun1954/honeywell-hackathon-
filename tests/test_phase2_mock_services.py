"""Unit tests for the deterministic Phase 2 in-memory services."""

from __future__ import annotations

import unittest
from datetime import timedelta

from pydantic import ValidationError

from src.phase2_contracts import (
    CarbonIntensityCategory,
    ControlActionStatus,
    GridCarbonIntensityRequest,
    ParseRuntimeErrorsRequest,
    ReadSensorDataRequest,
    RuntimeErrorSeverity,
    SafetyErrorCode,
    SensorSource,
    SetControlActionRequest,
)
from src.phase2_mock_services import (
    PHASE1_ZONE_IDS,
    TIMESTEP,
    DuplicateLedgerEntryError,
    MockRuntimeErrorStore,
    MockSensorStore,
    Phase2Fixture,
    Phase2Services,
    UnknownRuntimeErrorCursorError,
    build_control_action_fixture,
    build_reasoning_fixture,
)


def build_logged_action(
    services: Phase2Services,
    fixture: Phase2Fixture,
) -> SetControlActionRequest:
    """Log a deterministic rationale and return its correlated action."""

    snapshot = services.sensor_store.snapshot_for(fixture)
    reasoning = services.log_reasoning(
        build_reasoning_fixture(fixture, snapshot)
    )
    return build_control_action_fixture(
        fixture,
        snapshot,
        reasoning.reasoning_log_id,
    )


def replace_action_fields(
    request: SetControlActionRequest,
    **updates: object,
) -> SetControlActionRequest:
    """Re-parse an action after updating its decoded JSON representation."""

    payload = request.model_dump(mode="json")
    payload.update(updates)
    return SetControlActionRequest.model_validate(payload)


class DeterministicSensorFixtureTests(unittest.TestCase):
    """Protect the fixed nine-scenario 15-minute timeline."""

    def test_fixture_registry_contains_every_requested_scenario(self) -> None:
        self.assertEqual(
            {fixture.value for fixture in Phase2Fixture},
            {
                "unoccupied_mild_conditions",
                "occupied_too_cold",
                "occupied_too_warm",
                "comfortable_occupied_conditions",
                "high_demand",
                "high_mock_carbon_intensity",
                "stale_snapshot",
                "actuator_runtime_error",
                "invalid_setpoint_error",
            },
        )

    def test_independent_stores_have_identical_fixed_snapshots(self) -> None:
        first = MockSensorStore()
        second = MockSensorStore()

        self.assertEqual(
            tuple(item.model_dump(mode="json") for item in first.timeline),
            tuple(item.model_dump(mode="json") for item in second.timeline),
        )

    def test_all_snapshots_have_five_zones_and_exact_timestep_spacing(self) -> None:
        snapshots = MockSensorStore().timeline

        self.assertEqual(len(snapshots), len(Phase2Fixture))
        self.assertEqual(
            tuple(snapshot.sequence for snapshot in snapshots),
            tuple(range(1, len(snapshots) + 1)),
        )
        self.assertEqual(
            len({snapshot.snapshot_id for snapshot in snapshots}),
            len(snapshots),
        )
        for snapshot in snapshots:
            with self.subTest(snapshot=snapshot.snapshot_id):
                self.assertEqual(snapshot.source, SensorSource.MOCK)
                self.assertEqual(snapshot.timestep_minutes, 15)
                self.assertEqual(
                    tuple(zone.zone_id for zone in snapshot.zones),
                    PHASE1_ZONE_IDS,
                )

        for earlier, later in zip(snapshots, snapshots[1:], strict=False):
            self.assertEqual(later.timestamp - earlier.timestamp, TIMESTEP)
            self.assertGreater(
                later.facility_electricity_kwh_since_start,
                earlier.facility_electricity_kwh_since_start,
            )

    def test_condition_values_match_named_fixtures(self) -> None:
        store = MockSensorStore()
        unoccupied = store.snapshot_for(Phase2Fixture.UNOCCUPIED_MILD)
        too_cold = store.snapshot_for(Phase2Fixture.OCCUPIED_TOO_COLD)
        too_warm = store.snapshot_for(Phase2Fixture.OCCUPIED_TOO_WARM)
        comfortable = store.snapshot_for(Phase2Fixture.COMFORTABLE_OCCUPIED)
        high_demand = store.snapshot_for(Phase2Fixture.HIGH_DEMAND)

        self.assertTrue(
            all(zone.occupant_count == 0.0 for zone in unoccupied.zones)
        )
        self.assertTrue(
            all(
                zone.occupant_count > 0.0 and zone.fanger_pmv < -0.7
                for zone in too_cold.zones
            )
        )
        self.assertTrue(
            all(
                zone.occupant_count > 0.0 and zone.fanger_pmv > 0.7
                for zone in too_warm.zones
            )
        )
        self.assertTrue(
            all(
                zone.occupant_count > 0.0 and abs(zone.fanger_pmv) <= 0.7
                for zone in comfortable.zones
            )
        )
        self.assertGreater(
            high_demand.facility_electricity_demand_w,
            comfortable.facility_electricity_demand_w,
        )

    def test_snapshot_ids_are_stable_and_models_are_frozen(self) -> None:
        store = MockSensorStore()
        snapshot = store.current
        original_id = snapshot.snapshot_id

        self.assertIs(store.get(original_id), snapshot)
        with self.assertRaises(ValidationError):
            snapshot.snapshot_id = "changed"  # type: ignore[misc]
        self.assertEqual(store.get(original_id).snapshot_id, original_id)

    def test_staleness_is_relative_to_current_snapshot(self) -> None:
        store = MockSensorStore()
        stale = store.snapshot_for(Phase2Fixture.STALE_SNAPSHOT)
        current = store.current

        self.assertTrue(store.is_stale(stale.snapshot_id))
        self.assertFalse(store.is_stale(current.snapshot_id))
        self.assertTrue(store.is_stale("unknown-snapshot"))

        advanced = store.advance()
        self.assertTrue(store.is_stale(current.snapshot_id))
        self.assertFalse(store.is_stale(advanced.snapshot_id))
        self.assertEqual(advanced.timestamp - current.timestamp, TIMESTEP)

    def test_sensor_read_returns_bounded_ascending_history(self) -> None:
        store = MockSensorStore()
        response = store.read(
            ReadSensorDataRequest(
                request_id="read-sensors-1",
                history_steps=4,
            )
        )

        self.assertEqual(response.snapshot, store.current)
        self.assertEqual(len(response.history), 4)
        self.assertEqual(
            tuple(item.sequence for item in response.history),
            (1, 2, 3, 4),
        )


class DeterministicGridCarbonTests(unittest.TestCase):
    """Exercise stable current and forecast carbon fixtures."""

    def test_repeated_reads_are_identical_and_honor_forecast_length(self) -> None:
        services = Phase2Services.deterministic()
        snapshot = services.sensor_store.current
        request = GridCarbonIntensityRequest(
            request_id="grid-read-1",
            snapshot_id=snapshot.snapshot_id,
            forecast_steps=16,
        )

        first = services.grid_carbon_store.read(request)
        second = services.grid_carbon_store.read(request)

        self.assertEqual(first, second)
        self.assertEqual(first.current.offset_steps, 0)
        self.assertEqual(
            tuple(sample.offset_steps for sample in first.forecast),
            tuple(range(1, 17)),
        )
        self.assertTrue(
            all(
                sample.timestamp
                == snapshot.timestamp + sample.offset_steps * TIMESTEP
                for sample in first.forecast
            )
        )

    def test_high_carbon_fixture_is_deterministically_very_high(self) -> None:
        services = Phase2Services.deterministic(
            initial_fixture=Phase2Fixture.HIGH_MOCK_CARBON_INTENSITY
        )
        snapshot = services.sensor_store.current
        response = services.grid_carbon_store.read(
            GridCarbonIntensityRequest(
                request_id="grid-high-1",
                snapshot_id=snapshot.snapshot_id,
                forecast_steps=4,
            )
        )

        self.assertEqual(
            response.current.category,
            CarbonIntensityCategory.VERY_HIGH,
        )
        self.assertGreaterEqual(response.current.g_co2_per_kwh, 700.0)
        self.assertTrue(
            all(
                sample.category is CarbonIntensityCategory.VERY_HIGH
                for sample in response.forecast
            )
        )

    def test_unknown_snapshot_is_rejected(self) -> None:
        services = Phase2Services.deterministic()

        with self.assertRaisesRegex(KeyError, "unknown snapshot_id"):
            services.grid_carbon_store.read(
                GridCarbonIntensityRequest(
                    request_id="grid-unknown-1",
                    snapshot_id="snapshot-unknown",
                    forecast_steps=1,
                )
            )


class AppendOnlyLedgerTests(unittest.TestCase):
    """Prove reasoning, action, and runtime stores never overwrite records."""

    def test_reasoning_ledger_is_append_only_and_correlations_are_read_only(
        self,
    ) -> None:
        services = Phase2Services.deterministic()
        current = services.sensor_store.current
        first_request = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            current,
        )
        first_response = services.reasoning_ledger.append(first_request)
        first_prefix = services.reasoning_ledger.records

        second_request = first_request.model_copy(
            update={"request_id": "fixture-reasoning-request-second"}
        )
        second_response = services.reasoning_ledger.append(second_request)

        self.assertEqual(
            services.reasoning_ledger.records[:1],
            first_prefix,
        )
        self.assertNotEqual(
            first_response.reasoning_log_id,
            second_response.reasoning_log_id,
        )
        correlations = services.reasoning_ledger.correlations
        with self.assertRaises(TypeError):
            correlations[first_response.reasoning_log_id] = (  # type: ignore[index]
                "changed",
                "changed",
            )

    def test_reasoning_retry_is_idempotent_and_duplicate_log_id_is_refused(
        self,
    ) -> None:
        services = Phase2Services.deterministic()
        request = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            services.sensor_store.current,
        )
        first = services.reasoning_ledger.append(
            request,
            reasoning_log_id=" reasoning-fixed ",
        )
        retry = services.reasoning_ledger.append(
            request,
            reasoning_log_id="reasoning-fixed",
        )
        conflicting_request = request.model_copy(
            update={"request_id": "reasoning-conflict"}
        )

        self.assertEqual(retry, first)
        self.assertEqual(first.reasoning_log_id, "reasoning-fixed")
        self.assertIn(
            "reasoning-fixed",
            services.reasoning_ledger.correlations,
        )
        self.assertEqual(len(services.reasoning_ledger.records), 1)
        with self.assertRaises(DuplicateLedgerEntryError):
            services.reasoning_ledger.append(
                conflicting_request,
                reasoning_log_id="reasoning-fixed",
            )
        with self.assertRaises(ValidationError):
            services.reasoning_ledger.append(
                conflicting_request,
                reasoning_log_id="",
            )
        self.assertEqual(len(services.reasoning_ledger.records), 1)

    def test_action_ledger_retains_accepted_prefix(self) -> None:
        services = Phase2Services.deterministic()
        first_request = build_logged_action(
            services,
            Phase2Fixture.COMFORTABLE_OCCUPIED,
        )
        first_response = services.submit_action(first_request)
        first_prefix = services.action_ledger.records
        second_request = replace_action_fields(
            first_request,
            request_id="fixture-action-request-second",
            idempotency_key="fixture-action-key-second",
        )
        second_response = services.submit_action(second_request)

        self.assertEqual(first_response.status, ControlActionStatus.ACCEPTED)
        self.assertEqual(second_response.status, ControlActionStatus.ACCEPTED)
        self.assertEqual(services.action_ledger.records[:1], first_prefix)

    def test_runtime_error_store_is_append_only(self) -> None:
        store = MockRuntimeErrorStore()
        first = store.inject_fixture(
            "cycle-1",
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
        )
        first_prefix = store.records
        store.inject_fixture(
            "cycle-1",
            Phase2Fixture.INVALID_SETPOINT_ERROR,
        )

        self.assertEqual(store.records[:1], first_prefix)
        with self.assertRaises(DuplicateLedgerEntryError):
            store.inject("cycle-1", first)
        self.assertEqual(len(store.records), 2)


class ActionSafetyAndDuplicateTests(unittest.TestCase):
    """Exercise validator reuse, staleness, and duplicate prevention."""

    def test_identical_idempotency_replay_returns_duplicate_without_append(
        self,
    ) -> None:
        services = Phase2Services.deterministic()
        request = build_logged_action(
            services,
            Phase2Fixture.COMFORTABLE_OCCUPIED,
        )
        accepted = services.submit_action(request)
        retry_request = replace_action_fields(
            request,
            request_id="fixture-action-request-retry",
        )
        duplicate = services.submit_action(retry_request)

        self.assertEqual(accepted.status, ControlActionStatus.ACCEPTED)
        self.assertEqual(duplicate.status, ControlActionStatus.DUPLICATE)
        self.assertEqual(duplicate.action_id, accepted.action_id)
        self.assertEqual(len(services.action_ledger.records), 1)

    def test_conflicting_idempotency_key_is_rejected_without_overwrite(
        self,
    ) -> None:
        services = Phase2Services.deterministic()
        request = build_logged_action(
            services,
            Phase2Fixture.COMFORTABLE_OCCUPIED,
        )
        services.submit_action(request)
        payload = request.model_dump(mode="json")
        payload["request_id"] = "fixture-action-request-conflict"
        payload["commands"][0]["heating_c"] = 21.0
        conflict = SetControlActionRequest.model_validate(payload)

        response = services.submit_action(conflict)

        self.assertEqual(response.status, ControlActionStatus.REJECTED)
        self.assertEqual(response.errors[0].field, "idempotency_key")
        self.assertEqual(len(services.action_ledger.records), 1)

    def test_duplicate_action_id_is_rejected_without_append(self) -> None:
        services = Phase2Services.deterministic()
        first_request = build_logged_action(
            services,
            Phase2Fixture.COMFORTABLE_OCCUPIED,
        )
        first = services.submit_action(
            first_request,
            action_id=" action-fixed ",
        )
        second_request = replace_action_fields(
            first_request,
            request_id="fixture-action-request-new-id",
            idempotency_key="fixture-action-key-new-id",
        )

        duplicate_id = services.submit_action(
            second_request,
            action_id="action-fixed",
        )

        self.assertEqual(first.status, ControlActionStatus.ACCEPTED)
        self.assertEqual(first.action_id, "action-fixed")
        self.assertEqual(duplicate_id.status, ControlActionStatus.REJECTED)
        self.assertEqual(duplicate_id.errors[0].field, "action_id")
        self.assertEqual(len(services.action_ledger.records), 1)

    def test_stale_snapshot_is_rejected_without_append(self) -> None:
        services = Phase2Services.deterministic()
        request = build_logged_action(
            services,
            Phase2Fixture.STALE_SNAPSHOT,
        )

        response = services.submit_action(request)

        self.assertEqual(response.status, ControlActionStatus.REJECTED)
        self.assertIn(
            SafetyErrorCode.STALE_SNAPSHOT,
            {error.code for error in response.errors},
        )
        self.assertEqual(services.action_ledger.records, ())

    def test_blank_explicit_action_id_is_rejected_before_append(self) -> None:
        services = Phase2Services.deterministic()
        request = build_logged_action(
            services,
            Phase2Fixture.COMFORTABLE_OCCUPIED,
        )

        with self.assertRaises(ValidationError):
            services.submit_action(request, action_id="")
        self.assertEqual(services.action_ledger.records, ())

    def test_invalid_setpoint_fixture_reuses_safety_validator(self) -> None:
        services = Phase2Services.deterministic(
            initial_fixture=Phase2Fixture.INVALID_SETPOINT_ERROR
        )
        request = build_logged_action(
            services,
            Phase2Fixture.INVALID_SETPOINT_ERROR,
        )

        response = services.submit_action(request)

        self.assertEqual(response.status, ControlActionStatus.REJECTED)
        self.assertIn(
            SafetyErrorCode.OUT_OF_RANGE,
            {error.code for error in response.errors},
        )
        self.assertEqual(services.action_ledger.records, ())

    def test_blocking_runtime_fixture_rejects_action(self) -> None:
        services = Phase2Services.deterministic(
            initial_fixture=Phase2Fixture.ACTUATOR_RUNTIME_ERROR
        )
        snapshot = services.sensor_store.current
        services.runtime_error_store.inject_fixture(
            snapshot.cycle_id,
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
        )
        request = build_logged_action(
            services,
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
        )

        response = services.submit_action(request)

        self.assertEqual(response.status, ControlActionStatus.REJECTED)
        self.assertIn(
            SafetyErrorCode.RUNTIME_ERROR_PENDING,
            {error.code for error in response.errors},
        )
        self.assertEqual(services.action_ledger.records, ())


class RuntimeErrorRetrievalTests(unittest.TestCase):
    """Exercise deterministic injection, cycle isolation, and pagination."""

    def test_fixture_errors_can_be_injected_and_retrieved_in_pages(self) -> None:
        store = MockRuntimeErrorStore()
        first = store.inject_fixture(
            " cycle-1 ",
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
            action_id="action-1",
        )
        second = store.inject_fixture(
            "cycle-1",
            Phase2Fixture.INVALID_SETPOINT_ERROR,
            action_id="action-1",
        )
        store.inject_fixture(
            "cycle-2",
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
        )

        first_page = store.retrieve(
            ParseRuntimeErrorsRequest(
                request_id="runtime-page-1",
                cycle_id="cycle-1",
                limit=1,
            )
        )
        second_page = store.retrieve(
            ParseRuntimeErrorsRequest(
                request_id="runtime-page-2",
                cycle_id="cycle-1",
                after_error_id=first_page.next_error_id,
                limit=1,
            )
        )

        self.assertEqual(first_page.errors, (first,))
        self.assertTrue(first_page.has_more)
        self.assertEqual(first_page.next_error_id, first.error_id)
        self.assertEqual(second_page.errors, (second,))
        self.assertFalse(second_page.has_more)
        self.assertIsNone(second_page.next_error_id)
        self.assertEqual(
            first.severity,
            RuntimeErrorSeverity.SEVERE,
        )
        self.assertEqual(first.code, "ACTUATOR_WRITEBACK_FAILED")
        self.assertEqual(second.code, "INVALID_SETPOINT")

    def test_invalid_runtime_override_does_not_consume_generated_id(self) -> None:
        store = MockRuntimeErrorStore()

        with self.assertRaises(ValidationError):
            store.inject_fixture(
                "cycle-1",
                Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
                action_id="invalid action id",
            )
        first = store.inject_fixture(
            "cycle-1",
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
        )

        self.assertEqual(first.error_id, "runtime-error-000001")

    def test_blank_explicit_runtime_error_id_is_rejected(self) -> None:
        store = MockRuntimeErrorStore()

        with self.assertRaises(ValidationError):
            store.inject_fixture(
                "cycle-1",
                Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
                error_id="",
            )
        self.assertEqual(store.records, ())

    def test_unknown_or_cross_cycle_cursor_fails_closed(self) -> None:
        store = MockRuntimeErrorStore()
        first = store.inject_fixture(
            "cycle-1",
            Phase2Fixture.ACTUATOR_RUNTIME_ERROR,
        )

        with self.assertRaises(UnknownRuntimeErrorCursorError):
            store.retrieve(
                ParseRuntimeErrorsRequest(
                    request_id="runtime-bad-cursor",
                    cycle_id="cycle-2",
                    after_error_id=first.error_id,
                    limit=1,
                )
            )

    def test_blocking_state_is_cycle_scoped(self) -> None:
        store = MockRuntimeErrorStore()
        store.inject_fixture(
            "cycle-1",
            Phase2Fixture.INVALID_SETPOINT_ERROR,
        )

        self.assertTrue(store.has_blocking_error("cycle-1"))
        self.assertFalse(store.has_blocking_error("cycle-2"))


class Phase2ServicesBundleTests(unittest.TestCase):
    """Ensure dependency bundles are coherent but state-independent."""

    def test_bundles_start_equal_but_do_not_share_mutable_state(self) -> None:
        first = Phase2Services.deterministic()
        second = Phase2Services.deterministic()

        self.assertEqual(first.sensor_store.timeline, second.sensor_store.timeline)
        first.sensor_store.advance()
        self.assertNotEqual(first.sensor_store.current, second.sensor_store.current)

        request = build_reasoning_fixture(
            Phase2Fixture.COMFORTABLE_OCCUPIED,
            second.sensor_store.current,
        )
        first_snapshot = first.sensor_store.snapshot_for(
            Phase2Fixture.COMFORTABLE_OCCUPIED
        )
        first.log_reasoning(
            build_reasoning_fixture(
                Phase2Fixture.COMFORTABLE_OCCUPIED,
                first_snapshot,
            )
        )
        self.assertEqual(len(first.reasoning_ledger.records), 1)
        self.assertEqual(second.reasoning_ledger.records, ())
        self.assertEqual(request.snapshot_id, second.sensor_store.current.snapshot_id)

    def test_bundle_sensor_and_carbon_stores_share_snapshot_identity(self) -> None:
        services = Phase2Services.deterministic()
        snapshot = services.sensor_store.current

        response = services.grid_carbon_store.read(
            GridCarbonIntensityRequest(
                request_id="bundle-carbon-1",
                snapshot_id=snapshot.snapshot_id,
                forecast_steps=1,
            )
        )

        self.assertEqual(response.snapshot_id, snapshot.snapshot_id)
        self.assertEqual(
            response.forecast[0].timestamp - response.current.timestamp,
            timedelta(minutes=15),
        )

    def test_stale_fixture_bundle_keeps_a_later_snapshot_current(self) -> None:
        services = Phase2Services.deterministic(
            initial_fixture="stale_snapshot"
        )
        stale = services.sensor_store.snapshot_for(
            Phase2Fixture.STALE_SNAPSHOT
        )

        self.assertTrue(services.sensor_store.is_stale(stale.snapshot_id))
        self.assertEqual(
            services.sensor_store.current,
            services.sensor_store.snapshot_for(
                Phase2Fixture.COMFORTABLE_OCCUPIED
            ),
        )

    def test_runtime_error_fixture_bundle_is_preseeded(self) -> None:
        services = Phase2Services.deterministic(
            initial_fixture="actuator_runtime_error"
        )
        current = services.sensor_store.current

        self.assertEqual(len(services.runtime_error_store.records), 1)
        self.assertEqual(
            services.runtime_error_store.records[0].error.code,
            "ACTUATOR_WRITEBACK_FAILED",
        )
        self.assertTrue(
            services.runtime_error_store.has_blocking_error(current.cycle_id)
        )

    def test_bundle_uses_one_transaction_lock_per_independent_instance(
        self,
    ) -> None:
        first = Phase2Services.deterministic()
        second = Phase2Services.deterministic()

        self.assertIs(first.sensor_store._lock, first._transaction_lock)
        self.assertIs(first.action_ledger._lock, first._transaction_lock)
        self.assertIs(first.reasoning_ledger._lock, first._transaction_lock)
        self.assertIs(
            first.runtime_error_store._lock,
            first._transaction_lock,
        )
        self.assertIsNot(first._transaction_lock, second._transaction_lock)


if __name__ == "__main__":
    unittest.main()
