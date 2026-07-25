"""Focused tests for the live EnergyPlus-to-scripted-agent bridge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.energyplus_wrapper import (
    ControlAction,
    SafetyLimits,
    SensorSnapshot,
    ZoneSensorData,
    ZoneSetpoints,
)
from src.live_energyplus_integration import (
    LiveAgentOutcome,
    LiveScriptedAgentController,
    map_phase1_snapshot,
)
from src.phase2_mock_services import PHASE1_ZONE_IDS
from src.scripted_provider import ScriptedProviderExhausted


SAFETY_LIMITS = SafetyLimits(
    heating_minimum_c=16.0,
    heating_maximum_c=24.0,
    cooling_minimum_c=20.0,
    cooling_maximum_c=30.0,
    minimum_deadband_c=1.0,
    writeback_tolerance_c=0.1,
)


def phase1_snapshot(sequence: int) -> SensorSnapshot:
    zone_step = ((sequence - 1) % 4) + 1
    hour = (sequence - 1) // 4
    return SensorSnapshot(
        sequence=sequence,
        environment_number=3,
        simulation_time_hours=sequence * 0.25,
        calendar_year=2026,
        month=7,
        day_of_month=21,
        hour=hour,
        minute=zone_step * 15,
        zone_timestep_number=zone_step,
        outdoor_drybulb_c=31.5,
        facility_electricity_j=900_000.0,
        facility_electricity_demand_w=7_500.0,
        zones={
            zone_id: ZoneSensorData(
                air_temperature_c=22.0 + index / 10,
                relative_humidity_pct=45.0,
                co2_ppm=700.0,
                occupant_count=2.0,
                fanger_pmv=0.1,
                heating_setpoint_c=20.0,
                cooling_setpoint_c=26.0,
            )
            for index, zone_id in enumerate(PHASE1_ZONE_IDS)
        },
    )


def accepted_outcome(
    decision_number: int,
    *,
    heating_c: float | None = None,
    cooling_c: float = 25.0,
) -> LiveAgentOutcome:
    heating = (
        20.0 + decision_number
        if heating_c is None
        else heating_c
    )
    return LiveAgentOutcome(
        action=ControlAction(
            setpoints={
                zone_id: ZoneSetpoints(
                    heating_c=heating,
                    cooling_c=cooling_c,
                )
                for zone_id in PHASE1_ZONE_IDS
            },
            source="test-scripted-agent",
            reason="MCP-validated test action",
        ),
        action_status="accepted",
        fallback_used=False,
        terminal_status="accepted",
        mcp_tools_called=(
            "read_sensor_data",
            "get_grid_carbon_intensity",
            "log_reasoning",
            "set_control_action",
        ),
        action_id=f"test-action-{decision_number}",
    )


class LiveEnergyPlusIntegrationTests(unittest.TestCase):
    def make_controller(
        self,
        temporary_root: Path,
        runner: object,
    ) -> LiveScriptedAgentController:
        return LiveScriptedAgentController(
            repository_root=Path(__file__).resolve().parents[1],
            work_directory=temporary_root / "live-work",
            controlled_zones=PHASE1_ZONE_IDS,
            safety_limits=SAFETY_LIMITS,
            cycle_runner=runner,  # type: ignore[arg-type]
        )

    def test_phase1_sensor_maps_to_phase2_live_snapshot(self) -> None:
        source = phase1_snapshot(4)

        mapped = map_phase1_snapshot(
            source,
            facility_electricity_kwh_since_start=1.25,
        )

        self.assertEqual(mapped.source.value, "energyplus")
        self.assertEqual(mapped.sequence, 4)
        self.assertEqual(mapped.snapshot_id, "energyplus-3-000004")
        self.assertEqual(mapped.timestamp.isoformat(), "2026-07-21T01:00:00+00:00")
        self.assertEqual(mapped.facility_electricity_demand_w, 7_500.0)
        self.assertEqual(mapped.facility_electricity_kwh_since_start, 1.25)
        self.assertEqual(
            tuple(zone.zone_id for zone in mapped.zones),
            PHASE1_ZONE_IDS,
        )
        self.assertEqual(mapped.zones[0].air_temperature_c, 22.0)
        self.assertEqual(mapped.zones[0].fanger_pmv, 0.1)
        self.assertEqual(mapped.zones[0].occupant_count, 2.0)

    def test_agent_decision_runs_once_each_simulated_hour(self) -> None:
        decisions: list[tuple[int, int]] = []

        def runner(
            snapshot: object,
            history: tuple[object, ...],
            number: int,
        ) -> LiveAgentOutcome:
            decisions.append((getattr(snapshot, "sequence"), number))
            return accepted_outcome(number, heating_c=21.0)

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary), runner)
            for sequence in range(1, 13):
                controller(phase1_snapshot(sequence))

        self.assertEqual(decisions, [(4, 1), (8, 2), (12, 3)])

    def test_accepted_action_is_held_for_four_timesteps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary),
                lambda snapshot, history, number: accepted_outcome(number),
            )
            actions = [
                controller(phase1_snapshot(sequence))
                for sequence in range(1, 9)
            ]

        self.assertEqual(actions[:3], [None, None, None])
        first_hour = [
            action.setpoints["SPACE1-1"].heating_c
            for action in actions[3:7]
            if action is not None
        ]
        self.assertEqual(first_hour, [21.0, 21.0, 21.0, 21.0])
        self.assertIsNotNone(actions[7])
        assert actions[7] is not None
        self.assertEqual(
            actions[7].setpoints["SPACE1-1"].heating_c,
            22.0,
        )

    def test_unsafe_agent_action_falls_back_before_writeback(self) -> None:
        def unsafe_runner(
            snapshot: object,
            history: tuple[object, ...],
            number: int,
        ) -> LiveAgentOutcome:
            return accepted_outcome(
                number,
                heating_c=15.0,
                cooling_c=31.0,
            )

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary),
                unsafe_runner,
            )
            actions = [
                controller(phase1_snapshot(sequence))
                for sequence in range(1, 5)
            ]
            event = dict(controller.events[-1])

        selected = actions[-1]
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.setpoints, {})
        self.assertEqual(selected.source, "phase2-scripted-fallback")
        self.assertTrue(event["fallback_used"])
        self.assertEqual(event["action_status"], "fallback_release")

    def test_scripted_agent_failure_uses_release_fallback(self) -> None:
        def failing_runner(
            snapshot: object,
            history: tuple[object, ...],
            number: int,
        ) -> LiveAgentOutcome:
            raise ScriptedProviderExhausted("intentional test failure")

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary),
                failing_runner,
            )
            for sequence in range(1, 5):
                selected = controller(phase1_snapshot(sequence))
            event = dict(controller.events[-1])

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.setpoints, {})
        self.assertTrue(event["fallback_used"])
        self.assertEqual(event["mcp_tools_called"], [])


if __name__ == "__main__":
    unittest.main()
