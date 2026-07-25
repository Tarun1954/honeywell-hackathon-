"""Thin live bridge from Phase 1 EnergyPlus callbacks to the Phase 2 agent."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any

import anyio
import yaml
from mcp import StdioServerParameters

from src.energyplus_wrapper import (
    ControlAction,
    SafetyLimits,
    SensorSnapshot as Phase1SensorSnapshot,
    ZoneSetpoints,
    validate_control_action as validate_phase1_control_action,
)
from src.mcp_client import Phase2MCPClient
from src.phase2_agent import (
    AgentCycleResult,
    AgentLoopLimits,
    Phase2AgentOrchestrator,
)
from src.phase2_contracts import (
    CarbonIntensityCategory,
    CarbonIntensitySample,
    ControlActionStatus,
    GridCarbonIntensityRequest,
    GridCarbonIntensityResponse,
    ReadSensorDataRequest,
    ReadSensorDataResponse,
    SensorSnapshot,
    SensorSource,
    SetControlActionRequest,
    SetControlActionResponse,
    SetZoneCommand,
    ZoneSensorData,
)
from src.phase2_mock_services import (
    DEFAULT_PHASE2_CONFIG,
    MockActionLedger,
    MockRuntimeErrorStore,
    Phase2Services,
    ReasoningLedger,
)
from src.phase2_validation import SafetyPolicy
from src.scripted_provider import ScriptedProvider, ScriptedToolCall


DEFAULT_DECISION_INTERVAL_STEPS = 4
DEFAULT_HOLD_STEPS = 4
DEFAULT_HEATING_SETPOINT_C = 21.0
DEFAULT_COOLING_SETPOINT_C = 25.0


def map_phase1_snapshot(
    snapshot: Phase1SensorSnapshot,
    *,
    facility_electricity_kwh_since_start: float,
) -> SensorSnapshot:
    """Map one completed live Phase 1 reading into the Phase 2 contract."""

    # EnergyPlus reports local simulation clock fields without a timezone.
    # A fixed UTC attachment makes the ordering explicit without altering
    # the simulated wall-clock values.
    timestamp = datetime.fromisoformat(
        snapshot.energyplus_timestamp
    ).replace(tzinfo=UTC)
    return SensorSnapshot(
        cycle_id=f"live-cycle-{snapshot.environment_number}-{snapshot.sequence:06d}",
        snapshot_id=(
            f"energyplus-{snapshot.environment_number}-{snapshot.sequence:06d}"
        ),
        sequence=snapshot.sequence,
        timestamp=timestamp,
        source=SensorSource.ENERGYPLUS,
        outdoor_drybulb_c=snapshot.outdoor_drybulb_c,
        facility_electricity_demand_w=snapshot.facility_electricity_demand_w,
        facility_electricity_kwh_since_start=(
            facility_electricity_kwh_since_start
        ),
        zones=tuple(
            ZoneSensorData(
                zone_id=zone_id,
                air_temperature_c=zone.air_temperature_c,
                relative_humidity_pct=zone.relative_humidity_pct,
                co2_ppm=zone.co2_ppm,
                occupant_count=zone.occupant_count,
                fanger_pmv=zone.fanger_pmv,
                heating_setpoint_c=zone.heating_setpoint_c,
                cooling_setpoint_c=zone.cooling_setpoint_c,
            )
            for zone_id, zone in snapshot.zones.items()
        ),
    )


class LiveSensorStore:
    """Read-only MCP sensor store loaded with one live snapshot and history."""

    def __init__(
        self,
        snapshot: SensorSnapshot,
        history: tuple[SensorSnapshot, ...] = (),
        *,
        lock: RLock | None = None,
    ) -> None:
        self._current = snapshot
        self._history = history
        self._by_id = {
            item.snapshot_id: item for item in (*history, snapshot)
        }
        self._lock = lock or RLock()

    @property
    def current(self) -> SensorSnapshot:
        with self._lock:
            return self._current

    def get(self, snapshot_id: str) -> SensorSnapshot:
        try:
            return self._by_id[snapshot_id]
        except KeyError as exc:
            raise KeyError(f"unknown snapshot_id: {snapshot_id}") from exc

    def is_stale(self, snapshot_id: str) -> bool:
        return snapshot_id != self._current.snapshot_id

    def read(self, request: ReadSensorDataRequest) -> ReadSensorDataResponse:
        with self._lock:
            selected_history = (
                self._history[-request.history_steps :]
                if request.history_steps
                else ()
            )
            return ReadSensorDataResponse(
                request_id=request.request_id,
                snapshot=self._current,
                history=selected_history,
            )


class LiveGridCarbonStore:
    """Stable placeholder for the existing MCP carbon tool surface."""

    def __init__(self, sensor_store: LiveSensorStore) -> None:
        self._sensor_store = sensor_store

    def read(
        self,
        request: GridCarbonIntensityRequest,
    ) -> GridCarbonIntensityResponse:
        snapshot = self._sensor_store.get(request.snapshot_id)
        current = CarbonIntensitySample(
            offset_steps=0,
            timestamp=snapshot.timestamp,
            g_co2_per_kwh=350.0,
            category=CarbonIntensityCategory.MODERATE,
        )
        forecast = tuple(
            CarbonIntensitySample(
                offset_steps=offset,
                timestamp=snapshot.timestamp + offset * timedelta(minutes=15),
                g_co2_per_kwh=350.0,
                category=CarbonIntensityCategory.MODERATE,
            )
            for offset in range(1, request.forecast_steps + 1)
        )
        return GridCarbonIntensityResponse(
            request_id=request.request_id,
            snapshot_id=request.snapshot_id,
            current=current,
            forecast=forecast,
        )


class RecordingActionLedger(MockActionLedger):
    """Persist the exact MCP-validated action for the EnergyPlus parent."""

    def __init__(
        self,
        output_path: Path,
        *,
        lock: RLock | None = None,
    ) -> None:
        super().__init__(lock=lock)
        self._output_path = output_path

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
        response = super().submit(
            request,
            snapshot=snapshot,
            policy=policy,
            known_reasoning_logs=known_reasoning_logs,
            runtime_error_pending=runtime_error_pending,
            action_id=action_id,
        )
        if response.status in {
            ControlActionStatus.ACCEPTED,
            ControlActionStatus.DUPLICATE,
        }:
            payload = {
                "request": request.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
            }
            self._output_path.write_text(
                json.dumps(payload, allow_nan=False, sort_keys=True),
                encoding="utf-8",
            )
        return response


def _load_safety_policy(path: Path) -> SafetyPolicy:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("safety"), dict
    ):
        raise ValueError(f"Phase 2 config has no safety mapping: {path}")
    return SafetyPolicy.model_validate(payload["safety"])


def create_live_services(
    snapshot: SensorSnapshot,
    history: tuple[SensorSnapshot, ...],
    *,
    action_output_path: Path,
    config_path: Path = DEFAULT_PHASE2_CONFIG,
) -> Phase2Services:
    """Create one isolated MCP service bundle around live EnergyPlus data."""

    transaction_lock = RLock()
    sensor_store = LiveSensorStore(
        snapshot,
        history,
        lock=transaction_lock,
    )
    return Phase2Services(
        sensor_store=sensor_store,  # type: ignore[arg-type]
        grid_carbon_store=LiveGridCarbonStore(sensor_store),  # type: ignore[arg-type]
        action_ledger=RecordingActionLedger(
            action_output_path,
            lock=transaction_lock,
        ),
        reasoning_ledger=ReasoningLedger(lock=transaction_lock),
        runtime_error_store=MockRuntimeErrorStore(lock=transaction_lock),
        safety_policy=_load_safety_policy(config_path),
        _transaction_lock=transaction_lock,
    )


@dataclass(frozen=True, slots=True)
class LiveAgentOutcome:
    """Validated action and bounded MCP evidence returned to the callback."""

    action: ControlAction
    action_status: str
    fallback_used: bool
    terminal_status: str
    mcp_tools_called: tuple[str, ...]
    action_id: str | None = None
    provider_name: str = "scripted"


AgentCycleRunner = Callable[
    [SensorSnapshot, tuple[SensorSnapshot, ...], int],
    LiveAgentOutcome,
]
ProviderFactory = Callable[[SensorSnapshot], Any]


def _set_commands(
    zones: tuple[str, ...],
    heating_c: float,
    cooling_c: float,
) -> list[dict[str, Any]]:
    return [
        {
            "mode": "set",
            "zone_id": zone,
            "heating_c": heating_c,
            "cooling_c": cooling_c,
        }
        for zone in zones
    ]


def build_live_scripted_provider(
    *,
    sequence: int,
    zones: tuple[str, ...],
    heating_c: float = DEFAULT_HEATING_SETPOINT_C,
    cooling_c: float = DEFAULT_COOLING_SETPOINT_C,
    hold_steps: int = DEFAULT_HOLD_STEPS,
) -> ScriptedProvider:
    """Build the one deterministic live decision used by the smoke path."""

    prefix = f"live-{sequence:06d}"
    return ScriptedProvider(
        [
            ScriptedToolCall(
                call_id=f"{prefix}-read",
                tool_name="read_sensor_data",
                arguments={"history_steps": 2},
            ),
            ScriptedToolCall(
                call_id=f"{prefix}-carbon",
                tool_name="get_grid_carbon_intensity",
                arguments={"forecast_steps": 4},
            ),
            ScriptedToolCall(
                call_id=f"{prefix}-reason",
                tool_name="log_reasoning",
                arguments={
                    "decision_summary": (
                        "Apply the proven thermostat setpoints for one hour."
                    ),
                    "objective_tags": [
                        "thermal_comfort",
                        "energy_reduction",
                        "safety",
                    ],
                    "tradeoff_summary": (
                        "Use conservative occupied comfort limits and the "
                        "existing thermostat actuator surface."
                    ),
                    "confidence": 0.95,
                },
            ),
            ScriptedToolCall(
                call_id=f"{prefix}-action",
                tool_name="set_control_action",
                arguments={
                    "commands": _set_commands(
                        zones,
                        heating_c,
                        cooling_c,
                    ),
                    "hold_steps": hold_steps,
                },
            ),
        ],
        name="live-scripted",
    )


def _request_to_phase1_action(
    request: SetControlActionRequest,
    *,
    source: str,
    reason: str,
    controlled_zones: tuple[str, ...],
    safety_limits: SafetyLimits,
) -> ControlAction:
    setpoints = {
        command.zone_id: ZoneSetpoints(
            heating_c=command.heating_c,
            cooling_c=command.cooling_c,
        )
        for command in request.commands
        if isinstance(command, SetZoneCommand)
    }
    action = ControlAction(
        setpoints=setpoints,
        source=source,
        reason=reason,
    )
    return validate_phase1_control_action(
        action,
        controlled_zones,
        safety_limits,
    )


def deterministic_release_fallback(
    *,
    controlled_zones: tuple[str, ...],
    safety_limits: SafetyLimits,
    reason: str,
) -> ControlAction:
    """Return the existing safe behavior: release all proven actuators."""

    action = ControlAction(
        setpoints={},
        source="phase2-scripted-fallback",
        reason=reason,
    )
    return validate_phase1_control_action(
        action,
        controlled_zones,
        safety_limits,
    )


class LiveScriptedAgentController:
    """Synchronous Phase 1 policy backed by hourly real MCP agent cycles."""

    def __init__(
        self,
        *,
        repository_root: Path,
        work_directory: Path,
        controlled_zones: tuple[str, ...],
        safety_limits: SafetyLimits,
        decision_interval_steps: int = DEFAULT_DECISION_INTERVAL_STEPS,
        hold_steps: int = DEFAULT_HOLD_STEPS,
        heating_c: float = DEFAULT_HEATING_SETPOINT_C,
        cooling_c: float = DEFAULT_COOLING_SETPOINT_C,
        cycle_runner: AgentCycleRunner | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        if decision_interval_steps != 4:
            raise ValueError("live decisions must run every four 15-minute steps")
        if hold_steps != 4:
            raise ValueError("live actions must hold for four 15-minute steps")
        self.repository_root = repository_root.resolve()
        self.work_directory = work_directory.resolve()
        self.work_directory.mkdir(parents=True, exist_ok=True)
        self.controlled_zones = controlled_zones
        self.safety_limits = safety_limits
        self.decision_interval_steps = decision_interval_steps
        self.hold_steps = hold_steps
        self.heating_c = heating_c
        self.cooling_c = cooling_c
        self._cycle_runner = cycle_runner or self._run_real_mcp_cycle
        self._provider_factory = provider_factory
        self._facility_electricity_kwh = 0.0
        self._history: list[SensorSnapshot] = []
        self._active_action: ControlAction | None = None
        self._hold_returns_remaining = 0
        self._decision_count = 0
        self._last_actuator_write: dict[str, Any] = {
            "status": "not_yet_observed"
        }
        self._events: list[dict[str, Any]] = []
        self._last_outcome: LiveAgentOutcome | None = None
        self.log_path = self.work_directory / "live_agent_timesteps.jsonl"
        if self.log_path.exists():
            raise FileExistsError(
                f"Live integration log already exists: {self.log_path}"
            )

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._events)

    @property
    def last_outcome(self) -> LiveAgentOutcome | None:
        return self._last_outcome

    def record_actuator_write(self, result: Mapping[str, Any]) -> None:
        """Receive write evidence from the existing Phase 1 callback."""

        self._last_actuator_write = json.loads(
            json.dumps(result, allow_nan=False)
        )

    def __call__(
        self,
        phase1_snapshot: Phase1SensorSnapshot,
    ) -> ControlAction | None:
        self._facility_electricity_kwh += (
            phase1_snapshot.facility_electricity_j / 3_600_000.0
        )
        snapshot = map_phase1_snapshot(
            phase1_snapshot,
            facility_electricity_kwh_since_start=(
                self._facility_electricity_kwh
            ),
        )
        decision_outcome: LiveAgentOutcome | None = None
        if snapshot.sequence % self.decision_interval_steps == 0:
            self._decision_count += 1
            try:
                decision_outcome = self._cycle_runner(
                    snapshot,
                    tuple(self._history[-4:]),
                    self._decision_count,
                )
                validate_phase1_control_action(
                    decision_outcome.action,
                    self.controlled_zones,
                    self.safety_limits,
                )
            except BaseException as exc:
                decision_outcome = LiveAgentOutcome(
                    action=deterministic_release_fallback(
                        controlled_zones=self.controlled_zones,
                        safety_limits=self.safety_limits,
                        reason=(
                            "The live provider failed safely: "
                            f"{type(exc).__name__}"
                        ),
                    ),
                    action_status="fallback_release",
                    fallback_used=True,
                    terminal_status="failed",
                    mcp_tools_called=(),
                    provider_name="deterministic-fallback",
                )
            self._active_action = decision_outcome.action
            self._hold_returns_remaining = self.hold_steps
            self._last_outcome = decision_outcome

        selected_action = (
            self._active_action
            if self._hold_returns_remaining > 0
            else None
        )
        if selected_action is not None:
            validate_phase1_control_action(
                selected_action,
                self.controlled_zones,
                self.safety_limits,
            )
            self._hold_returns_remaining -= 1

        event = self._build_event(
            snapshot,
            selected_action,
            decision_outcome,
        )
        self._events.append(event)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    event,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
        self._history.append(snapshot)
        return selected_action

    def _build_event(
        self,
        snapshot: SensorSnapshot,
        action: ControlAction | None,
        outcome: LiveAgentOutcome | None,
    ) -> dict[str, Any]:
        previous_setpoints = {
            zone.zone_id: {
                "heating_c": zone.heating_setpoint_c,
                "cooling_c": zone.cooling_setpoint_c,
            }
            for zone in snapshot.zones
        }
        if action is None:
            chosen_setpoints: dict[str, Any] = {}
        else:
            chosen_setpoints = {
                zone: (
                    {
                        "mode": "set",
                        "heating_c": action.setpoints[zone].heating_c,
                        "cooling_c": action.setpoints[zone].cooling_c,
                    }
                    if zone in action.setpoints
                    else {"mode": "release"}
                )
                for zone in self.controlled_zones
            }
        return {
            "event": "live_phase2_timestep",
            "simulated_timestamp": snapshot.timestamp.isoformat(),
            "snapshot_id": snapshot.snapshot_id,
            "cycle_id": snapshot.cycle_id,
            "sequence": snapshot.sequence,
            "zone_temperatures_c": {
                zone.zone_id: zone.air_temperature_c
                for zone in snapshot.zones
            },
            "pmv": {
                zone.zone_id: zone.fanger_pmv for zone in snapshot.zones
            },
            "occupancy": {
                zone.zone_id: zone.occupant_count for zone in snapshot.zones
            },
            "facility_electricity_demand_w": (
                snapshot.facility_electricity_demand_w
            ),
            "facility_electricity_kwh_since_start": (
                snapshot.facility_electricity_kwh_since_start
            ),
            "previous_setpoints": previous_setpoints,
            "chosen_setpoints": chosen_setpoints,
            "mcp_tools_called": (
                list(outcome.mcp_tools_called) if outcome is not None else []
            ),
            "provider_name": (
                outcome.provider_name
                if outcome is not None
                else (
                    self._last_outcome.provider_name
                    if action is not None and self._last_outcome is not None
                    else None
                )
            ),
            "action_status": (
                outcome.action_status
                if outcome is not None
                else ("held" if action is not None else "no_action")
            ),
            "fallback_used": (
                outcome.fallback_used
                if outcome is not None
                else bool(
                    action is not None
                    and action.source == "phase2-scripted-fallback"
                )
            ),
            "actuator_write_result": self._last_actuator_write,
        }

    def _run_real_mcp_cycle(
        self,
        snapshot: SensorSnapshot,
        history: tuple[SensorSnapshot, ...],
        decision_number: int,
    ) -> LiveAgentOutcome:
        payload_path = (
            self.work_directory
            / f"cycle-{decision_number:03d}-snapshot.json"
        )
        action_path = (
            self.work_directory
            / f"cycle-{decision_number:03d}-validated-action.json"
        )
        payload_path.write_text(
            json.dumps(
                {
                    "snapshot": snapshot.model_dump(mode="json"),
                    "history": [
                        item.model_dump(mode="json") for item in history
                    ],
                },
                allow_nan=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-u",
                "-m",
                "src.live_mcp_server",
                "--snapshot-payload",
                str(payload_path),
                "--action-output",
                str(action_path),
            ],
            cwd=self.repository_root,
            env=None,
            encoding="utf-8",
            encoding_error_handler="strict",
        )
        provider = (
            self._provider_factory(snapshot)
            if self._provider_factory is not None
            else build_live_scripted_provider(
                sequence=snapshot.sequence,
                zones=self.controlled_zones,
                heating_c=self.heating_c,
                cooling_c=self.cooling_c,
                hold_steps=self.hold_steps,
            )
        )

        async def execute() -> AgentCycleResult:
            orchestrator = Phase2AgentOrchestrator(
                client_factory=lambda: Phase2MCPClient(
                    parameters,
                    allow_read_reconnect=False,
                ),
                limits=AgentLoopLimits(
                    provider_response_timeout_seconds=30.0
                ),
            )
            return await orchestrator.run_cycle(
                provider,
                run_id=f"live-{snapshot.sequence:06d}",
            )

        result = anyio.run(execute)
        if not action_path.is_file():
            raise RuntimeError(
                "MCP cycle did not persist an accepted action"
            )
        action_payload = json.loads(action_path.read_text(encoding="utf-8"))
        request = SetControlActionRequest.model_validate(
            action_payload["request"]
        )
        response = SetControlActionResponse.model_validate(
            action_payload["response"]
        )
        if (
            response.status
            not in {
                ControlActionStatus.ACCEPTED,
                ControlActionStatus.DUPLICATE,
            }
            or result.record.action_status != response.status
            or result.record.action_id != response.action_id
        ):
            raise RuntimeError(
                "MCP action evidence does not match the agent result"
            )
        action = _request_to_phase1_action(
            request,
            source=(
                "phase2-scripted-fallback"
                if result.record.fallback_used
                else f"phase2-{result.record.provider_name}-agent"
            ),
            reason=(
                "MCP-validated deterministic release fallback."
                if result.record.fallback_used
                else "MCP-validated hourly live-provider action."
            ),
            controlled_zones=self.controlled_zones,
            safety_limits=self.safety_limits,
        )
        return LiveAgentOutcome(
            action=action,
            action_status=response.status.value,
            fallback_used=result.record.fallback_used,
            terminal_status=result.record.terminal_status.value,
            mcp_tools_called=result.record.tool_sequence,
            action_id=response.action_id,
            provider_name=result.record.provider_name,
        )


__all__ = [
    "DEFAULT_COOLING_SETPOINT_C",
    "DEFAULT_DECISION_INTERVAL_STEPS",
    "DEFAULT_HEATING_SETPOINT_C",
    "DEFAULT_HOLD_STEPS",
    "LiveAgentOutcome",
    "LiveGridCarbonStore",
    "LiveScriptedAgentController",
    "LiveSensorStore",
    "ProviderFactory",
    "RecordingActionLedger",
    "build_live_scripted_provider",
    "create_live_services",
    "deterministic_release_fallback",
    "map_phase1_snapshot",
]
