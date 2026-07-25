"""Typed, defensive EnergyPlus Runtime/DataExchange wrapper for Phase 1."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from dotenv import load_dotenv


LOGGER = logging.getLogger("eco_loop.phase1.energyplus")


class Phase1Error(RuntimeError):
    """Base class for Phase 1 failures."""


class ConfigurationError(Phase1Error):
    """Raised when a required configured path or value is invalid."""


class HandleResolutionError(Phase1Error):
    """Raised when EnergyPlus does not expose a required exchange handle."""


class DataExchangeError(Phase1Error):
    """Raised when an EnergyPlus exchange operation sets the API error flag."""


class InvalidControlAction(Phase1Error):
    """Raised when a proposed setpoint action violates safety constraints."""


class CallbackExecutionError(Phase1Error):
    """Raised after EnergyPlus returns if a ctypes callback failed."""


class EnergyPlusRunError(Phase1Error):
    """Raised when an EnergyPlus run returns a nonzero code or severe error."""


class EnergyCrosscheckError(Phase1Error):
    """Raised when API and EnergyPlus-native energy totals disagree."""


class RuntimeCleanupError(Phase1Error):
    """Raised when state/callback cleanup fails and retry would be unsafe."""


def _first_idf_object_fields(path: Path, object_type: str) -> tuple[str, ...]:
    """Return comment-free fields from the first matching IDF object."""

    text = path.read_text(encoding="utf-8", errors="replace")
    without_comments = "\n".join(line.split("!", 1)[0] for line in text.splitlines())
    match = re.search(
        rf"(?:^|;)\s*{re.escape(object_type)}\s*,(.*?);",
        without_comments,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ConfigurationError(f"Model has no {object_type} object: {path}")
    return tuple(field.strip() for field in match.group(1).split(","))


@dataclass(frozen=True)
class SafetyLimits:
    """Hard limits applied before any actuator write."""

    heating_minimum_c: float
    heating_maximum_c: float
    cooling_minimum_c: float
    cooling_maximum_c: float
    minimum_deadband_c: float
    writeback_tolerance_c: float


@dataclass(frozen=True)
class Phase1Config:
    """Resolved Phase 1 configuration."""

    repository_root: Path
    config_path: Path
    energyplus_version: str
    energyplus_home: Path
    executable: Path
    idd: Path
    baseline_model: Path
    runtime_model: Path
    weather_file: Path
    output_root: Path
    timestep_minutes: int
    run_period_start: tuple[int, int]
    run_period_end: tuple[int, int]
    controlled_zones: tuple[str, ...]
    people_objects: Mapping[str, str]
    safety: SafetyLimits
    max_attempts: int
    retry_delay_seconds: float
    diagnostic_heating_setpoint_c: float
    diagnostic_cooling_setpoint_c: float
    timestep_log_name: str
    summary_name: str

    @classmethod
    def load(cls, config_path: str | Path) -> "Phase1Config":
        """Load YAML, expand environment variables, and resolve all paths."""

        resolved_config = Path(config_path).resolve()
        if not resolved_config.is_file():
            raise ConfigurationError(f"Configuration file does not exist: {resolved_config}")

        repository_root = resolved_config.parent.parent
        load_dotenv(repository_root / ".env", override=False)
        raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError("Phase 1 configuration must contain a YAML mapping")

        try:
            engine = raw["engine"]
            model = raw["model"]
            weather = raw["weather"]
            output = raw["output"]
            simulation = raw["simulation"]
            safety_raw = raw["safety"]
            runtime = raw["runtime"]
        except KeyError as exc:
            raise ConfigurationError(f"Missing configuration section: {exc.args[0]}") from exc

        def resolve_repository_path(value: str) -> Path:
            expanded = os.path.expandvars(str(value))
            if "$" in expanded:
                raise ConfigurationError(f"Unresolved environment variable in path: {value}")
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = repository_root / candidate
            return candidate.resolve()

        energyplus_home = resolve_repository_path(str(engine["home"]))

        def resolve_engine_path(value: str) -> Path:
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = energyplus_home / candidate
            return candidate.resolve()

        zones = tuple(str(zone).strip() for zone in simulation["controlled_zones"])
        people_raw = simulation["people_objects"]
        people_objects = {
            str(zone).strip(): str(person).strip()
            for zone, person in people_raw.items()
        }
        missing_people = sorted(set(zones) - set(people_objects))
        if missing_people:
            raise ConfigurationError(
                f"Missing People-object mapping for zones: {', '.join(missing_people)}"
            )
        run_period = simulation["run_period"]

        limits = SafetyLimits(
            heating_minimum_c=float(safety_raw["heating_setpoint_c"]["minimum"]),
            heating_maximum_c=float(safety_raw["heating_setpoint_c"]["maximum"]),
            cooling_minimum_c=float(safety_raw["cooling_setpoint_c"]["minimum"]),
            cooling_maximum_c=float(safety_raw["cooling_setpoint_c"]["maximum"]),
            minimum_deadband_c=float(safety_raw["minimum_deadband_c"]),
            writeback_tolerance_c=float(safety_raw["writeback_tolerance_c"]),
        )

        configuration = cls(
            repository_root=repository_root,
            config_path=resolved_config,
            energyplus_version=str(engine["version"]),
            energyplus_home=energyplus_home,
            executable=resolve_engine_path(str(engine["executable"])),
            idd=resolve_engine_path(str(engine["idd"])),
            baseline_model=resolve_repository_path(str(model["baseline"])),
            runtime_model=resolve_repository_path(str(model["optimized_runtime"])),
            weather_file=resolve_repository_path(str(weather["file"])),
            output_root=resolve_repository_path(str(output["root"])),
            timestep_minutes=int(simulation["timestep_minutes"]),
            run_period_start=(
                int(run_period["start_month"]),
                int(run_period["start_day"]),
            ),
            run_period_end=(
                int(run_period["end_month"]),
                int(run_period["end_day"]),
            ),
            controlled_zones=zones,
            people_objects=people_objects,
            safety=limits,
            max_attempts=int(runtime["max_attempts"]),
            retry_delay_seconds=float(runtime["retry_delay_seconds"]),
            diagnostic_heating_setpoint_c=float(
                runtime["diagnostic_heating_setpoint_c"]
            ),
            diagnostic_cooling_setpoint_c=float(
                runtime["diagnostic_cooling_setpoint_c"]
            ),
            timestep_log_name=str(output["timestep_log"]),
            summary_name=str(output["summary"]),
        )
        configuration.validate_values()
        return configuration

    def validate_values(self) -> None:
        """Reject invalid scalar and mapping configuration before API startup."""

        if self.timestep_minutes <= 0 or 60 % self.timestep_minutes:
            raise ConfigurationError(
                "simulation.timestep_minutes must be a positive divisor of 60"
            )
        if not self.controlled_zones:
            raise ConfigurationError("At least one controlled zone is required")
        if any(not zone for zone in self.controlled_zones):
            raise ConfigurationError("Controlled-zone names must not be empty")
        if len(set(self.controlled_zones)) != len(self.controlled_zones):
            raise ConfigurationError("Controlled-zone names must be unique")
        zone_names = set(self.controlled_zones)
        mapping_names = set(self.people_objects)
        if mapping_names != zone_names:
            missing = sorted(zone_names - mapping_names)
            extra = sorted(mapping_names - zone_names)
            raise ConfigurationError(
                "People-object mappings must exactly match controlled zones; "
                f"missing={missing}, extra={extra}"
            )
        if any(not person for person in self.people_objects.values()):
            raise ConfigurationError("People-object names must not be empty")

        values = {
            "heating minimum": self.safety.heating_minimum_c,
            "heating maximum": self.safety.heating_maximum_c,
            "cooling minimum": self.safety.cooling_minimum_c,
            "cooling maximum": self.safety.cooling_maximum_c,
            "minimum deadband": self.safety.minimum_deadband_c,
            "writeback tolerance": self.safety.writeback_tolerance_c,
            "retry delay": self.retry_delay_seconds,
            "diagnostic heating setpoint": self.diagnostic_heating_setpoint_c,
            "diagnostic cooling setpoint": self.diagnostic_cooling_setpoint_c,
        }
        nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
        if nonfinite:
            raise ConfigurationError(
                f"Configuration values must be finite: {', '.join(nonfinite)}"
            )
        if self.safety.heating_minimum_c > self.safety.heating_maximum_c:
            raise ConfigurationError("Heating setpoint minimum exceeds maximum")
        if self.safety.cooling_minimum_c > self.safety.cooling_maximum_c:
            raise ConfigurationError("Cooling setpoint minimum exceeds maximum")
        if self.safety.minimum_deadband_c < 0:
            raise ConfigurationError("Minimum deadband must be nonnegative")
        if self.safety.writeback_tolerance_c < 0:
            raise ConfigurationError("Writeback tolerance must be nonnegative")
        if self.max_attempts < 1:
            raise ConfigurationError("runtime.max_attempts must be at least 1")
        if self.retry_delay_seconds < 0:
            raise ConfigurationError("runtime.retry_delay_seconds must be nonnegative")
        if not (
            self.safety.heating_minimum_c
            <= self.diagnostic_heating_setpoint_c
            <= self.safety.heating_maximum_c
        ):
            raise ConfigurationError("Diagnostic heating setpoint is outside limits")
        if not (
            self.safety.cooling_minimum_c
            <= self.diagnostic_cooling_setpoint_c
            <= self.safety.cooling_maximum_c
        ):
            raise ConfigurationError("Diagnostic cooling setpoint is outside limits")
        if (
            self.diagnostic_cooling_setpoint_c
            - self.diagnostic_heating_setpoint_c
            < self.safety.minimum_deadband_c
        ):
            raise ConfigurationError("Diagnostic setpoints violate minimum deadband")
        for label, value in (
            ("output.timestep_log", self.timestep_log_name),
            ("output.summary", self.summary_name),
        ):
            if not value or Path(value).name != value:
                raise ConfigurationError(f"{label} must be a plain file name")
        for label, (month, day) in (
            ("run-period start", self.run_period_start),
            ("run-period end", self.run_period_end),
        ):
            try:
                time.strptime(f"2021-{month:02d}-{day:02d}", "%Y-%m-%d")
            except ValueError as exc:
                raise ConfigurationError(f"Invalid {label}: {month}/{day}") from exc

    def validate_paths(self, model_path: Path | None = None) -> None:
        """Fail before simulation when a required local asset is missing."""

        required = {
            "EnergyPlus home": self.energyplus_home,
            "EnergyPlus executable": self.executable,
            "EnergyPlus IDD": self.idd,
            "EnergyPlus Python API": self.energyplus_home / "pyenergyplus" / "api.py",
            "EnergyPlus API DLL": self.energyplus_home / "EnergyPlusAPI.dll",
            "model": model_path or self.baseline_model,
            "weather": self.weather_file,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise ConfigurationError("Missing required Phase 1 assets:\n" + "\n".join(missing))
        self.validate_values()

        selected_model = model_path or self.baseline_model
        timestep_fields = _first_idf_object_fields(selected_model, "Timestep")
        try:
            model_steps_per_hour = int(timestep_fields[0])
        except (IndexError, ValueError) as exc:
            raise ConfigurationError(
                f"Model Timestep is not an integer: {selected_model}"
            ) from exc
        configured_steps_per_hour = 60 // self.timestep_minutes
        if model_steps_per_hour != configured_steps_per_hour:
            raise ConfigurationError(
                "Configured timestep does not match model: "
                f"{configured_steps_per_hour} vs {model_steps_per_hour} per hour"
            )

        run_period_fields = _first_idf_object_fields(selected_model, "RunPeriod")
        try:
            model_start = (int(run_period_fields[1]), int(run_period_fields[2]))
            model_end = (int(run_period_fields[4]), int(run_period_fields[5]))
        except (IndexError, ValueError) as exc:
            raise ConfigurationError(
                f"Model RunPeriod is malformed: {selected_model}"
            ) from exc
        if (
            model_start != self.run_period_start
            or model_end != self.run_period_end
        ):
            raise ConfigurationError(
                "Configured run period does not match model: "
                f"{self.run_period_start}-{self.run_period_end} vs "
                f"{model_start}-{model_end}"
            )


@dataclass(frozen=True)
class ZoneSensorData:
    """One controlled zone's completed-timestep state."""

    air_temperature_c: float
    relative_humidity_pct: float
    co2_ppm: float
    occupant_count: float
    fanger_pmv: float
    heating_setpoint_c: float
    cooling_setpoint_c: float


@dataclass(frozen=True)
class SensorSnapshot:
    """Completed EnergyPlus zone-timestep observation."""

    sequence: int
    environment_number: int
    simulation_time_hours: float
    calendar_year: int
    month: int
    day_of_month: int
    hour: int
    minute: int
    zone_timestep_number: int
    outdoor_drybulb_c: float
    facility_electricity_j: float
    facility_electricity_demand_w: float
    zones: Mapping[str, ZoneSensorData]

    @property
    def energyplus_timestamp(self) -> str:
        """Return a normalized ISO timestamp for the completed timestep."""

        timestamp = datetime(
            self.calendar_year,
            self.month,
            self.day_of_month,
        ) + timedelta(
            hours=self.hour,
            minutes=self.minute,
        )
        return timestamp.isoformat(timespec="seconds")


@dataclass(frozen=True)
class ZoneSetpoints:
    """Heating and cooling setpoints for one zone."""

    heating_c: float
    cooling_c: float


@dataclass(frozen=True)
class ControlAction:
    """A policy decision to be applied at the next predictor callback."""

    setpoints: Mapping[str, ZoneSetpoints]
    source: str
    reason: str


Policy: TypeAlias = Callable[[SensorSnapshot], ControlAction | None]


@dataclass(frozen=True)
class EnergyPlusErrorSummary:
    """Parsed contents of an EnergyPlus .err file."""

    warning_count: int
    severe_count: int
    fatal_count: int
    messages: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    """Successful Phase 1 run evidence."""

    run_id: str
    mode: str
    model_path: str
    output_directory: str
    attempt: int
    exit_code: int
    timestep_count: int
    facility_electricity_kwh: float
    hvac_electricity_kwh: float
    natural_gas_kwh: float
    csv_facility_electricity_kwh: float
    energy_crosscheck_relative_error: float
    warning_count: int
    severe_count: int
    fatal_count: int
    simulator_error_callback_count: int
    elapsed_seconds: float
    first_snapshot: Mapping[str, Any] | None
    last_snapshot: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result for the Phase 1 evidence artifact."""

        return asdict(self)


@dataclass
class _HandleRegistry:
    variables: dict[str, int]
    meters: dict[str, int]
    actuators: dict[str, dict[str, int]]


def _control_action_evidence(action: Any) -> dict[str, Any]:
    """Serialize even malformed/non-finite policy output without JSON NaN values."""

    if not isinstance(action, ControlAction):
        return {
            "value_type": type(action).__name__,
            "representation": repr(action)[:1_000],
        }
    zones: dict[str, Any] = {}
    try:
        setpoint_items = tuple(action.setpoints.items())
    except Exception as exc:
        return {
            "value_type": type(action).__name__,
            "representation": repr(action)[:1_000],
            "serialization_error": f"{type(exc).__name__}: {exc}",
        }
    for zone, setpoints in setpoint_items:
        zone_values: dict[str, float | str] = {}
        for label in ("heating_c", "cooling_c"):
            try:
                raw_value = getattr(setpoints, label)
                numeric_value = float(raw_value)
            except Exception as exc:
                zone_values[label] = f"<{type(exc).__name__}: {exc}>"
            else:
                zone_values[label] = (
                    numeric_value if math.isfinite(numeric_value) else repr(numeric_value)
                )
        zones[str(zone)] = zone_values
    return {
        "source": str(getattr(action, "source", "")),
        "reason": str(getattr(action, "reason", "")),
        "zones": zones,
    }


def validate_control_action(
    action: ControlAction,
    controlled_zones: tuple[str, ...],
    limits: SafetyLimits,
) -> ControlAction:
    """Validate one policy action before any EnergyPlus write occurs."""

    unknown = sorted(set(action.setpoints) - set(controlled_zones))
    if unknown:
        raise InvalidControlAction(f"Action targets unknown zones: {', '.join(unknown)}")

    for zone, setpoints in action.setpoints.items():
        heating = float(setpoints.heating_c)
        cooling = float(setpoints.cooling_c)
        if not math.isfinite(heating) or not math.isfinite(cooling):
            raise InvalidControlAction(f"{zone}: setpoints must be finite")
        if not limits.heating_minimum_c <= heating <= limits.heating_maximum_c:
            raise InvalidControlAction(
                f"{zone}: heating setpoint {heating} C is outside "
                f"[{limits.heating_minimum_c}, {limits.heating_maximum_c}]"
            )
        if not limits.cooling_minimum_c <= cooling <= limits.cooling_maximum_c:
            raise InvalidControlAction(
                f"{zone}: cooling setpoint {cooling} C is outside "
                f"[{limits.cooling_minimum_c}, {limits.cooling_maximum_c}]"
            )
        if cooling - heating < limits.minimum_deadband_c:
            raise InvalidControlAction(
                f"{zone}: {cooling - heating:.3f} C deadband is below "
                f"{limits.minimum_deadband_c:.3f} C"
            )
    return action


_ERROR_PATTERN = re.compile(r"\*\*\s*(Warning|Severe|Fatal)\s*\*\*", re.IGNORECASE)


def parse_energyplus_error_file(path: str | Path) -> EnergyPlusErrorSummary:
    """Count Warning/Severe/Fatal records in an EnergyPlus error file."""

    error_path = Path(path)
    if not error_path.is_file():
        return EnergyPlusErrorSummary(0, 0, 0, ())
    warning_count = 0
    severe_count = 0
    fatal_count = 0
    messages: list[str] = []
    for line in error_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _ERROR_PATTERN.search(line)
        if not match:
            continue
        severity = match.group(1).lower()
        warning_count += severity == "warning"
        severe_count += severity == "severe"
        fatal_count += severity == "fatal"
        messages.append(line.strip())
    return EnergyPlusErrorSummary(
        warning_count=warning_count,
        severe_count=severe_count,
        fatal_count=fatal_count,
        messages=tuple(messages),
    )


def _read_csv_timestep_energy_kwh(
    path: str | Path,
    candidate_fragments: tuple[str, ...],
    label: str,
) -> float:
    """Sum one EnergyPlus-native timestep energy column from eplusout.csv."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise EnergyCrosscheckError(f"EnergyPlus CSV was not produced: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        candidates = [
            name
            for name in fieldnames
            if any(fragment in name for fragment in candidate_fragments)
        ]
        if not candidates:
            raise EnergyCrosscheckError(
                f"No timestep {label} energy column exists in EnergyPlus CSV"
            )
        column = candidates[0]
        total_joules = 0.0
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                total_joules += float(value)
    return total_joules / 3_600_000.0


def read_csv_facility_electricity_kwh(path: str | Path) -> float:
    """Sum EnergyPlus-native facility electricity joules from eplusout.csv."""

    return _read_csv_timestep_energy_kwh(
        path,
        (
            "Facility Total Purchased Electricity Energy [J](TimeStep)",
            "Electricity:Facility [J](TimeStep)",
        ),
        "facility electricity",
    )


def read_csv_hvac_electricity_kwh(path: str | Path) -> float:
    """Sum EnergyPlus-native HVAC electricity joules from eplusout.csv."""

    return _read_csv_timestep_energy_kwh(
        path,
        ("Electricity:HVAC [J](TimeStep)",),
        "HVAC electricity",
    )


def read_csv_natural_gas_kwh(path: str | Path) -> float:
    """Sum EnergyPlus-native facility natural-gas joules from eplusout.csv."""

    return _read_csv_timestep_energy_kwh(
        path,
        ("NaturalGas:Facility [J](TimeStep)",),
        "facility natural gas",
    )


def _write_json(path: Path, value: Any) -> None:
    """Write one deterministic UTF-8 JSON artifact."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _create_jsonl_logger(path: Path, run_id: str) -> logging.Logger:
    """Create an isolated structured logger whose lines are valid JSON."""

    logger = logging.getLogger(f"eco_loop.phase1.{run_id}.{time.time_ns()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    """Flush and close the run's file handlers."""

    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


class EnergyPlusWrapper:
    """Own EnergyPlus state, callbacks, exchange handles, logging, and retries."""

    def __init__(
        self,
        config: Phase1Config,
        api_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self._api_factory = api_factory or self._load_api_factory()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        api_factory: Callable[[], Any] | None = None,
    ) -> "EnergyPlusWrapper":
        """Create a wrapper from a Phase 1 YAML file."""

        return cls(Phase1Config.load(config_path), api_factory=api_factory)

    def _load_api_factory(self) -> Callable[[], Any]:
        """Import the API bundled with the configured EnergyPlus distribution."""

        install_root = str(self.config.energyplus_home)
        if install_root not in sys.path:
            sys.path.insert(0, install_root)
        module = importlib.import_module("pyenergyplus.api")
        factory = getattr(module, "EnergyPlusAPI", None)
        if factory is None:
            raise ConfigurationError(
                f"EnergyPlusAPI is unavailable beneath {self.config.energyplus_home}"
            )
        return factory

    def run(
        self,
        run_id: str,
        mode: str,
        policy: Policy | None = None,
        model_path: str | Path | None = None,
    ) -> RunResult:
        """Run EnergyPlus with bounded retry for recoverable engine failures."""

        selected_model = Path(model_path).resolve() if model_path else self.config.baseline_model
        self.config.validate_paths(selected_model)
        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            output_directory = self._unique_output_directory(run_id, attempt)
            try:
                return self._run_once(
                    run_id=run_id,
                    mode=mode,
                    policy=policy,
                    model_path=selected_model,
                    output_directory=output_directory,
                    attempt=attempt,
                )
            except (Phase1Error, OSError) as exc:
                cause = exc.__cause__
                permanent = isinstance(
                    exc,
                    (
                        ConfigurationError,
                        HandleResolutionError,
                        InvalidControlAction,
                        RuntimeCleanupError,
                    ),
                ) or (
                    isinstance(exc, CallbackExecutionError)
                    and isinstance(
                        cause,
                        (
                            ConfigurationError,
                            HandleResolutionError,
                            InvalidControlAction,
                        ),
                    )
                )
                recoverable = isinstance(
                    exc,
                    (
                        EnergyPlusRunError,
                        EnergyCrosscheckError,
                        DataExchangeError,
                        CallbackExecutionError,
                        OSError,
                    ),
                )
                if permanent or not recoverable:
                    raise
                last_error = exc
                if attempt >= self.config.max_attempts:
                    raise
                LOGGER.warning(
                    json.dumps(
                        {
                            "event": "energyplus_retry",
                            "run_id": run_id,
                            "failed_attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        sort_keys=True,
                    )
                )
                time.sleep(self.config.retry_delay_seconds)
        raise EnergyPlusRunError(f"EnergyPlus failed without an exception: {last_error}")

    def _unique_output_directory(self, run_id: str, attempt: int) -> Path:
        """Create a new output directory without overwriting prior evidence."""

        self.config.output_root.mkdir(parents=True, exist_ok=True)
        suffix = "" if attempt == 1 else f"-retry-{attempt}"
        candidate = self.config.output_root / f"{run_id}{suffix}"
        if candidate.exists():
            counter = 2
            while (self.config.output_root / f"{run_id}{suffix}-{counter}").exists():
                counter += 1
            candidate = self.config.output_root / f"{run_id}{suffix}-{counter}"
        candidate.mkdir(parents=False)
        return candidate.resolve()

    def _run_once(
        self,
        run_id: str,
        mode: str,
        policy: Policy | None,
        model_path: Path,
        output_directory: Path,
        attempt: int,
    ) -> RunResult:
        """Execute one EnergyPlus state from initialization through cleanup."""

        attempt_started = time.perf_counter()

        def write_pre_runtime_failure(
            stage: str,
            exc: BaseException,
            cleanup_failures: tuple[str, ...] = (),
        ) -> None:
            """Write diagnostics for failures that occur before callbacks run."""

            try:
                error_summary = parse_energyplus_error_file(
                    output_directory / "eplusout.err"
                )
            except Exception:
                error_summary = EnergyPlusErrorSummary(0, 0, 0, ())
            payload = {
                "status": "failed",
                "run_id": run_id,
                "mode": mode,
                "model_path": str(model_path),
                "output_directory": str(output_directory),
                "attempt": attempt,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exit_code": None,
                "elapsed_seconds": time.perf_counter() - attempt_started,
                "timestep_count": 0,
                "energyplus_errors": asdict(error_summary),
                "simulator_error_callbacks": [],
                "callback_failures": [],
                "cleanup_errors": list(cleanup_failures),
            }
            try:
                _write_json(output_directory / self.config.summary_name, payload)
                _write_json(
                    output_directory / "simulator_error_callbacks.json",
                    [],
                )
            except Exception as artifact_error:
                LOGGER.error(
                    json.dumps(
                        {
                            "event": "failure_artifact_write_failed",
                            "run_id": run_id,
                            "stage": stage,
                            "error_type": type(artifact_error).__name__,
                            "error": str(artifact_error),
                        },
                        sort_keys=True,
                    )
                )

        try:
            api = self._api_factory()
            state = api.state_manager.new_state()
        except Exception as exc:
            write_pre_runtime_failure("api_initialization", exc)
            if isinstance(exc, (Phase1Error, OSError)):
                raise
            raise EnergyPlusRunError(
                f"EnergyPlus API/state initialization failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        logger: logging.Logger | None = None
        cleanup_complete = False
        cleanup_errors: list[str] = []

        def cleanup_runtime() -> None:
            """Release global callback references, state, and log handlers once."""

            nonlocal cleanup_complete
            if cleanup_complete:
                return
            cleanup_complete = True
            cleanup_steps: tuple[tuple[str, Callable[[], None]], ...] = (
                ("runtime.clear_callbacks", api.runtime.clear_callbacks),
                ("functional.clear_callbacks", api.functional.clear_callbacks),
                ("state_manager.delete_state", lambda: api.state_manager.delete_state(state)),
            )
            for label, operation in cleanup_steps:
                try:
                    operation()
                except BaseException as exc:
                    cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            if logger is not None:
                try:
                    _close_logger(logger)
                except BaseException as exc:
                    cleanup_errors.append(
                        f"close_timestep_logger: {type(exc).__name__}: {exc}"
                    )
            if cleanup_errors:
                try:
                    _write_json(
                        output_directory / "cleanup_errors.json",
                        {"errors": cleanup_errors},
                    )
                except BaseException:
                    pass

        try:
            api.verify_api_version_match(state)
            api.runtime.set_console_output_status(state, False)
            logger = _create_jsonl_logger(
                output_directory / self.config.timestep_log_name,
                run_id,
            )
        except Exception as exc:
            cleanup_runtime()
            write_pre_runtime_failure("api_setup", exc, tuple(cleanup_errors))
            if cleanup_errors:
                raise RuntimeCleanupError(
                    "EnergyPlus cleanup failed after API setup error: "
                    + "; ".join(cleanup_errors)
                ) from exc
            raise
        assert logger is not None

        callback_failure: list[str] = []
        callback_exceptions: list[BaseException] = []
        simulator_errors: list[dict[str, Any]] = []
        handles: _HandleRegistry | None = None
        latest_action: ControlAction | None = None
        latest_action_sequence = 0
        last_applied: dict[str, Any] = {"status": "not_yet_applied"}
        last_timestep_identity: tuple[int, int, int, int, int, int] | None = None
        timestep_count = 0
        facility_electricity_j = 0.0
        zone_buffer_facility_electricity_j = 0.0
        zone_buffer_system_callbacks = 0
        last_system_timestep_identity: tuple[int, int, int, int, int, int, float] | None = (
            None
        )
        first_snapshot: dict[str, Any] | None = None
        last_snapshot: dict[str, Any] | None = None

        try:
            requested_variables = self._requested_variables()
            for variable_name, variable_key in requested_variables:
                api.exchange.reset_api_error_flag(state)
                api.exchange.request_variable(state, variable_name, variable_key)
                if api.exchange.api_error_flag(state):
                    raise DataExchangeError(
                        "EnergyPlus rejected variable request: "
                        f"{variable_name} / {variable_key}"
                    )
        except Exception as exc:
            cleanup_runtime()
            write_pre_runtime_failure(
                "variable_request",
                exc,
                tuple(cleanup_errors),
            )
            if cleanup_errors:
                raise RuntimeCleanupError(
                    "EnergyPlus cleanup failed after variable-request error: "
                    + "; ".join(cleanup_errors)
                ) from exc
            raise

        def record_callback_failure(callback_name: str, exc: BaseException) -> None:
            if callback_failure:
                return
            failure = (
                f"{callback_name}: {exc}\n"
                f"{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )
            callback_failure.append(failure)
            callback_exceptions.append(exc)
            try:
                api.runtime.issue_severe(
                    state,
                    f"Eco-Loop callback failure in {callback_name}: {exc}",
                )
                api.runtime.stop_simulation(state)
            except Exception:
                callback_failure.append(traceback.format_exc())

        def guard(
            callback_name: str,
            callback: Callable[[Any], None],
        ) -> Callable[[Any], None]:
            def guarded(callback_state: Any) -> None:
                if callback_failure:
                    return
                try:
                    callback(callback_state)
                except BaseException as exc:
                    record_callback_failure(callback_name, exc)

            return guarded

        def on_error(severity: int, raw_message: bytes) -> None:
            message = raw_message.decode("utf-8", errors="replace").strip()
            simulator_errors.append({"severity": int(severity), "message": message})

        def ensure_handles(callback_state: Any) -> _HandleRegistry:
            nonlocal handles
            if handles is not None:
                return handles
            if not api.exchange.api_data_fully_ready(callback_state):
                raise HandleResolutionError("EnergyPlus API data is not fully ready")
            points = [
                {
                    "what": str(point.what),
                    "name": str(point.name),
                    "key": str(point.key),
                    "type": str(point.type),
                    "unit": str(point.unit),
                }
                for point in api.exchange.get_api_data(callback_state)
            ]
            points.sort(
                key=lambda point: (
                    point["what"],
                    point["name"],
                    point["key"],
                    point["type"],
                )
            )
            _write_json(output_directory / "exchange_points.json", points)

            variable_handles: dict[str, int] = {}
            missing: list[str] = []
            for label, variable_name, variable_key in self._variable_specs():
                handle = api.exchange.get_variable_handle(
                    callback_state,
                    variable_name,
                    variable_key,
                )
                variable_handles[label] = int(handle)
                if handle == -1:
                    missing.append(f"variable {variable_name} / {variable_key}")

            meter_handles: dict[str, int] = {}
            for label, meter_name in self._meter_specs():
                handle = api.exchange.get_meter_handle(callback_state, meter_name)
                meter_handles[label] = int(handle)
                if handle == -1:
                    missing.append(f"meter {meter_name}")

            actuator_handles: dict[str, dict[str, int]] = {}
            for zone in self.config.controlled_zones:
                heating_handle = api.exchange.get_actuator_handle(
                    callback_state,
                    "Zone Temperature Control",
                    "Heating Setpoint",
                    zone,
                )
                cooling_handle = api.exchange.get_actuator_handle(
                    callback_state,
                    "Zone Temperature Control",
                    "Cooling Setpoint",
                    zone,
                )
                actuator_handles[zone] = {
                    "heating": int(heating_handle),
                    "cooling": int(cooling_handle),
                }
                if heating_handle == -1:
                    missing.append(f"actuator heating setpoint / {zone}")
                if cooling_handle == -1:
                    missing.append(f"actuator cooling setpoint / {zone}")

            facility_meter_handle = api.exchange.get_meter_handle(
                callback_state,
                "Electricity:Facility",
            )
            inventory = {
                "variables": variable_handles,
                "meters": meter_handles,
                "actuators": actuator_handles,
                "known_v26_1_facility_meter_handle": int(facility_meter_handle),
                "facility_energy_source": (
                    "Facility Total Purchased Electricity Energy / Whole Building"
                ),
                "meter_note": (
                    "v26.1 live meter handles are inventoried only; verified run totals "
                    "come from EnergyPlus CSV while the live snapshot uses the facility "
                    "energy output variable"
                ),
            }
            _write_json(output_directory / "resolved_handles.json", inventory)
            if missing:
                raise HandleResolutionError(
                    "Required EnergyPlus exchange handles are missing:\n"
                    + "\n".join(missing)
                )
            handles = _HandleRegistry(
                variables=variable_handles,
                meters=meter_handles,
                actuators=actuator_handles,
            )
            return handles

        def read_variable(
            callback_state: Any,
            registry: _HandleRegistry,
            label: str,
        ) -> float:
            return self._read_exchange_value(
                api.exchange,
                callback_state,
                api.exchange.get_variable_value,
                registry.variables[label],
                label,
            )

        def on_begin_timestep(callback_state: Any) -> None:
            nonlocal last_applied
            if not api.exchange.api_data_fully_ready(callback_state):
                return
            if api.exchange.kind_of_sim(callback_state) != 3:
                return
            if api.exchange.warmup_flag(callback_state):
                return
            registry = ensure_handles(callback_state)
            action = latest_action
            if action is not None:
                validate_control_action(
                    action,
                    self.config.controlled_zones,
                    self.config.safety,
                )

            applied: dict[str, Any] = {
                "status": "applied" if action is not None else "reset",
                "selected_snapshot_sequence": latest_action_sequence,
                "simulation_time_hours": (
                    latest_action_sequence * self.config.timestep_minutes / 60.0
                ),
                "zones": {},
            }
            for zone in self.config.controlled_zones:
                zone_handles = registry.actuators[zone]
                setpoints = action.setpoints.get(zone) if action is not None else None
                api.exchange.reset_api_error_flag(callback_state)
                if setpoints is None:
                    api.exchange.reset_actuator(
                        callback_state,
                        zone_handles["heating"],
                    )
                    api.exchange.reset_actuator(
                        callback_state,
                        zone_handles["cooling"],
                    )
                    applied["zones"][zone] = {"status": "reset"}
                else:
                    api.exchange.set_actuator_value(
                        callback_state,
                        zone_handles["heating"],
                        float(setpoints.heating_c),
                    )
                    api.exchange.set_actuator_value(
                        callback_state,
                        zone_handles["cooling"],
                        float(setpoints.cooling_c),
                    )
                    applied["zones"][zone] = {
                        "status": "applied",
                        "heating_c": float(setpoints.heating_c),
                        "cooling_c": float(setpoints.cooling_c),
                    }
                if api.exchange.api_error_flag(callback_state):
                    raise DataExchangeError(f"Actuator write failed for {zone}")
            if action is not None:
                applied["source"] = action.source
                applied["reason"] = action.reason
            last_applied = applied

        def on_end_system_timestep(callback_state: Any) -> None:
            nonlocal zone_buffer_facility_electricity_j
            nonlocal zone_buffer_system_callbacks
            nonlocal last_system_timestep_identity

            if not api.exchange.api_data_fully_ready(callback_state):
                return
            if api.exchange.kind_of_sim(callback_state) != 3:
                return
            if api.exchange.warmup_flag(callback_state):
                return
            registry = ensure_handles(callback_state)
            identity = (
                int(api.exchange.current_environment_num(callback_state)),
                int(api.exchange.calendar_year(callback_state)),
                int(api.exchange.month(callback_state)),
                int(api.exchange.day_of_month(callback_state)),
                int(api.exchange.hour(callback_state)),
                int(api.exchange.minutes(callback_state)),
                round(float(api.exchange.system_time_step(callback_state)), 9),
            )
            if identity == last_system_timestep_identity:
                return
            last_system_timestep_identity = identity
            zone_buffer_facility_electricity_j += read_variable(
                callback_state,
                registry,
                "facility_electricity",
            )
            zone_buffer_system_callbacks += 1

        def on_end_timestep(callback_state: Any) -> None:
            nonlocal latest_action
            nonlocal latest_action_sequence
            nonlocal last_timestep_identity
            nonlocal timestep_count
            nonlocal facility_electricity_j
            nonlocal zone_buffer_facility_electricity_j
            nonlocal zone_buffer_system_callbacks
            nonlocal first_snapshot
            nonlocal last_snapshot

            if not api.exchange.api_data_fully_ready(callback_state):
                return
            if api.exchange.kind_of_sim(callback_state) != 3:
                return
            if api.exchange.warmup_flag(callback_state):
                return
            registry = ensure_handles(callback_state)
            environment_number = int(
                api.exchange.current_environment_num(callback_state)
            )
            zone_timestep_number = int(
                api.exchange.zone_time_step_number(callback_state)
            )
            identity = (
                environment_number,
                int(api.exchange.calendar_year(callback_state)),
                int(api.exchange.month(callback_state)),
                int(api.exchange.day_of_month(callback_state)),
                int(api.exchange.hour(callback_state)),
                zone_timestep_number,
            )
            if identity == last_timestep_identity:
                return
            last_timestep_identity = identity

            timestep_count += 1
            simulation_time = (
                timestep_count * self.config.timestep_minutes / 60.0
            )
            zone_data: dict[str, ZoneSensorData] = {}
            for zone in self.config.controlled_zones:
                zone_data[zone] = ZoneSensorData(
                    air_temperature_c=read_variable(
                        callback_state,
                        registry,
                        f"{zone}.temperature",
                    ),
                    relative_humidity_pct=read_variable(
                        callback_state,
                        registry,
                        f"{zone}.relative_humidity",
                    ),
                    co2_ppm=read_variable(
                        callback_state,
                        registry,
                        f"{zone}.co2",
                    ),
                    occupant_count=read_variable(
                        callback_state,
                        registry,
                        f"{zone}.occupancy",
                    ),
                    fanger_pmv=read_variable(
                        callback_state,
                        registry,
                        f"{zone}.pmv",
                    ),
                    heating_setpoint_c=read_variable(
                        callback_state,
                        registry,
                        f"{zone}.heating_setpoint",
                    ),
                    cooling_setpoint_c=read_variable(
                        callback_state,
                        registry,
                        f"{zone}.cooling_setpoint",
                    ),
                )

            if zone_buffer_system_callbacks:
                timestep_facility_j = zone_buffer_facility_electricity_j
            else:
                timestep_facility_j = read_variable(
                    callback_state,
                    registry,
                    "facility_electricity",
                )
            zone_buffer_facility_electricity_j = 0.0
            zone_buffer_system_callbacks = 0
            facility_electricity_j += timestep_facility_j

            snapshot = SensorSnapshot(
                sequence=timestep_count,
                environment_number=environment_number,
                simulation_time_hours=simulation_time,
                calendar_year=int(api.exchange.calendar_year(callback_state)),
                month=int(api.exchange.month(callback_state)),
                day_of_month=int(api.exchange.day_of_month(callback_state)),
                hour=int(api.exchange.hour(callback_state)),
                minute=zone_timestep_number * self.config.timestep_minutes,
                zone_timestep_number=zone_timestep_number,
                outdoor_drybulb_c=read_variable(
                    callback_state,
                    registry,
                    "outdoor_drybulb",
                ),
                facility_electricity_j=timestep_facility_j,
                facility_electricity_demand_w=read_variable(
                    callback_state,
                    registry,
                    "facility_demand",
                ),
                zones=zone_data,
            )
            snapshot_dict = asdict(snapshot)
            proposed: ControlAction | None = None
            proposal_evidence: dict[str, Any] = {"status": "no_action"}
            proposal_error: BaseException | None = None
            try:
                proposed = policy(snapshot) if policy is not None else None
                if proposed is not None:
                    proposal_evidence = {
                        "status": "accepted",
                        **_control_action_evidence(proposed),
                    }
                    validate_control_action(
                        proposed,
                        self.config.controlled_zones,
                        self.config.safety,
                    )
            except BaseException as exc:
                proposal_error = exc
                proposal_evidence = {
                    "status": "rejected",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "proposed": _control_action_evidence(proposed),
                }
            record = {
                "event": "zone_timestep",
                "run_id": run_id,
                "mode": mode,
                "timestamp": snapshot.energyplus_timestamp,
                "snapshot": snapshot_dict,
                "action_applied": last_applied,
                "action_selected_for_next_timestep": proposal_evidence,
            }
            logger.info(json.dumps(record, sort_keys=True, allow_nan=False))
            if first_snapshot is None:
                first_snapshot = snapshot_dict
            last_snapshot = snapshot_dict
            if proposal_error is not None:
                latest_action = None
                raise proposal_error
            latest_action = proposed
            latest_action_sequence = timestep_count

        started = time.perf_counter()
        exit_code = -1
        execution_exception: BaseException | None = None
        try:
            api.functional.callback_error(state, on_error)
            api.runtime.callback_begin_system_timestep_before_predictor(
                state,
                guard("begin_system_timestep_before_predictor", on_begin_timestep),
            )
            api.runtime.callback_end_system_timestep_after_hvac_reporting(
                state,
                guard(
                    "end_system_timestep_after_hvac_reporting",
                    on_end_system_timestep,
                ),
            )
            api.runtime.callback_end_zone_timestep_after_zone_reporting(
                state,
                guard("end_zone_timestep_after_zone_reporting", on_end_timestep),
            )

            arguments = [
                "-w",
                str(self.config.weather_file),
                "-r",
                "-d",
                str(output_directory),
                str(model_path),
            ]
            exit_code = int(api.runtime.run_energyplus(state, arguments))
        except Exception as exc:
            execution_exception = exc
        finally:
            elapsed_seconds = time.perf_counter() - started
            cleanup_runtime()

        error_summary = parse_energyplus_error_file(output_directory / "eplusout.err")

        def write_failure_artifacts(
            error_type: str,
            error_message: str,
            stage: str,
        ) -> None:
            """Best-effort structured diagnostics without masking the root failure."""

            payload = {
                "status": "failed",
                "run_id": run_id,
                "mode": mode,
                "model_path": str(model_path),
                "output_directory": str(output_directory),
                "attempt": attempt,
                "stage": stage,
                "error_type": error_type,
                "error": error_message,
                "exit_code": exit_code,
                "elapsed_seconds": elapsed_seconds,
                "timestep_count": timestep_count,
                "energyplus_errors": asdict(error_summary),
                "simulator_error_callbacks": simulator_errors,
                "callback_failures": callback_failure,
                "cleanup_errors": cleanup_errors,
            }
            try:
                _write_json(output_directory / self.config.summary_name, payload)
                _write_json(
                    output_directory / "simulator_error_callbacks.json",
                    simulator_errors,
                )
            except Exception as artifact_error:
                LOGGER.error(
                    json.dumps(
                        {
                            "event": "failure_artifact_write_failed",
                            "run_id": run_id,
                            "error_type": type(artifact_error).__name__,
                            "error": str(artifact_error),
                        },
                        sort_keys=True,
                    )
                )

        if cleanup_errors:
            primary_cause: BaseException | None = None
            primary_detail = ""
            if callback_failure:
                try:
                    _write_json(
                        output_directory / "callback_failure.json",
                        {"failures": callback_failure},
                    )
                except Exception:
                    pass
                primary_cause = (
                    callback_exceptions[0] if callback_exceptions else None
                )
                primary_detail = f"; preceding callback failure: {callback_failure[0]}"
            elif execution_exception is not None:
                primary_cause = execution_exception
                primary_detail = (
                    f"; preceding runtime failure: "
                    f"{type(execution_exception).__name__}: {execution_exception}"
                )
            message = (
                "EnergyPlus cleanup failed: "
                + "; ".join(cleanup_errors)
                + primary_detail
            )
            write_failure_artifacts(
                "RuntimeCleanupError",
                message,
                "runtime_cleanup",
            )
            cleanup_error = RuntimeCleanupError(message)
            if primary_cause is not None:
                raise cleanup_error from primary_cause
            raise cleanup_error

        if callback_failure:
            _write_json(
                output_directory / "callback_failure.json",
                {"failures": callback_failure},
            )
            write_failure_artifacts(
                "CallbackExecutionError",
                callback_failure[0],
                "runtime_callback",
            )
            cause = callback_exceptions[0] if callback_exceptions else None
            raise CallbackExecutionError(callback_failure[0]) from cause

        if execution_exception is not None:
            write_failure_artifacts(
                type(execution_exception).__name__,
                str(execution_exception),
                "runtime_execution",
            )
            if isinstance(execution_exception, (Phase1Error, OSError)):
                raise execution_exception
            raise EnergyPlusRunError(
                f"EnergyPlus runtime call failed for {run_id!r}: "
                f"{type(execution_exception).__name__}: {execution_exception}"
            ) from execution_exception

        if exit_code != 0 or error_summary.severe_count or error_summary.fatal_count:
            error = EnergyPlusRunError(
                f"EnergyPlus run {run_id!r} failed with exit={exit_code}, "
                f"severe={error_summary.severe_count}, fatal={error_summary.fatal_count}; "
                f"outputs: {output_directory}"
            )
            write_failure_artifacts(
                type(error).__name__,
                str(error),
                "energyplus_result",
            )
            raise error
        if timestep_count == 0:
            error = EnergyPlusRunError(
                f"EnergyPlus run {run_id!r} produced no weather-run timesteps"
            )
            write_failure_artifacts(
                type(error).__name__,
                str(error),
                "energyplus_result",
            )
            raise error

        api_facility_kwh = facility_electricity_j / 3_600_000.0
        try:
            csv_facility_kwh = read_csv_facility_electricity_kwh(
                output_directory / "eplusout.csv"
            )
            csv_hvac_kwh = read_csv_hvac_electricity_kwh(
                output_directory / "eplusout.csv"
            )
            csv_natural_gas_total_kwh = read_csv_natural_gas_kwh(
                output_directory / "eplusout.csv"
            )
        except (EnergyCrosscheckError, OSError, ValueError) as exc:
            error = (
                exc
                if isinstance(exc, (EnergyCrosscheckError, OSError))
                else EnergyCrosscheckError(
                    f"EnergyPlus CSV contains invalid numeric energy data: {exc}"
                )
            )
            write_failure_artifacts(
                type(error).__name__,
                str(error),
                "energy_crosscheck",
            )
            if error is exc:
                raise error
            raise error from exc
        denominator = max(abs(csv_facility_kwh), 1.0e-12)
        relative_error = abs(api_facility_kwh - csv_facility_kwh) / denominator
        if relative_error > 0.005:
            error = EnergyCrosscheckError(
                f"API facility electricity {api_facility_kwh:.6f} kWh differs from "
                f"EnergyPlus CSV {csv_facility_kwh:.6f} kWh by "
                f"{relative_error:.3%}"
            )
            write_failure_artifacts(
                type(error).__name__,
                str(error),
                "energy_crosscheck",
            )
            raise error
        result = RunResult(
            run_id=run_id,
            mode=mode,
            model_path=str(model_path),
            output_directory=str(output_directory),
            attempt=attempt,
            exit_code=exit_code,
            timestep_count=timestep_count,
            facility_electricity_kwh=api_facility_kwh,
            hvac_electricity_kwh=csv_hvac_kwh,
            natural_gas_kwh=csv_natural_gas_total_kwh,
            csv_facility_electricity_kwh=csv_facility_kwh,
            energy_crosscheck_relative_error=relative_error,
            warning_count=error_summary.warning_count,
            severe_count=error_summary.severe_count,
            fatal_count=error_summary.fatal_count,
            simulator_error_callback_count=sum(
                int(event["severity"]) >= 2 for event in simulator_errors
            ),
            elapsed_seconds=elapsed_seconds,
            first_snapshot=first_snapshot,
            last_snapshot=last_snapshot,
        )
        _write_json(output_directory / self.config.summary_name, result.to_dict())
        _write_json(output_directory / "simulator_error_callbacks.json", simulator_errors)
        return result

    def _requested_variables(self) -> tuple[tuple[str, str], ...]:
        """Return all variable requests required before every EnergyPlus run."""

        requests: list[tuple[str, str]] = [
            ("Site Outdoor Air Drybulb Temperature", "Environment"),
            ("Facility Total Purchased Electricity Energy", "Whole Building"),
            ("Facility Total Electricity Demand Rate", "Whole Building"),
        ]
        for zone in self.config.controlled_zones:
            requests.extend(
                [
                    ("Zone Mean Air Temperature", zone),
                    ("Zone Air Relative Humidity", zone),
                    ("Zone Air CO2 Concentration", zone),
                    ("Zone People Occupant Count", zone),
                    ("Zone Thermostat Heating Setpoint Temperature", zone),
                    ("Zone Thermostat Cooling Setpoint Temperature", zone),
                    (
                        "Zone Thermal Comfort Fanger Model PMV",
                        self.config.people_objects[zone],
                    ),
                ]
            )
        return tuple(requests)

    def _variable_specs(self) -> tuple[tuple[str, str, str], ...]:
        """Map stable wrapper labels to EnergyPlus variable name/key pairs."""

        specs: list[tuple[str, str, str]] = [
            (
                "outdoor_drybulb",
                "Site Outdoor Air Drybulb Temperature",
                "Environment",
            ),
            (
                "facility_electricity",
                "Facility Total Purchased Electricity Energy",
                "Whole Building",
            ),
            (
                "facility_demand",
                "Facility Total Electricity Demand Rate",
                "Whole Building",
            ),
        ]
        for zone in self.config.controlled_zones:
            specs.extend(
                [
                    (f"{zone}.temperature", "Zone Mean Air Temperature", zone),
                    (
                        f"{zone}.relative_humidity",
                        "Zone Air Relative Humidity",
                        zone,
                    ),
                    (f"{zone}.co2", "Zone Air CO2 Concentration", zone),
                    (f"{zone}.occupancy", "Zone People Occupant Count", zone),
                    (
                        f"{zone}.heating_setpoint",
                        "Zone Thermostat Heating Setpoint Temperature",
                        zone,
                    ),
                    (
                        f"{zone}.cooling_setpoint",
                        "Zone Thermostat Cooling Setpoint Temperature",
                        zone,
                    ),
                    (
                        f"{zone}.pmv",
                        "Zone Thermal Comfort Fanger Model PMV",
                        self.config.people_objects[zone],
                    ),
                ]
            )
        return tuple(specs)

    @staticmethod
    def _meter_specs() -> tuple[tuple[str, str], ...]:
        """Return required non-index-zero EnergyPlus meter handles."""

        return (
            ("hvac_electricity", "Electricity:HVAC"),
            ("natural_gas", "NaturalGas:Facility"),
        )

    @staticmethod
    def _read_exchange_value(
        exchange: Any,
        state: Any,
        getter: Callable[[Any, int], float],
        handle: int,
        label: str,
    ) -> float:
        """Read one exchange value while disambiguating legitimate zero values."""

        exchange.reset_api_error_flag(state)
        value = float(getter(state, handle))
        if exchange.api_error_flag(state):
            raise DataExchangeError(f"EnergyPlus API error while reading {label}")
        if not math.isfinite(value):
            raise DataExchangeError(f"EnergyPlus returned non-finite data for {label}")
        return value
