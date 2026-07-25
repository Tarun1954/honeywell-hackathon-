# Eco-Loop Building Agents

Phase 1 is a standalone EnergyPlus 26.1 integration proof. It runs a
five-zone reference building, reads zone and whole-building telemetry through
the EnergyPlus Python Data Exchange API, and writes thermostat setpoints
through EnergyPlus EMS actuators at runtime.

Phase 2 (MCP and agent integration) has intentionally not been started.

## Phase 1 quick start

Prerequisites:

- Python 3.12 or newer
- EnergyPlus 26.1.0 for Windows

From the repository root:

```powershell
Copy-Item .env.example .env
# Set ENERGYPLUS_HOME in .env to the EnergyPlus 26.1.0 installation directory.
python -m pip install -e .
python -m scripts.run_phase1_smoke --mode both
```

The default run executes three seven-day, 15-minute simulations:

1. An unmodified baseline.
2. A deterministic occupied-hours setpoint actuation case.
3. A repeat of the actuation case to prove deterministic execution.

The consolidated result is written to
`runs/phase1/phase1_report.json`. Each simulation also gets a unique output
directory containing:

- `timesteps.jsonl`: structured sensor, action, and setpoint-readback evidence
- `summary.json`: parsed run and energy totals
- `resolved_handles.json`: resolved sensor and actuator handles
- `exchange_points.json`: the EnergyPlus exchange-point inventory
- Standard EnergyPlus outputs, including `eplusout.err`, CSV, and SQLite files

Run the automated checks with:

```powershell
python -m py_compile src/energyplus_wrapper.py scripts/run_phase1_smoke.py tests/test_energyplus_wrapper.py
python -m unittest discover -s tests -v
$env:RUN_ENERGYPLUS_INTEGRATION = "1"
python -m unittest tests.test_energyplus_wrapper.RealEngineNegativeTests -v
```

## Runtime wiring

`src/energyplus_wrapper.py` owns the EnergyPlus state and callback lifecycle.
Before each run it requests every variable, then waits for the exchange API to
report that data is ready before resolving handles.

The control path is:

```text
end of zone timestep
  -> read temperature, RH, CO2, occupancy, PMV, setpoints, and energy
  -> invoke the supplied policy
  -> validate finite values, zone names, bounds, and deadband
  -> begin of next system timestep before predictor
  -> set or reset heating and cooling EMS actuators
  -> read reported thermostat setpoints to verify writeback
```

Only weather-run timesteps are logged or controlled; sizing and warmup periods
are excluded. Callback failures stop the simulator and are re-raised after
EnergyPlus returns. Engine failures have a bounded retry, while invalid
controls, missing handles, and cleanup failures fail immediately with
diagnostic artifacts. Each JSONL record correlates the action selected from a
completed snapshot with the action applied during the following timestep.

The Phase 1 diagnostic policy is deliberately simple and deterministic. When
any configured zone is occupied, it commands 20 °C heating and 26 °C cooling
setpoints in all five zones. When the building is empty, it releases the
actuators back to the IDF schedules. It exists to prove the sensor-to-actuator
loop and is not the Phase 2 optimization policy.

## Project layout

- `models/baseline.idf`: frozen, instrumented five-zone model
- `models/optimized_runtime.idf`: runtime copy used by the actuated run
- `weather/`: Chicago O'Hare TMY3 weather input
- `config/phase1.yaml`: paths, zones, setpoint safety limits, and retry policy
- `src/energyplus_wrapper.py`: typed EnergyPlus integration layer
- `scripts/run_phase1_smoke.py`: baseline/actuated/repeatability evidence runner
- `tests/test_energyplus_wrapper.py`: validation and real-engine negative tests
- `docs/phase1_report.md`: Phase 1 evidence and known limitations
