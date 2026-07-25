# Historical Phase 1 handoff: EnergyPlus + EMS wiring

Phase 1 status: complete. This report was written at the Phase 1 checkpoint,
before the now-completed Phase 2 MCP and Ollama work. It is retained as the
original deterministic EnergyPlus evidence record.

## Acceptance evidence

The final evidence run used EnergyPlus 26.1.0, the Chicago O'Hare TMY3 weather
file, five controlled zones, a seven-day period (July 21–27), and 15-minute
zone timesteps.

| Gate | Result |
| --- | ---: |
| Timesteps per run | 672 |
| EnergyPlus warnings / severe / fatal | 0 / 0 / 0 |
| Controlled zones | 5 |
| Occupied timesteps with applied action | 220 |
| Setpoint readback comparisons | 2,200 |
| Maximum setpoint readback error | 0.000 °C |
| Maximum baseline-to-actuated zone-temperature delta | 2.100 °C |
| Baseline facility electricity | 1,024.291 kWh |
| Actuated facility electricity | 955.179 kWh |
| Diagnostic-case electricity delta | -69.112 kWh (-6.747%) |
| Occupied PMV samples within \|PMV\| ≤ 0.7 | 97.455% |
| Maximum occupied CO2 signal | 1,777.888 ppm |
| Repeatability electricity delta | 0.000000000 kWh |

The energy delta is evidence that the commanded setpoints changed EnergyPlus
physics; it is not a forecast of savings from the future agent policy. The
diagnostic policy does not yet control ventilation in response to CO2.

Machine-readable evidence is in `runs/phase1/phase1_report.json`. The final
successful outputs referenced by that report are:

- `runs/phase1/baseline-8`
- `runs/phase1/actuated-5`
- `runs/phase1/actuated-repeat-4`

## Sensors and actuators

Each zone exposes:

- Mean air temperature
- Relative humidity
- CO2 concentration
- Occupant count
- Fanger thermal-comfort PMV
- Heating and cooling thermostat setpoints

The wrapper also captures outdoor dry-bulb temperature, facility electric
demand, and facility electricity. It resolves one heating and one cooling
`Zone Temperature Control` actuator for each zone and applies validated values
at `BeginSystemTimestepBeforePredictor`.

EnergyPlus 26.1 on this Windows build returns an invalid handle for the
`Electricity:Facility` alias because the matching meter occupies exchange
index zero. The implementation therefore reads the equivalent
`Facility Total Purchased Electricity Energy` meter and cross-checks its
accumulated value against the EnergyPlus CSV total. The final runs match
exactly (relative error 0.0). HVAC-electricity and natural-gas totals are
reported from the generated CSV because their live API meter values were not
consistent enough to use as acceptance evidence.

## Reproducibility

Pinned input hashes:

| Input | SHA-256 |
| --- | --- |
| `models/upstream/5ZoneAirCooled-v26.1.0.idf` | `0187CF7F2CA9C27C43D435A68A8C66A557A43678846813A7E21463A0B0C716CD` |
| `models/baseline.idf` | `4A18BAA3FCDAB2A3968D3AD46FEAF5AED63DF93D4ED1FB2F4610D9827F3E36D2` |
| `models/optimized_runtime.idf` | `4A18BAA3FCDAB2A3968D3AD46FEAF5AED63DF93D4ED1FB2F4610D9827F3E36D2` |
| `weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` | `C7D4EFCF93BA316A1D874352E743DF5CF137BA5C0E3459EB2DC4B5442D5B7F5C` |

The baseline and runtime IDFs are intentionally byte-identical at rest:
actuation is performed through live EnergyPlus handles rather than by
rewriting model objects.

## Verification

- `python -m py_compile ...`: passed
- `python -m scripts.run_phase1_smoke --mode both`: three successful runs
- `RUN_ENERGYPLUS_INTEGRATION=1` full test discovery: 21/21 passed, including
  a real-engine missing-handle failure test
- Independent code and artifact audits: no remaining blocker, high, or medium
  findings

## Items deferred at the Phase 1 checkpoint

- MCP server and tool schemas
- LLM/provider integration
- Agent-generated control decisions
- CO2-aware ventilation or IAQ remediation
- Persistent closed-loop orchestration outside the deterministic proof runner

The first three items and the live closed-loop orchestration were subsequently
implemented in Phase 2. CO2-aware ventilation remains out of scope.
