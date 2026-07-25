# Current Results

This page separates deterministic Phase 1 evidence, deterministic
ScriptedProvider live-loop evidence, and pending real-Ollama evidence. Phase 1
and ScriptedProvider results are **not real-LLM-generated savings**.

## Phase 1 deterministic seven-day comparison

- Baseline electricity: **1024.291 kWh**
- Controlled electricity: **955.179 kWh**
- Deterministic reduction: **69.112 kWh (6.747%)**
- Duration: **168.0 hours**, **672 timesteps**
- Baseline zone temperature range: **20.63 to 30.40 °C**
- Controlled zone temperature range: **20.63 to 30.41 °C**
- Baseline PMV range: **-1.41 to 1.42**
- Controlled PMV range: **-1.41 to 1.42**
- Occupied PMV comfort violations: baseline **24**, controlled **28**
- Deterministic actions accepted/rejected: **220/0**
- Successful zone actuator writebacks: **1100**

![Deterministic Phase 1 energy comparison](../runs/final/energy_comparison.png)

## Live ScriptedProvider smoke

- Evidence type: **deterministic ScriptedProvider, not a real LLM**
- Facility electricity during smoke: **1.900 kWh**
- Energy savings: **not calculated** because there is no matched four-hour baseline
- Duration: **4.0 hours**, **16 timesteps**
- Zone temperature range: **20.67 to 21.55 °C**
- PMV range: **-1.39 to -1.11**
- Occupied PMV comfort violations: **0**
- Actions accepted/rejected: **4/0**
- Corrections/fallbacks: **0/0**
- MCP tools: **read_sensor_data, get_grid_carbon_intensity, log_reasoning, set_control_action**
- MCP tool calls: **16**
- Successful zone actuator writebacks: **60**

![Zone temperature and PMV ranges](../runs/final/zone_temperature_pmv.png)

![Action and setpoint evidence](../runs/final/action_setpoint_evidence.png)

## Real Ollama evidence

- Status: **pending**
- Result: No completed live-Ollama smoke report is available. No real-LLM result or savings claim is reported.

No real-Ollama energy or control claim is made while this section is pending.
