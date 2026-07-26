# Eco-Loop Building Agents

Eco-Loop is a closed-loop building-control proof of concept that connects a
live EnergyPlus simulation to a tool-using local language model. It was built
to test a practical question: can an agent reduce building electricity and
peak demand while every proposed thermostat change remains observable,
validated, bounded, and recoverable?

The project does not let the model write directly to EnergyPlus. Live
telemetry is converted to a typed `SensorSnapshot`, exposed through a
Model Context Protocol (MCP) server, and consumed by an agent that must follow
a fixed tool sequence. A separate safety layer accepts or rejects the final
five-zone command. Only accepted actions reach the existing EnergyPlus EMS
actuators; provider failure releases those actuators back to the original
EnergyPlus schedules.

## Live dashboard

The verified, read-only results dashboard is deployed at:

**[Open the Eco-Loop production dashboard](https://eco-loop-honeywell-tarun.pnvtarun.chatgpt.site)**

It presents the checked-in 24-hour comparison, comfort trade-off, safety
bounds, and downloadable JSON/CSV evidence. The deployed dashboard does not
control a live building or mutate simulation state.

## Final architecture

```text
EnergyPlus 26.1 Python API
  |  live zone/facility sensors, every 15 minutes
  v
Phase 1 wrapper -> Phase 2 SensorSnapshot adapter -> live MCP server/client
                                                    |
                                                    v
                                      ScriptedProvider or OllamaToolProvider
                                                    |
                           read -> carbon -> reasoning -> control proposal
                                                    |
                                                    v
                                   schema + state + safety validation
                                      | accepted       | failure/rejected
                                      v                v
                           four-step action hold   deterministic release
                                      \                /
                                       v              v
                               proven Phase 1 EMS actuator writeback
                                               |
                                               v
                                JSONL -> JSON/CSV -> PNG/dashboard
```

The live loop makes one decision every four 15-minute timesteps (one simulated
hour) and holds an accepted action for four timesteps. See the
[system architecture](docs/system_architecture.md) for the detailed data and
failure paths.

## Prerequisites

- Windows with Python 3.12 or newer
- EnergyPlus 26.1.0, including its Python API
- Ollama for the real-provider commands
- The local `llama3.2:3b` model for the reproduced 24-hour comparison
- Enough disk space for EnergyPlus output and the local model

The MCP and ScriptedProvider tests do not require EnergyPlus or Ollama.

## Installation

From PowerShell in the repository root:

```powershell
git clone https://github.com/Tarun1954/honeywell-hackathon-.git
Set-Location honeywell-hackathon-
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item .env.example .env
```

`.env` is ignored by Git. Keep machine-specific paths and provider
configuration there; never commit it.

## EnergyPlus setup

Install EnergyPlus 26.1.0 and set `ENERGYPLUS_HOME` to the directory containing
`energyplus.exe`, `Energy+.idd`, and the Python API. A repository-local install
would use:

```dotenv
ENERGYPLUS_HOME=.tools/EnergyPlus-26.1.0
```

The versioned inputs are:

- [five-zone baseline model](models/baseline.idf)
- [runtime model copy](models/optimized_runtime.idf)
- [Chicago O'Hare TMY3 weather](weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw)
- [Phase 1 configuration](config/phase1.yaml)

The baseline and runtime IDFs are intentionally identical at rest. Control is
applied through live `Zone Temperature Control` actuator handles, not by
rewriting the model.

## Ollama setup

Install Ollama, start its local service, and install the model used by the
verified run:

```powershell
ollama pull llama3.2:3b
ollama list
```

Then configure `.env` without committing or printing it:

```dotenv
PHASE2_LLM_PROVIDER=ollama
PHASE2_LLM_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

`OLLAMA_BASE_URL` is optional and defaults to the loopback URL shown above.
The live Ollama commands perform a startup check and fail clearly when the
provider, model, service, or installed model is unavailable.

## Exact run commands

Run commands from the repository root.

Phase 1 deterministic seven-day baseline/control/repeatability comparison:

```powershell
python -m scripts.run_phase1_smoke --mode both
```

MCP protocol smoke test, which discovers and invokes all five tools:

```powershell
python -m scripts.run_phase2_mcp_smoke
```

Four-hour live EnergyPlus smoke with the deterministic ScriptedProvider:

```powershell
python -m scripts.run_live_scripted_energyplus_smoke
```

Four-hour live EnergyPlus smoke with the configured real Ollama provider:

```powershell
python -u -m scripts.run_live_ollama_energyplus_smoke
```

Matched 24-hour baseline versus real Ollama-supervised comparison:

```powershell
python -u -m scripts.run_ollama_energyplus_comparison
```

Regenerate the deterministic/scripted results aggregation from existing
artifacts:

```powershell
python -m scripts.generate_results_summary
```

## MCP tools

The stdio MCP server exposes exactly five typed tools:

1. `read_sensor_data` returns the current correlated `SensorSnapshot` and up to
   four prior 15-minute snapshots without advancing simulation time.
2. `get_grid_carbon_intensity` returns the current and forecast deterministic
   carbon signal for the current snapshot.
3. `log_reasoning` stores only a concise decision summary, objective tags,
   trade-off summary, and confidence. It does not request or store hidden
   chain-of-thought.
4. `set_control_action` submits one idempotent action containing exactly one
   `set` or `release` command for each of the five configured zones.
5. `parse_runtime_errors` returns a bounded, cycle-scoped page of runtime
   errors when corrective handling is required.

The normal agent order is `read_sensor_data`,
`get_grid_carbon_intensity`, `log_reasoning`, then
`set_control_action`. `parse_runtime_errors` is used only on the error path.
Snapshot, cycle, reasoning, and idempotency identifiers keep every action
correlated and auditable.

## Safety and deterministic fallback

Validation occurs at the MCP action boundary and again immediately before
EnergyPlus writeback. A proposal is rejected rather than silently clamped if
it has an unknown, duplicate, or missing zone; non-finite or out-of-range
values; less than a 1 C heating/cooling deadband; an unsafe occupied-zone
setting; a stale snapshot; a missing reasoning record; too long a hold; or a
pending severe/fatal runtime error.

Configured thermostat bounds are 16-24 C for heating and 20-30 C for cooling.
When occupied, heating must be at least 20 C and cooling at most 26 C. The
configured PMV safety threshold is `abs(PMV) <= 0.7`.

A retryable rejection is returned to the agent as compact correction feedback.
Timeout, malformed output, exhausted correction, transport failure, or any
other provider exception activates the deterministic fallback. That fallback
issues a validated five-zone release, returning control to the original
EnergyPlus schedules. It never invents an unchecked setpoint.

## Quantitative results

### 24-hour real Ollama-supervised hybrid comparison

The matched runs used the same model, weather, 15-minute timestep, run-period
start, and initial snapshot. The controlled case used `llama3.2:3b`, one
decision per simulated hour, a four-timestep hold, mandatory MCP validation,
and deterministic fallback.

| Metric | Baseline | Ollama hybrid |
| --- | ---: | ---: |
| Total electricity | 180.556 kWh | 168.703 kWh |
| Peak demand | 17.258 kW | 15.566 kW |
| Zone-temperature range | 20.63-23.90 C | 20.00-26.00 C |
| PMV range | -1.41 to 0.21 | -1.49 to 0.67 |
| Occupied PMV compliance | 96.82% | 95.45% |
| Occupied comfort violations | 7 | 10 |
| EnergyPlus severe/fatal errors | 0/0 | 0/0 |

The Ollama hybrid used **11.853 kWh less electricity, a 6.565% reduction**.
All 24 hourly real-model actions were accepted; 0 were rejected, 0 required
correction, and 0 used fallback. There were 460 successful zone-actuator
writes. Median model latency was 19.457 seconds and maximum latency was
28.487 seconds.

The Ollama-supervised run reduced energy and peak demand, while occupied PMV
compliance decreased slightly from 96.82% to 95.45%, with comfort violations
increasing from 7 to 10.

Machine-readable evidence is in
[JSON](runs/final/ollama_24h_comparison.json) and
[CSV](runs/final/ollama_24h_comparison.csv). The narrative report is
[here](docs/ollama_24h_comparison.md), and the
[static results dashboard](docs/dashboard/index.html) presents the same
verified values.

### Seven-day deterministic-controller comparison

This is a separate Phase 1 result and **was not produced by Ollama**.

| Metric | Baseline | Deterministic control |
| --- | ---: | ---: |
| Total electricity | 1,024.291 kWh | 955.179 kWh |
| Absolute reduction | - | 69.112 kWh |
| Percentage reduction | - | 6.747% |
| Duration | 168 hours / 672 timesteps | 168 hours / 672 timesteps |

The deterministic policy applies fixed occupied-hours thermostat values. Its
6.747% result proves repeatable EnergyPlus control and is not an LLM savings
claim.

### Four-hour ScriptedProvider smoke

The ScriptedProvider smoke is deterministic integration evidence, not a real
LLM energy comparison. It completed 16 timesteps, accepted four hourly
actions, used no fallbacks, made 16 MCP calls, and recorded 60 successful
zone-actuator writes. Because there was no matched four-hour baseline, no
energy-savings claim is made for this smoke.

See [current results](docs/current_results.md) for the evidence classes and
source files.

## Repository structure

| Path | Purpose |
| --- | --- |
| `config/` | Phase 1 EnergyPlus and Phase 2 safety/provider configuration |
| `models/` | Baseline, runtime, and upstream EnergyPlus IDFs |
| `weather/` | Versioned TMY3 weather input |
| `src/energyplus_wrapper.py` | Proven EnergyPlus API lifecycle, sensors, and actuators |
| `src/live_energyplus_integration.py` | Snapshot adapter, hourly scheduling, hold, fallback, and JSONL evidence |
| `src/mcp_server.py`, `src/mcp_client.py` | Typed MCP tool boundary and stdio client |
| `src/phase2_agent.py` | Bounded tool-calling agent and correction loop |
| `src/ollama_provider.py` | Local Ollama tool provider and response normalization |
| `src/phase2_validation.py` | Configuration-driven action safety rules |
| `scripts/` | Reproducible smoke, comparison, and aggregation commands |
| `tests/` | Unit, mocked-provider, MCP, integration, results, and documentation checks |
| `runs/final/` | Tracked JSON/CSV evidence and PNG charts |
| `docs/dashboard/` | Offline static results dashboard |

## Limitations and future improvements

- The real-provider comparison covers one 24-hour summer period in one
  five-zone reference building; it does not establish annual or
  cross-building performance.
- Comfort degraded slightly in the controlled case, so future policies should
  optimize an explicit energy/comfort objective and be evaluated over more
  weather periods.
- Grid-carbon data is deterministic test data rather than a live utility or
  ISO feed.
- The proven actuator surface is limited to five heating/cooling thermostat
  pairs. Ventilation, lighting, storage, and equipment controls are out of
  scope.
- The local 3B model has non-trivial latency. Future work can test caching,
  smaller prompts, faster local hardware, and asynchronous supervisory
  strategies without weakening safety checks.
- A real deployment needs building-automation authentication, permissions,
  monitoring, operator override, rollback procedures, and staged field
  validation.

The [final deliverables checklist](docs/final_deliverables.md) identifies the
completed repository artifacts and honestly marks the video and presentation
as pending.
