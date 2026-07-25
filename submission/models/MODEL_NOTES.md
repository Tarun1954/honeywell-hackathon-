# Building model notes

## Submitted model inventory

| Submission file | Repository source | SHA-256 | Meaning |
| --- | --- | --- | --- |
| `upstream_reference_building.idf` | `models/upstream/5ZoneAirCooled-v26.1.0.idf` | `0187CF7F2CA9C27C43D435A68A8C66A557A43678846813A7E21463A0B0C716CD` | Original EnergyPlus 26.1 five-zone reference model retained for provenance |
| `baseline_building.idf` | `models/baseline.idf` | `4A18BAA3FCDAB2A3968D3AD46FEAF5AED63DF93D4ED1FB2F4610D9827F3E36D2` | Instrumented baseline and exact source IDF used by both matched 24-hour runs |
| `api_ready_runtime.idf` | `models/optimized_runtime.idf` | `4A18BAA3FCDAB2A3968D3AD46FEAF5AED63DF93D4ED1FB2F4610D9827F3E36D2` | Phase 1 runtime copy used for the deterministic actuated/repeatability pipeline |

The weather input is committed separately at
`weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` with SHA-256
`C7D4EFCF93BA316A1D874352E743DF5CF137BA5C0E3459EB2DC4B5442D5B7F5C`.

## Why there is no `controlled_building.idf`

There is no permanently modified controlled IDF, so this submission does not
fabricate or mislabel one. `baseline_building.idf` and
`api_ready_runtime.idf` are byte-identical. The control changes exist only
during simulation: the Python EnergyPlus API writes validated heating and
cooling setpoints to ten proven `Zone Temperature Control` actuator handles
(one pair in each of five zones).

The matched 24-hour comparison explicitly passes the same
`models/baseline.idf` to both EnergyPlus runs:

- the baseline run supplies no control policy, leaving the original
  EnergyPlus schedules in control;
- the controlled run supplies the live Ollama/MCP policy, validates every
  five-zone action, and applies accepted setpoints through API actuators;
- provider failure uses a validated deterministic release, which resets the
  actuators to the original EnergyPlus schedules.

This design makes the comparison a runtime-control experiment, not an
IDF-to-IDF comparison. The evidence is recorded in
`runs/final/ollama_24h_comparison.json` under `matched_conditions`.

## Runtime-control implementation

| File | Responsibility |
| --- | --- |
| `src/energyplus_wrapper.py` | Resolves the heating/cooling actuator handles and calls `set_actuator_value` or `reset_actuator` at `BeginSystemTimestepBeforePredictor` |
| `src/live_energyplus_integration.py` | Maps live sensors, schedules hourly decisions, holds accepted actions for four timesteps, revalidates before writeback, and records JSONL evidence |
| `src/mcp_server.py` and `src/mcp_client.py` | Provide the typed MCP boundary used by the live agent |
| `src/phase2_agent.py` | Enforces the tool order, bounded correction flow, and safe fallback |
| `src/ollama_provider.py` | Converts local Ollama responses into normalized tool calls |
| `src/phase2_validation.py` | Rejects incomplete, stale, out-of-range, or otherwise unsafe actions without clamping |
| `scripts/run_phase1_smoke.py` | Copies `models/baseline.idf` to `models/optimized_runtime.idf` before the deterministic actuated run |
| `scripts/run_ollama_energyplus_comparison.py` | Runs the matched baseline and live Ollama-supervised cases from the same baseline IDF |

## Reproducing the runtime-controlled version

From the repository root:

1. Install EnergyPlus 26.1.0 and the Python dependencies described in
   `README.md`.
2. Copy `.env.example` to `.env`, set `ENERGYPLUS_HOME`, and, for the real
   provider run, set `PHASE2_LLM_PROVIDER=ollama` and
   `PHASE2_LLM_MODEL=llama3.2:3b`.
3. To reproduce the deterministic Phase 1 runtime copy and comparison, run:

   ```powershell
   python -m scripts.run_phase1_smoke --mode both
   ```

4. To reproduce the matched 24-hour real-provider comparison, run:

   ```powershell
   python -u -m scripts.run_ollama_energyplus_comparison
   ```

The second command uses `models/baseline.idf` for both cases. Its controlled
state is observable in `timesteps.jsonl`, `live_agent_timesteps.jsonl`, and
actuator-write evidence, not in a modified IDF. The verified, tracked summary
is `runs/final/ollama_24h_comparison.json`.
