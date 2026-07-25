# Eco-Loop submission manifest

This directory gathers the required building-model, result, and documentation
deliverables without presenting runtime API control as a permanently modified
IDF. It intentionally does not contain a final ZIP, video, or presentation.

## Hackathon deliverable mapping

| Required deliverable | Status | Exact repository path |
| --- | --- | --- |
| Functional source code | Complete | `src/`, `scripts/`, `config/`, and `tests/` |
| EnergyPlus Python API wrapper | Complete | `src/energyplus_wrapper.py` |
| MCP server and client | Complete | `src/mcp_server.py` and `src/mcp_client.py` |
| Ollama agent orchestration | Complete | `src/ollama_provider.py`, `src/phase2_agent.py`, and `src/live_energyplus_integration.py` |
| Original reference building model | Complete | `submission/models/upstream_reference_building.idf` |
| Matched-run baseline building model | Complete | `submission/models/baseline_building.idf` |
| Modified/controlled building model | Runtime-controlled; no permanent modified IDF exists | `submission/models/MODEL_NOTES.md` |
| API-ready runtime IDF actually produced by Phase 1 | Complete | `submission/models/api_ready_runtime.idf` |
| Weather input | Complete | `weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw` |
| Quantitative savings dashboard | Complete | `submission/dashboard/index.html` |
| Real Ollama JSON and CSV evidence | Complete | `submission/results/ollama_24h_comparison.json` and `submission/results/ollama_24h_comparison.csv` |
| Deterministic/scripted JSON and CSV evidence | Complete and separately labeled | `submission/results/results_summary.json` and `submission/results/results_summary.csv` |
| PNG charts | Complete | `submission/results/ollama_24h_energy_peak.png`, `submission/results/ollama_24h_comfort.png`, `submission/results/ollama_24h_actions_latency.png`, `submission/results/energy_comparison.png`, `submission/results/zone_temperature_pmv.png`, and `submission/results/action_setpoint_evidence.png` |
| Architecture document | Complete | `submission/docs/system_architecture.md` |
| Quantitative results documents | Complete | `submission/docs/current_results.md`, `submission/docs/ollama_24h_comparison.md`, and `submission/docs/phase1_report.md` |
| Demo video | Pending placeholder only | `submission/video/VIDEO_PENDING.md` |
| Presentation | Pending placeholder only | `submission/presentation/PRESENTATION_PENDING.md` |
| GitHub repository | Available | `https://github.com/Tarun1954/honeywell-hackathon-` |

## Model statement

Both matched 24-hour cases used the same
`models/baseline.idf`. The controlled case was created by live, validated
EnergyPlus API actuator overrides, not by changing IDF objects. Accordingly,
there is no `controlled_building.idf`. The exact model hashes, actuator files,
and reproduction commands are recorded in
`submission/models/MODEL_NOTES.md`.

## Results statement

The `24-hour real Ollama-supervised hybrid comparison` reduced electricity
from 180.556 kWh to 168.703 kWh, a reduction of 11.853 kWh or 6.565%.
Baseline and controlled peak demand were 17.258 kW and 15.566 kW. Occupied PMV
compliance decreased from 96.82% to 95.45%, and comfort violations increased
from 7 to 10.

The separate seven-day 6.747% result was produced by the deterministic Phase 1
controller and is not claimed as an Ollama result.

## Repository hygiene

- `.env` is excluded by `.gitignore`; only the empty `.env.example` template
  is tracked.
- `.tools/`, Python caches, test caches, runtime `runs/`, logs, partial
  downloads, and common local-model formats are excluded.
- No Ollama model weights are included. Ollama stores its models outside this
  repository.
- Submission checks scan tracked text for high-confidence API-key, token, and
  private-key signatures.
- The tracked submission contains verified aggregate evidence, not the large
  raw EnergyPlus run directories.

## Packaging status

Do not create the final ZIP yet. The video and presentation are still pending,
and their placeholders must be replaced before final packaging.
